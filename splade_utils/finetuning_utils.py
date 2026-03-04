import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from sentence_transformers import CrossEncoder
from sklearn.model_selection import train_test_split
import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from beir.datasets.data_loader import GenericDataLoader

# Import inference utils
from splade_utils.splade_utils import run_pipeline, run_hybrid_pipeline, SpladeModel, InvertedIndex

# CLASS 1: TRAINABLE SPLADE MODEL
class TrainableSpladeModel(nn.Module):
    """
    Wrapper around HuggingFace MaskedLM (e.g., DistilBERT/BERT) to enable SPLADE training.
    Unlike the inference-only class, this class maintains gradients for backpropagation.
    """
    def __init__(self, model_name):
        super().__init__()
        # Load the pre-trained backbone (e.g., naver/splade-cocondenser-selfdistil)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Enable Gradient Checkpointing. This trades compute speed for memory.
        # It re-computes parts of the graph during backward pass instead of storing them,
        # allowing us to fit larger batches or longer sequences in limited GPU memory.
        self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        """
        Standard SPLADE forward pass: BERT -> Logits -> ReLU -> MaxPool -> Log1p
        """
        # 1. Get raw logits from BERT (Batch, Seq_Len, Vocab_Size)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits 
        
        # 2. Apply SPLADE transformation (Equation 1 in the paper)
        # ReLU ensures sparsity (negative values become 0)
        values = torch.relu(logits)
        
        # 3. Max Pooling over the sequence dimension
        # We multiply by attention_mask to ignore padding tokens (setting them to 0)
        # Result shape: (Batch, Vocab_Size) - A sparse vector for the whole document/query
        values, _ = torch.max(values * attention_mask.unsqueeze(-1), dim=1)
        
        # 4. Log Saturation: Compresses high values to stabilize training
        return torch.log(1 + values)

    def save_pretrained(self, path):
        """Helper to save both model and tokenizer to disk."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


# CLASS 2: DATASET MANAGEMENT
class TripletsDataset(Dataset):
    """
    PyTorch Dataset that handles the specific input format for MarginMSE Distillation.
    Each item is a quadruplet: (Query, Positive Doc, Hard Negative Doc, Margin Score).
    """
    def __init__(self, triplets, tokenizer, max_len=256):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        # Unpack the data
        query, pos_text, neg_text, margin_score = self.triplets[idx]
        
        # Tokenize Query
        q_enc = self.tokenizer(query, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        # Tokenize Positive Document
        p_enc = self.tokenizer(pos_text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        # Tokenize Negative Document
        n_enc = self.tokenizer(neg_text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")

        # Return a dictionary compatible with the model's forward pass
        return {
            "q_ids": q_enc["input_ids"].squeeze(0),
            "q_att": q_enc["attention_mask"].squeeze(0),
            "p_ids": p_enc["input_ids"].squeeze(0),
            "p_att": p_enc["attention_mask"].squeeze(0),
            "n_ids": n_enc["input_ids"].squeeze(0),
            "n_att": n_enc["attention_mask"].squeeze(0),
            "margin": torch.tensor(margin_score, dtype=torch.float)
        }


# PHASE 1: OPTIMIZED MINING & SCORING
def mine_and_score_negatives(dataset_name, model_name, top_k_negatives=1, batch_size_mining=32):
    """
    Executes the 'Self-Mining' strategy described in the paper.
    
    Workflow:
    1. Load the current SPLADE model (Student).
    2. Encode the entire Corpus + Training Queries.
    3. Perform retrieval to find 'Hard Negatives' (documents that the student
       incorrectly thinks are relevant).
    4. Use a Cross-Encoder (Teacher) to score the pairs (Query, Pos) and (Query, Neg).
    5. Calculate the 'Margin' (Teacher Score Pos - Teacher Score Neg) for distillation.
    
    Optimization:
    - Loads the SPLADE model only once for both corpus and query encoding.
    - Uses a larger batch size for the Cross-Encoder inference.
    """
    print(f"\n{'='*60}")
    print(f"PHASE 1: SELF-MINING HARD NEGATIVES & DISTILLATION")
    print(f"{'='*60}")

    # Step 1: Load Data
    data_path = f"data/{dataset_name}"
    print(f"Loading training set from {data_path}...")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="train")
    
    # Step 2: Initialize SPLADE Model (Used for mining)
    # We initialize it here to avoid reloading it multiple times (Optimization)
    miner_model = SpladeModel(model_name)
    
    # Step 3: Encode Corpus
    print("Indexing corpus for mining...")
    doc_ids = list(corpus.keys())
    # Concatenate title + text for better representation
    texts = [corpus[did].get("title", "") + " " + corpus[did].get("text", "") for did in doc_ids]
    
    # Use the inference wrapper to get sparse vectors efficiently
    sparse_vecs = miner_model.encode(texts, batch_size=batch_size_mining, desc="Encoding Corpus")
    
    # Build an Inverted Index for fast retrieval
    index = InvertedIndex()
    index.add_documents(doc_ids, sparse_vecs)
    
    # Free memory (we don't need raw texts anymore)
    del texts, sparse_vecs
    gc.collect()

    # Step 4: Encode Queries
    print("Encoding queries for mining...")
    q_list = list(queries.keys())
    q_texts = [queries[qid] for qid in q_list]
    
    # Reuse the same model instance!
    q_vecs = miner_model.encode(q_texts, batch_size=batch_size_mining, desc="Encoding Queries")
    
    # Now we can delete the mining model to free VRAM for the next steps
    del miner_model
    torch.cuda.empty_cache()
    gc.collect()

    # Step 5: Mine Hard Negatives
    print("Mining negatives (searching for mistakes)...")
    triplets_data = [] 
    
    for qid, q_vec in tqdm(zip(q_list, q_vecs), total=len(q_list), desc="Mining"):
        if qid not in qrels: continue
        
        # Get the ground truth positive documents
        pos_doc_ids = list(qrels[qid].keys())
        if not pos_doc_ids: continue
        
        # Retrieve top-10 documents using the student model
        retrieved_docs, _ = index.search(q_vec, k=10)
        retrieved_ids = list(retrieved_docs.keys())
        
        # Identify Hard Negatives: Retrieved docs that are NOT in the ground truth
        hard_negatives = [did for did in retrieved_ids if did not in pos_doc_ids]
        
        if not hard_negatives: continue
            
        # Create Triplets (Query, Pos, Neg)
        query_text = queries[qid]
        for pos_id in pos_doc_ids:
            if pos_id not in corpus: continue
            
            # Select top-k hardest negatives (usually just 1 for efficiency)
            selected_negs = hard_negatives[:top_k_negatives]
            
            for neg_id in selected_negs:
                pos_text = corpus[pos_id].get("title", "") + " " + corpus[pos_id].get("text", "")
                neg_text = corpus[neg_id].get("title", "") + " " + corpus[neg_id].get("text", "")
                triplets_data.append((query_text, pos_text, neg_text))

    # Free memory before loading the Teacher
    del index, q_vecs
    gc.collect()
    
    # Step 6: Teacher Scoring (Distillation)
    print(f"Scoring {len(triplets_data)} triplets with Cross-Encoder Teacher...")
    
    # Load the Teacher Model (Cross-Encoder is slow but accurate)
    teacher_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    teacher = CrossEncoder(teacher_name)
    
    # Prepare pairs for the Cross-Encoder: (Query, Doc)
    pairs_pos = [(t[0], t[1]) for t in triplets_data]
    pairs_neg = [(t[0], t[2]) for t in triplets_data]
    
    # Batch Prediction for Positives and Negatives
    batch_size_teacher = 64
    scores_pos = teacher.predict(pairs_pos, batch_size=batch_size_teacher, show_progress_bar=True)
    scores_neg = teacher.predict(pairs_neg, batch_size=batch_size_teacher, show_progress_bar=True)
    
    # Calculate the Margin for each triplet
    # Margin = Score(Positive) - Score(Negative)
    # The student model will try to predict this margin value.
    final_training_data = []
    for i, (q, p, n) in enumerate(triplets_data):
        margin = scores_pos[i] - scores_neg[i]
        final_training_data.append((q, p, n, margin))
        
    # Final Cleanup
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    
    return final_training_data


# PHASE 2: TRAINING (WITH VALIDATION & EARLY STOPPING)
def compute_flops_loss(vec, lambda_val):
    """
    Computes the FLOPS Regularization Loss (Sparsity Constraint).
    Formula from SPLADE paper: lambda * sum(mean_activation^2)
    
    Args:
        vec: The sparse vector output from SPLADE (Batch, Vocab). 
             Note: These values are already log(1+ReLU), so they are non-negative.
        lambda_val: The regularization strength (higher = sparser).
    """
    # 1. Calculate the mean activation for each term across the batch
    # (dim=0 is the batch dimension)
    avg_activation = torch.mean(vec, dim=0) 
    
    # 2. Square the means and sum them up
    # This penalizes terms that are frequently active across different documents
    return lambda_val * torch.sum(avg_activation ** 2)

def evaluate_validation(model, val_loader, device, lambda_q, lambda_d):
    """
    Runs evaluation on the Validation Set to check for overfitting.
    Computes the same loss used in training but without Backpropagation.
    """
    model.eval() # Set model to evaluation mode (disable dropout, etc.)
    total_val_loss = 0
    
    with torch.no_grad(): # Disable gradient calculation to save memory
        for batch in val_loader:
            # Move data to GPU
            q_ids, q_att = batch["q_ids"].to(device), batch["q_att"].to(device)
            p_ids, p_att = batch["p_ids"].to(device), batch["p_att"].to(device)
            n_ids, n_att = batch["n_ids"].to(device), batch["n_att"].to(device)
            teacher_margin = batch["margin"].to(device)
            
            # Forward pass with Mixed Precision
            with torch.amp.autocast('cuda'):
                q_vec = model(q_ids, q_att)
                p_vec = model(p_ids, p_att)
                n_vec = model(n_ids, n_att)
                
                # Compute Student Scores (Dot Product)
                pos_score = torch.sum(q_vec * p_vec, dim=1)
                neg_score = torch.sum(q_vec * n_vec, dim=1)
                student_margin = pos_score - neg_score
                
                # Compute Loss components
                loss_distil = nn.MSELoss()(student_margin, teacher_margin)
                loss_flops_q = compute_flops_loss(q_vec, lambda_q)
                loss_flops_d = compute_flops_loss(p_vec, lambda_d) + compute_flops_loss(n_vec, lambda_d)
                
                total_val_loss += (loss_distil + loss_flops_q + loss_flops_d).item()
    
    return total_val_loss / len(val_loader)

def train_splade_model(
    train_triplets,
    base_model_name,
    output_path,
    lambda_q=0.008,
    lambda_d=0.008,
    epochs=5,
    batch_size=2,
    grad_accum_steps=16,
    lr=2e-5,
    patience=3 # Number of epochs to wait before Early Stopping
):
    """
    Main Training Loop for SPLADE Fine-tuning.
    Includes: Mixed Precision, Gradient Accumulation, Validation, Early Stopping.
    """
    print(f"\n{'='*60}")
    print(f"PHASE 2: TRAINING (Lambda_d={lambda_d})")
    print(f"{'='*60}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Data Splitting (Train vs Validation)
    # We use 10% of the data for validation to monitor overfitting
    train_data, val_data = train_test_split(train_triplets, test_size=0.1, random_state=42)
    print(f"Training on {len(train_data)} samples, Validating on {len(val_data)} samples.")
    
    # 2. Model & Dataloaders Setup
    model = TrainableSpladeModel(base_model_name)
    model.to(device)
    
    train_dataset = TripletsDataset(train_data, model.tokenizer)
    val_dataset = TripletsDataset(val_data, model.tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)
    
    # 3. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    
    # Total optimization steps = (batches / accum_steps) * epochs
    total_steps = len(train_loader) * epochs // grad_accum_steps
    
    # Linear warmup for the first 10% of steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    
    # Scaler for Mixed Precision Training (Updated syntax for PyTorch > 2.0)
    scaler = torch.amp.GradScaler('cuda') 
    
    # 4. Training Loop
    global_step = 0
    best_val_loss = float('inf')
    patience_counter = 0 # Tracks how many epochs without improvement
    
    training_history = [] 

    for epoch in range(epochs):
        model.train() # Set to training mode
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # Move batch to GPU
            q_ids, q_att = batch["q_ids"].to(device), batch["q_att"].to(device)
            p_ids, p_att = batch["p_ids"].to(device), batch["p_att"].to(device)
            n_ids, n_att = batch["n_ids"].to(device), batch["n_att"].to(device)
            teacher_margin = batch["margin"].to(device)
            
            # Forward Pass with Mixed Precision
            with torch.amp.autocast('cuda'): 
                q_vec = model(q_ids, q_att)
                p_vec = model(p_ids, p_att)
                n_vec = model(n_ids, n_att)
                
                # Compute scores
                pos_score = torch.sum(q_vec * p_vec, dim=1)
                neg_score = torch.sum(q_vec * n_vec, dim=1)
                student_margin = pos_score - neg_score
                
                # Loss Calculation (from Paper)
                # 1. Distillation: MSE(Student Margin, Teacher Margin)
                loss_distil = nn.MSELoss()(student_margin, teacher_margin)
                
                # 2. Regularization: Penalize non-zero activations (FLOPS)
                loss_flops_q = compute_flops_loss(q_vec, lambda_q)
                loss_flops_d = compute_flops_loss(p_vec, lambda_d) + compute_flops_loss(n_vec, lambda_d)
                
                total_loss = loss_distil + loss_flops_q + loss_flops_d
                
                # Normalize loss by accumulation steps
                total_loss = total_loss / grad_accum_steps

            # Backward Pass (Scaled)
            scaler.scale(total_loss).backward()
            epoch_loss += total_loss.item()
            
            # Optimizer Step (Gradient Accumulation)
            if (step + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                progress_bar.set_postfix({"Loss": f"{total_loss.item() * grad_accum_steps:.4f}"})

        # CALCOLO MEDIA LOSS (Training)
        # Moltiplichiamo per grad_accum_steps per avere la loss "reale" non scalata, confrontabile con la validation
        avg_train_loss = (epoch_loss * grad_accum_steps) / len(train_loader) 
        
        # VALIDATION PHASE (End of Epoch)
        val_loss = evaluate_validation(model, val_loader, device, lambda_q, lambda_d)
        print(f"Epoch {epoch+1} finished. Val Loss: {val_loss:.4f}")

        # Salvo i dati nella storia per il grafico
        training_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss
        })
        
        # EARLY STOPPING LOGIC
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0 # Reset counter
            
            # Save the best model so far
            print(f"New best model found! Saving to {output_path}")
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            model.save_pretrained(output_path)
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered to prevent overfitting.")
                break
    
    print(f"Training Complete. Best model available at {output_path}")
    
    # Cleanup memory
    del model, optimizer, scaler, train_dataset, val_dataset
    gc.collect()
    torch.cuda.empty_cache()
    
    return output_path, training_history # returns the path and training history


# PHASE 3: EVALUATION WRAPPER
def evaluate_model_on_dataset(dataset_name, model_path, output_folder, batch_size=16):
    """
    Loads the trained model and runs the full inference pipeline (Encode -> Index -> Search).
    """
    print(f"\n{'='*60}")
    print(f"PHASE 3: EVALUATION ON {dataset_name.upper()}")
    print(f"Loading Model from: {model_path}")
    print(f"{'='*60}")
    
    # We rely on the optimized hybrid pipeline from splade_utils
    # This ensures consistent evaluation metrics (NDCG, FLOPS, etc.)
    stats = run_hybrid_pipeline(
        dataset_name=dataset_name,
        model_name=model_path,
        output_folder=output_folder,
        batch_size=batch_size
    )
    
    return stats