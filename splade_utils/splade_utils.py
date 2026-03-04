import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from splade_utils.bm25_utils import run_bm25_pipeline

# CLASS 1: SPLADE ENCODER
class SpladeModel:
    """
    SPLADE Inference Wrapper.
    Loads a Hugging Face Masked Language Model and overrides the output 
    to generate sparse vectors using the SPLADE formula (Max Pooling).
    """
    def __init__(self, model_name, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Model: {model_name} on {self.device}...")
        
        # Load Tokenizer and Model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts, batch_size=32, desc="Encoding"):
        """
        Computes sparse representations for a list of texts.
        """
        all_sparse_vecs = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch_texts = texts[i:i+batch_size]
            
            with torch.no_grad():
                # 1. Tokenization
                inputs = self.tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                ).to(self.device)
                
                # 2. Model Inference
                logits = self.model(**inputs).logits 
                
                # 3. SPLADE Formula: log(1 + ReLU(max(logits))) -> Max Pooling with ReLU, then Log Saturation
                # A. ReLU
                values = torch.relu(logits)
                
                # B. Max Pooling (as per SPLADE++ paper)
                # Multiply by attention mask to ignore padding
                values, _ = torch.max(values * inputs["attention_mask"].unsqueeze(-1), dim=1)
                
                # C. Log Saturation
                values = torch.log(1 + values)
                
                # 4. Conversion to Sparse Dictionary
                values_np = values.cpu().numpy()
                
                for row in values_np:
                    # Get indices where weight > 0
                    indices = np.nonzero(row)[0]
                    # Create dict {token_id: weight}
                    all_sparse_vecs.append(dict(zip(indices, row[indices])))
                    
        return all_sparse_vecs


# CLASS 2: INVERTED INDEX
class InvertedIndex:
    """
    Python-based Inverted Index for Sparse Retrieval.
    """
    def __init__(self):
        self.index = defaultdict(dict)
        self.doc_sparsity = []

    def add_documents(self, doc_ids, sparse_vecs): # Add documents to the inverted index
        for doc_id, vector in zip(doc_ids, sparse_vecs): # Iterate over documents
            self.doc_sparsity.append(len(vector)) 
            for token_id, weight in vector.items(): # Iterate over non-zero tokens
                self.index[token_id][doc_id] = weight # Posting list

    def search(self, query_vector, k=10): # Top-k retrieval
        scores = defaultdict(float)
        flops_count = 0 
        
        for token_id, q_weight in query_vector.items(): # Iterate over query tokens
            if token_id in self.index:
                posting_list = self.index[token_id] # Get posting list
                flops_count += len(posting_list)
                
                for doc_id, d_weight in posting_list.items(): # Iterate over documents containing the token
                    scores[doc_id] += q_weight * d_weight
        
        # Sort and return top-k
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k] # Top-k
        return dict(sorted_scores), flops_count


# PIPELINE FUNCTION
def run_pipeline(dataset_name, model_name, output_folder, batch_size=32, cpu_chunk_size=10000):
    """
    Executes the full retrieval pipeline: Load -> Encode -> Index -> Search -> Eval.
    OPTIMIZED: Uses chunking (cpu_chunk_size) to prevent RAM saturation on large datasets.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"\n{'='*60}")
    print(f"STARTING PIPELINE: {dataset_name.upper()} | Model: {model_name}")
    print(f"{'='*60}")

    # 1. LOAD DATA
    data_path = f"data/{dataset_name}"
    print(f"[1/6] Loading data from {data_path}...")
    try:
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return None

    # 2. LOAD MODEL
    try:
        splade = SpladeModel(model_name, device=device)
    except Exception as e:
        print(f"Model loading failed: {e}")
        return None
    
    # 3 and 4. ENCODE & INDEX IN STREAMING
    print(f"[3-4/6] Encoding & Indexing {len(corpus)} documents (Chunked)...")
    inv_index = InvertedIndex()
    
    doc_ids = list(corpus.keys())
    
    # Outer loop: Process chunks to save System RAM
    for i in range(0, len(doc_ids), cpu_chunk_size):
        chunk_ids = doc_ids[i : i + cpu_chunk_size]
        
        # Extract text only for this chunk
        chunk_texts = [
            (corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip() 
            for d in chunk_ids
        ]
        
        # Encode chunk (GPU uses small batch_size)
        chunk_vectors = splade.encode(chunk_texts, batch_size=batch_size, desc=f"Chunk {i//cpu_chunk_size + 1}")
        
        # Add to index
        inv_index.add_documents(chunk_ids, chunk_vectors)
        
        # FREE MEMORY IMMEDIATELY
        del chunk_texts, chunk_vectors
        gc.collect()
        
    avg_sparsity = np.mean(inv_index.doc_sparsity) # Average number of non-zero entries per document
    print(f"Index built. Avg Sparsity: {avg_sparsity:.2f}")

    total_docs = len(doc_ids)

    # FREE MEMORY BEFORE RETRIEVAL
    del corpus
    del doc_ids
    gc.collect()
    print(f"RAM freed (tracked {total_docs} docs). Starting CPU Retrieval.")

    # 5. RETRIEVAL
    print(f"[5/6] Searching {len(queries)} queries...")
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    
    # Encode queries
    q_vectors = splade.encode(query_texts, batch_size=batch_size, desc="Query Encoding")
    
    results = {}
    total_flops = 0
    
    # Search
    for q_vec, q_id in tqdm(zip(q_vectors, query_ids), total=len(query_ids), desc="Retrieval"):
        top_k, flops = inv_index.search(q_vec, k=1000)
        results[q_id] = top_k
        total_flops += flops
        
    avg_flops = total_flops / (len(query_ids) * total_docs)

    # 6. EVALUATION
    print(f"[6/6] Evaluating...")
    retriever = EvaluateRetrieval(None, score_function="dot")
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values=[10, 100, 1000])
    mrr = retriever.evaluate_custom(qrels, results, k_values=[10], metric="mrr")
    
    stats = {
        "Model": model_name,
        "Dataset": dataset_name,
        "nDCG@10": ndcg['NDCG@10'],
        "MRR@10": mrr['MRR@10'],
        "P@10": precision['P@10'],        
        "R@100": recall['Recall@100'],
        "R@1k": recall['Recall@1000'],
        "Avg_FLOPS": avg_flops,
        "Avg_Doc_Sparsity": avg_sparsity
    }
    
    # DISPLAY RESULTS
    print(f"\n{'*' * 40}")
    print(f"RESULTS FOR {dataset_name.upper()}")
    print(f"{'*' * 40}")
    print(f"nDCG@10: {stats['nDCG@10']:.4f}")
    print(f"MRR@10:  {stats['MRR@10']:.4f}")
    print(f"P@10:    {stats['P@10']:.4f}")
    print(f"R@100:   {stats['R@100']:.4f}")
    print(f"R@1k:    {stats['R@1k']:.4f}")
    print(f"Sparsity:{stats['Avg_Doc_Sparsity']:.1f}")
    print(f"FLOPS:   {stats['Avg_FLOPS']:.1f}")
    print(f"{'*' * 40}\n")

    # Save individual dataset results
    pd.DataFrame([stats]).to_csv(f"{output_folder}/{dataset_name}_results.csv", index=False)
    
    # Final cleanup
    del splade, inv_index, queries, qrels, q_vectors
    torch.cuda.empty_cache()
    gc.collect()
    
    return stats, results


# FUSION FUNCTIONS

def normalize_scores(results):
    """
    Min-Max Normalization per query.
    Essential because BM25 (e.g., 15-40) and SPLADE (e.g., dot product) have different scales.
    """
    normalized_results = {}
    for qid, doc_scores in results.items():
        if not doc_scores:
            continue
        scores = list(doc_scores.values())
        min_score, max_score = min(scores), max(scores)
        
        # Avoid division by zero
        if max_score == min_score:
            normalized_results[qid] = {doc: 1.0 for doc in doc_scores}
        else:
            denominator = max_score - min_score
            normalized_results[qid] = {
                doc: (score - min_score) / denominator 
                for doc, score in doc_scores.items()
            }
    return normalized_results


def apply_fusion(results_splade, results_bm25, alpha=0.5):
    """
    Combines results: Score = alpha * SPLADE + (1-alpha) * BM25
    """
    print("Normalizing scores for fusion...")
    norm_splade = normalize_scores(results_splade)
    norm_bm25 = normalize_scores(results_bm25)
    
    hybrid_results = {}
    all_qids = set(norm_splade.keys()) | set(norm_bm25.keys())
    
    for qid in tqdm(all_qids, desc="Fusing Scores"):
        hybrid_results[qid] = {}
        # Get {doc_id: score} dictionaries for the current query
        docs_s = norm_splade.get(qid, {})
        docs_b = norm_bm25.get(qid, {})
        
        # Union of documents retrieved by both systems
        # Note: if a doc is retrieved by only one, the other score is assumed to be 0
        all_docs = set(docs_s.keys()) | set(docs_b.keys())
        
        for doc_id in all_docs:
            s_score = docs_s.get(doc_id, 0.0)
            b_score = docs_b.get(doc_id, 0.0)
            
            hybrid_results[qid][doc_id] = (alpha * s_score) + ((1 - alpha) * b_score)
            
    return hybrid_results


# HYBRID PIPELINE FUNCTION
def run_hybrid_pipeline(dataset_name, model_name, output_folder, batch_size=32, alpha=0.5):
    """
    Runs SPLADE + BM25 + Fusion.
    Return: Dictionary with ALL SPLADE metrics (FLOPS, Sparsity, etc.) + nDCG@10_Hybrid and MRR@10_Hybrid.
    """
    print(f"\n{'#'*60}")
    print(f"STARTING HYBRID PIPELINE FOR {dataset_name.upper()}")
    print(f"{'#'*60}\n")
    
    # STEP 1: SPLADE
    print("STEP 1: Running SPLADE...")
    # Retrieve stats (all metrics) and results (the documents)
    stats_splade, results_splade = run_pipeline(dataset_name, model_name, output_folder, batch_size)
    
    # Aggressive VRAM cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    # STEP 2: BM25 
    print("STEP 2: Running BM25...")
    # We only need results for fusion, ignore BM25 stats with '_'
    _, results_bm25 = run_bm25_pipeline(dataset_name, output_folder)
    
    # RAM cleanup
    gc.collect()
    
    # STEP 3: FUSION 
    print(f"STEP 3: Computing Hybrid Fusion (alpha={alpha})...")
    
    # Load qrels for evaluation
    _, _, qrels = GenericDataLoader(data_folder=f"data/{dataset_name}").load(split="test")
    
    # Apply fusion formula
    results_hybrid = apply_fusion(results_splade, results_bm25, alpha=alpha)
    
    # STEP 4: HYBRID EVALUATION
    print(f"[Evaluation] Evaluating Hybrid Results...")
    retriever = EvaluateRetrieval(None, score_function="dot")
    
    # Calculate nDCG (interested in @10)
    ndcg, _, _, _ = retriever.evaluate(qrels, results_hybrid, k_values=[10])
    # Calculate MRR (interested in @10)
    mrr = retriever.evaluate_custom(qrels, results_hybrid, k_values=[10], metric="mrr")
    
    # STEP 5: OUTPUT CONSTRUCTION
    
    # 1. Take all original SPLADE metrics (FLOPS, Sparsity, R@1k, etc.)
    final_stats = stats_splade.copy()
    
    # 2. Add ONLY the required hybrid metrics
    final_stats["nDCG@10_Hybrid"] = ndcg['NDCG@10']
    final_stats["MRR@10_Hybrid"] = mrr['MRR@10']
    
    # STEP 6: PRINT HYBRID ONLY
    print(f"\n{'*' * 40}")
    print(f"HYBRID RESULTS (SPLADE + BM25) FOR {dataset_name.upper()}")
    print(f"{'*' * 40}")
    print(f"Hybrid nDCG@10: {final_stats['nDCG@10_Hybrid']:.4f}")
    print(f"Hybrid MRR@10:  {final_stats['MRR@10_Hybrid']:.4f}")
    print(f"{'*' * 40}\n")
    
    # Save to CSV (recommend using a different name to distinguish from base run)
    pd.DataFrame([final_stats]).to_csv(f"{output_folder}/{dataset_name}_results.csv", index=False)
    
    # Final cleanup of heavy variables
    del results_splade, results_bm25, results_hybrid, qrels
    gc.collect()
    
    return final_stats