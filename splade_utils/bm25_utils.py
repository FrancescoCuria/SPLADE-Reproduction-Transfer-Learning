import os
import gc
import string
import multiprocessing
import pandas as pd
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


# PRE-PROCESSOR CLASS
class BM25PreProcessor:
    def __init__(self):
        # Initialize Porter Stemmer (Crucial for BM25 performance)
        self.stemmer = PorterStemmer()
        # Translation table to remove punctuation efficiently
        self.translator = str.maketrans('', '', string.punctuation)
        # Load English stopwords set for O(1) lookups
        self.stop_words = set(stopwords.words('english'))

    def process(self, text):
        """
        Tokenizes, lowercases, removes punctuation, and stems the input text.
        """
        # 1. Lowercase and remove punctuation
        clean_text = text.lower().translate(self.translator)
        # 2. Simple split tokenization (faster than nltk.word_tokenize)
        tokens = clean_text.split()
        # 3. Stopword Removal & Stemming (Combined for efficiency)
        # We only stem if the token is NOT a stopword
        return [
            self.stemmer.stem(t) 
            for t in tokens 
            if t not in self.stop_words
        ]
    
# Helper function for multiprocessing (must be defined at module level), out of the class for 
# multiprocessing pickling (pickling means converting to bytes to send to other processes)
def _preprocess_doc(args):
    doc_id, title, text, processor = args
    # Combine title and text
    full_text = (title + " " + text).strip()
    return processor.process(full_text)

# MAIN PIPELINE FUNCTION
def run_bm25_pipeline(dataset_name, output_folder, k_values=[10, 100, 1000]):
    """
    Executes the full BM25 pipeline:
    Load -> Tokenize (Parallel) -> Build Index -> Search -> Evaluate -> Clear RAM.
    """
    print(f"\n{'='*60}")
    print(f"STARTING BM25 PIPELINE: {dataset_name.upper()}")
    print(f"{'='*60}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 1. LOAD DATA
    data_path = f"data/{dataset_name}"
    print(f"[1/5] Loading data from {data_path}...")
    try:
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return None

    # 2. PREPROCESSING & TOKENIZATION (PARALLELIZED)
    print(f"[2/5] Tokenizing corpus with multiprocessing...")
    
    doc_ids = list(corpus.keys())
    processor = BM25PreProcessor()
    
    # Prepare arguments for the worker pool
    # We pass strings directly to avoid excessive memory overhead during pickling
    pool_args_generator = (
        (did, corpus[did].get("title", ""), corpus[did].get("text", ""), processor) 
        for did in doc_ids
    )
    
    # Use ~80% of available CPU cores
    num_workers = max(1, int(os.cpu_count() * 0.8))
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use imap to show a real progress bar
        tokenized_corpus = list(tqdm(
            pool.imap(_preprocess_doc, pool_args_generator, chunksize=250), 
            total=len(doc_ids), 
            desc="Tokenizing"
        ))

    # Free memory of raw corpus immediately
    del corpus, pool_args_generator
    gc.collect()

    # 3. BUILD BM25 INDEX
    print(f"[3/5] Building BM25 Index (rank_bm25)...")
    # BM25Okapi computes idf and doc_len, but needs the tokenized corpus for scoring
    bm25_model = BM25Okapi(tokenized_corpus)
    
    # 4. RETRIEVAL
    print(f"[4/5] Searching {len(queries)} queries...")
    results = {}
    
    for qid, q_text in tqdm(queries.items(), desc="BM25 Search"):
        # Preprocess query (using the same stemmer)
        tokenized_query = processor.process(q_text)
        
        # Get scores
        scores = bm25_model.get_scores(tokenized_query)
        
        # Optimization: Use argpartition to get top-k without sorting the whole array
        top_k = 1000
        if len(scores) > top_k:
            # Get indices of top_k (unsorted)
            idx = np.argpartition(scores, -top_k)[-top_k:]
            # Sort only the top_k
            sorted_idx = idx[np.argsort(scores[idx])[::-1]]
        else:
            sorted_idx = np.argsort(scores)[::-1]
            
        # Build result dictionary {doc_id: score}
        results[qid] = {doc_ids[i]: float(scores[i]) for i in sorted_idx}

    # 5. EVALUATION
    print(f"[5/5] Evaluating...")
    retriever = EvaluateRetrieval(None, score_function="dot")
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values=k_values)
    mrr = retriever.evaluate_custom(qrels, results, k_values=[10], metric="mrr")

    stats = {
        "Model": "BM25",
        "Dataset": dataset_name,
        "nDCG@10": ndcg['NDCG@10'],
        "MRR@10": mrr['MRR@10'],
        "P@10": precision['P@10'],
        "R@100": recall['Recall@100'],
        "R@1k": recall['Recall@1000']
    }

    # PRINT SUMMARY
    print(f"\n{'*' * 40}")
    print(f"BM25 RESULTS FOR {dataset_name.upper()}")
    print(f"{'*' * 40}")
    print(f"nDCG@10: {stats['nDCG@10']:.4f}")
    print(f"MRR@10:  {stats['MRR@10']:.4f}")
    print(f"P@10:    {stats['P@10']:.4f}")
    print(f"R@100:   {stats['R@100']:.4f}")
    print(f"R@1k:    {stats['R@1k']:.4f}")
    print(f"{'*' * 40}\n")

    # Save individual results
    pd.DataFrame([stats]).to_csv(f"{output_folder}/{dataset_name}_bm25_results.csv", index=False)

    # MEMORY CLEANUP
    print("Cleaning memory...")
    del bm25_model
    del tokenized_corpus
    del doc_ids
    del queries
    del qrels
    gc.collect()
    
    return stats, results