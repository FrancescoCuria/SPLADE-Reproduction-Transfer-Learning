import os
import sys
from beir import util

def download_beir_datasets(data_path="data"):
    """
    Downloads and unzips the specific BEIR datasets selected for the 
    heterogeneous analysis (SciFact, ArguAna, FiQA, TREC-COVID).
    """
    
    # List of selected datasets.
    datasets = [
        "scifact",  # Scientific Fact Checking
        "arguana",  # Argument Retrieval
        "fiqa",     # Financial QA
        "trec-covid",  # COVID-19 Literature Retrieval
        "nfcorpus", # Medical Information Retrieval (Nutrition Facts)
        "scidocs",  # Scientific Literature Citation Prediction
        "webis-touche2020" # Argument Retrieval (Controversial Topics)
    ]

    # Create the data directory if it does not exist
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"Created directory: {data_path}")

    # Official public URL pattern for BEIR datasets
    base_url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"

    print(f"Starting download of {len(datasets)} datasets into '{data_path}'...\n")

    for dataset in datasets:
        dataset_folder = os.path.join(data_path, dataset)
        
        # Optimization: Check if the dataset folder already exists to skip download
        if os.path.exists(dataset_folder) and os.path.isdir(dataset_folder):
            print(f"{dataset} already exists. Skipped.")
            continue
        
        print(f"Downloading and unzipping: {dataset}...")
        try:
            url = base_url.format(dataset)
            # Utilizing BEIR's utility to handle download and extraction
            data_path_result = util.download_and_unzip(url, data_path)
            print(f"    -> {dataset} successfully downloaded.")
        except Exception as e:
            print(f"Error downloading {dataset}: {e}")

    print("\nAll tasks completed.")

if __name__ == "__main__":
    # If running this script from the project root, ensure the path points to 'data'
    # If running from inside the 'data' folder, you can change this to "."
    TARGET_DIR = "." 
    
    download_beir_datasets(TARGET_DIR)