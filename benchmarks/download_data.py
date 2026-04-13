import os
import argparse
from datasets import load_dataset
import pandas as pd

def download_datasets(cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    
    print("Downloading HarmBench...")
    try:
        # Load cais/harmbench, specifically standard behaviors
        dataset = load_dataset("cais/harmbench", "behaviors", split="train")
        df = dataset.to_pandas()
        csv_path = os.path.join(cache_dir, "harmbench_behaviors.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved HarmBench to {csv_path}")
    except Exception as e:
        print(f"Failed to download Harmbench: {e}")

    print("Downloading IHEval...")
    try:
        # Mocking IHEval download logic from HF or specific paths
        dataset = load_dataset("google/iheval", split="test")
        df = dataset.to_pandas()
        csv_path = os.path.join(cache_dir, "iheval.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved IHEval to {csv_path}")
    except Exception as e:
        print(f"Failed to download IHEval: {e}")

    # XSTest is typically downloaded similarly but for now we focus on the missing ones.
    print("Download completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="./artifacts/datasets", help="Directory to save datasets")
    args = parser.parse_args()
    
    download_datasets(args.cache_dir)
