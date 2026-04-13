import os
import argparse
from datasets import load_dataset
import pandas as pd

def download_datasets(cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    
    print("Downloading HarmBench...")
    try:
        # cais/harmbench standard behaviors — no subset config needed
        dataset = load_dataset("cais/harmbench", split="train")
        df = dataset.to_pandas()
        # Rename columns to match harmbench.py loader expectations
        if "behavior" in df.columns and "BehaviorID" not in df.columns:
            df = df.rename(columns={"behavior": "Behavior", "behavior_id": "BehaviorID",
                                    "functional_category": "FunctionalCategory"})
        csv_path = os.path.join(cache_dir, "harmbench_behaviors.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved HarmBench to {csv_path}")
    except Exception as e:
        print(f"Failed to download HarmBench: {e}")

    print("Downloading IHEval...")
    # IHEval (arxiv 2502.08745) is not yet publicly released on HuggingFace.
    # Download the data manually from the paper's GitHub repo:
    #   https://github.com/google-deepmind/iheval
    # and place it as: artifacts/datasets/iheval.csv
    # Expected columns: item_id, prompt, conflict_type
    iheval_path = os.path.join(cache_dir, "iheval.csv")
    if os.path.exists(iheval_path):
        print(f"IHEval already present at {iheval_path}, skipping download.")
    else:
        print("⚠️  IHEval is not available on HuggingFace Hub. Download it manually from "
              "https://github.com/google-deepmind/iheval and place it at "
              f"{iheval_path} with columns: item_id, prompt, conflict_type")

    # XSTest is typically downloaded similarly but for now we focus on the missing ones.
    print("Download completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="./artifacts/datasets", help="Directory to save datasets")
    args = parser.parse_args()
    
    download_datasets(args.cache_dir)
