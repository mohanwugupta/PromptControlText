import os
import argparse
import urllib.request
import pandas as pd

# HarmBench behaviors CSV hosted directly on GitHub
HARMBENCH_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench"
    "/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
)

def download_datasets(cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    
    print("Downloading HarmBench...")
    try:
        csv_path = os.path.join(cache_dir, "harmbench_behaviors.csv")
        urllib.request.urlretrieve(HARMBENCH_URL, csv_path)
        # Verify expected columns exist
        df = pd.read_csv(csv_path)
        # Normalize column names to match harmbench.py loader expectations
        col_map = {}
        if "Behavior" not in df.columns and "behavior" in df.columns:
            col_map["behavior"] = "Behavior"
        if "BehaviorID" not in df.columns and "BehaviorID_Original" in df.columns:
            col_map["BehaviorID_Original"] = "BehaviorID"
        elif "BehaviorID" not in df.columns and "SemanticCategory" in df.columns:
            pass  # ID column may be index; handled by harmbench.py
        if "FunctionalCategory" not in df.columns and "FunctionalCategory" not in df.columns:
            if "SemanticCategory" in df.columns:
                col_map["SemanticCategory"] = "FunctionalCategory"
        if col_map:
            df = df.rename(columns=col_map)
            df.to_csv(csv_path, index=False)
        print(f"Saved HarmBench ({len(df)} rows) to {csv_path}")
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
