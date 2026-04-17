import os
import io
import json
import zipfile
import argparse
import urllib.request
import pandas as pd

# HarmBench behaviors CSV hosted directly on GitHub
HARMBENCH_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench"
    "/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
)

# XSTest prompts CSV — official repo (Röttger et al., NAACL 2024)
# Columns: id, prompt, type, label, focus, note
# 250 safe + 200 unsafe = 450 prompts total
XSTEST_URL = (
    "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"
)

# IHEval repo zip archive
IHEVAL_ZIP_URL = "https://github.com/ytyz1307zzh/IHEval/archive/refs/heads/main.zip"


def download_xstest(cache_dir: str) -> str:
    """Download the official XSTest prompts CSV from GitHub.

    The real schema is: id, prompt, type, label, focus, note
      - label : 'safe' | 'unsafe'  (ground truth)
      - type  : prompt-type category (homonyms, contrast_homonyms, …)
    """
    csv_path = os.path.join(cache_dir, "xstest_prompts.csv")
    if os.path.exists(csv_path):
        print(f"XSTest already present at {csv_path}, skipping download.")
        return csv_path

    print(f"Downloading XSTest from {XSTEST_URL} ...")
    with urllib.request.urlopen(XSTEST_URL) as resp:
        raw = resp.read().decode("utf-8")

    # Parse with pandas to validate and normalise
    from io import StringIO
    df = pd.read_csv(StringIO(raw))

    required = {"id", "prompt", "type", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"XSTest CSV is missing expected columns: {missing}. "
            f"Actual columns: {list(df.columns)}"
        )

    # Normalise label to lowercase
    df["label"] = df["label"].str.strip().str.lower()
    df.to_csv(csv_path, index=False)
    safe_n   = (df["label"] == "safe").sum()
    unsafe_n = (df["label"] == "unsafe").sum()
    print(f"Saved XSTest ({len(df)} rows: {safe_n} safe, {unsafe_n} unsafe) to {csv_path}")
    return csv_path


def _extract_iheval_rows(json_obj, task: str, setting: str) -> list:
    """Convert one IHEval JSON object (one entry from a benchmark file) into CSV rows."""
    rows = []
    conflict_type = setting if setting == "conflict" else "none"
    entry_id = str(json_obj.get("id", ""))

    # Safety task: entries have a list of test dicts with an "input" key
    if "test" in json_obj:
        for i, test_case in enumerate(json_obj["test"]):
            prompt = test_case.get("input", "")
            rows.append({
                "item_id": f"{task}_{setting}_{entry_id}_{i}",
                "prompt": prompt,
                "conflict_type": conflict_type,
            })
    # Other tasks: entry likely has a top-level "input" or "prompt" field
    elif "input" in json_obj:
        rows.append({
            "item_id": f"{task}_{setting}_{entry_id}",
            "prompt": json_obj["input"],
            "conflict_type": conflict_type,
        })
    elif "prompt" in json_obj:
        rows.append({
            "item_id": f"{task}_{setting}_{entry_id}",
            "prompt": json_obj["prompt"],
            "conflict_type": conflict_type,
        })
    return rows


def download_iheval(cache_dir: str) -> str:
    """Download IHEval from GitHub and flatten all benchmark JSONs into a single CSV."""
    csv_path = os.path.join(cache_dir, "iheval.csv")
    if os.path.exists(csv_path):
        print(f"IHEval already present at {csv_path}, skipping download.")
        return csv_path

    print(f"Downloading IHEval archive from {IHEVAL_ZIP_URL} ...")
    with urllib.request.urlopen(IHEVAL_ZIP_URL) as resp:
        zip_bytes = resp.read()

    all_rows = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        benchmark_files = [
            name for name in zf.namelist()
            if "/benchmark/" in name and name.endswith(".json")
        ]
        for filepath in benchmark_files:
            # Derive task name and setting from path:
            # e.g. IHEval-main/benchmark/safety/align_data.json
            #      IHEval-main/benchmark/rule-following/conflict/data.json
            parts = filepath.split("/")
            try:
                bm_idx = parts.index("benchmark")
            except ValueError:
                continue
            task = parts[bm_idx + 1] if bm_idx + 1 < len(parts) else "unknown"
            # setting is encoded in the filename or parent folder
            fname = parts[-1].lower()
            parent = parts[-2].lower() if len(parts) > 2 else ""
            if "conflict" in fname or "attack" in fname or "conflict" in parent:
                setting = "conflict"
            else:
                setting = "aligned"

            with zf.open(filepath) as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    continue
            if not isinstance(data, list):
                data = [data]
            for entry in data:
                all_rows.extend(_extract_iheval_rows(entry, task, setting))

    if not all_rows:
        print("⚠️  No IHEval rows extracted — check benchmark/ JSON structure.")
        return csv_path

    df = pd.DataFrame(all_rows, columns=["item_id", "prompt", "conflict_type"])
    df.to_csv(csv_path, index=False)
    print(f"Saved IHEval ({len(df)} rows) to {csv_path}")
    return csv_path


def download_datasets(cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)

    print("Downloading XSTest...")
    try:
        download_xstest(cache_dir)
    except Exception as e:
        print(f"Failed to download XSTest: {e}")
    
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
        if "FunctionalCategory" not in df.columns:
            if "SemanticCategory" in df.columns:
                col_map["SemanticCategory"] = "FunctionalCategory"
        if col_map:
            df = df.rename(columns=col_map)
            df.to_csv(csv_path, index=False)
        print(f"Saved HarmBench ({len(df)} rows) to {csv_path}")
    except Exception as e:
        print(f"Failed to download HarmBench: {e}")

    print("Downloading IHEval...")
    try:
        download_iheval(cache_dir)
    except Exception as e:
        print(f"Failed to download IHEval: {e}")

    print("Download completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="./artifacts/datasets", help="Directory to save datasets")
    args = parser.parse_args()
    
    download_datasets(args.cache_dir)
