#!/usr/bin/env python
"""
Download HuggingFace models onto the cluster's shared model directory.

Reads the model list from `configs/model_registry.yaml` and downloads each
one with `huggingface_hub.snapshot_download` into
`/scratch/gpfs/JORDANAT/mg9965/models/<local_dir_name>`, matching the
directory-naming convention already used by the existing slurm scripts
(e.g. `meta-llama--Llama-3.3-70B-Instruct`).

Usage
-----
    # Download everything in the registry
    python -m models.download_models

    # Download only specific models by slug
    python -m models.download_models --only gemma4_12b deepseek_r1_distill_qwen_32b

    # Point at a different registry file / destination root
    python -m models.download_models --registry configs/model_registry.yaml \
        --models-root /scratch/gpfs/JORDANAT/mg9965/models

Auth
----
Several of these models are gated (Gemma, NVIDIA Nemotron). Before running,
log in once with:
    huggingface-cli login
or export HF_TOKEN=<your token>. The script will pick up HF_TOKEN
automatically.

Run this on a node with internet access (typically the login node, NOT a
compute node — most HPC compute nodes are firewalled). Once files land in
the shared `models/` directory, the offline `HF_HUB_OFFLINE=1` slurm serving
jobs can read them without any network access.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_MODELS_ROOT = "/scratch/gpfs/JORDANAT/mg9965/models"
DEFAULT_REGISTRY = str(Path(__file__).resolve().parents[1] / "configs" / "model_registry.yaml")


def load_registry(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data["models"]


def download_one(entry: Dict[str, Any], models_root: str, token: Optional[str], max_retries: int = 5) -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError

    slug = entry["slug"]
    repo_id = entry["repo_id"]
    revision = entry.get("revision", "main")
    local_dir_name = entry.get("local_dir_name") or repo_id.replace("/", "--")
    allow_patterns = entry.get("allow_patterns")
    dest = os.path.join(models_root, local_dir_name)

    print(f"\n=== [{slug}] {repo_id} (rev={revision}) -> {dest} ===")

    if entry.get("gated") and not token:
        print(
            f"⚠️  {repo_id} is marked as gated but no HF token was found. "
            "Set HF_TOKEN or run `huggingface-cli login`, and make sure "
            "you've accepted the model license on huggingface.co."
        )

    os.makedirs(dest, exist_ok=True)

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=dest,
                token=token,
                allow_patterns=allow_patterns,
                max_workers=8,
                resume_download=True,
            )
            print(f"✅ [{slug}] download complete: {dest}")
            return
        except HfHubHTTPError as e:
            last_err = e
            status = getattr(e.response, "status_code", None)
            if status == 401 or status == 403:
                print(
                    f"❌ [{slug}] access denied ({status}). This model is gated — "
                    f"accept the license at https://huggingface.co/{repo_id} and "
                    "ensure your token has access. Skipping."
                )
                return
            print(f"⚠️  [{slug}] attempt {attempt}/{max_retries} failed: {e}")
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"⚠️  [{slug}] attempt {attempt}/{max_retries} failed: {e}")

        if attempt < max_retries:
            sleep_s = 10 * attempt
            print(f"    retrying in {sleep_s}s...")
            time.sleep(sleep_s)

    print(f"❌ [{slug}] FAILED after {max_retries} attempts: {last_err}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", default=DEFAULT_REGISTRY, help="Path to model_registry.yaml")
    parser.add_argument(
        "--models-root",
        default=os.environ.get("MODELS_ROOT", DEFAULT_MODELS_ROOT),
        help="Destination root directory for downloaded models",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Only download these slugs (space-separated). Default: download all in registry.",
    )
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    entries = load_registry(args.registry)
    if args.only:
        wanted = set(args.only)
        entries = [e for e in entries if e["slug"] in wanted]
        missing = wanted - {e["slug"] for e in entries}
        if missing:
            print(f"⚠️  Unknown slugs requested (ignored): {sorted(missing)}")

    if not entries:
        print("Nothing to download.")
        return 1

    os.makedirs(args.models_root, exist_ok=True)
    print(f"Downloading {len(entries)} model(s) into {args.models_root}")

    for entry in entries:
        download_one(entry, args.models_root, token, max_retries=args.max_retries)

    print("\nAll requested downloads processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
