"""
scripts/precache_harmony_vocab.py

Run ONCE on the login node (which has internet access) to pre-cache the
openai_harmony GPT-oss BPE vocab file.

The cache lands in $TIKTOKEN_CACHE_DIR (default: ~/.tiktoken_cache), which is
on the shared NFS filesystem at Princeton Della, so compute nodes can read it
without any network access.

Usage (on the login node):
    conda activate PromptControlText
    python scripts/precache_harmony_vocab.py
"""
from __future__ import annotations

import os
import pathlib
import sys

# Mirror what the SLURM job sets
TIKTOKEN_CACHE_DIR = os.environ.get(
    "TIKTOKEN_CACHE_DIR",
    str(pathlib.Path.home() / ".tiktoken_cache"),
)
os.environ["TIKTOKEN_CACHE_DIR"] = TIKTOKEN_CACHE_DIR
pathlib.Path(TIKTOKEN_CACHE_DIR).mkdir(parents=True, exist_ok=True)

print(f"TIKTOKEN_CACHE_DIR = {TIKTOKEN_CACHE_DIR}")
print(f"Attempting to load HARMONY_GPT_OSS encoding...")

try:
    from openai_harmony import load_harmony_encoding, HarmonyEncodingName  # type: ignore
except ImportError as e:
    print(f"❌ openai_harmony not installed: {e}")
    sys.exit(1)

try:
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    print(f"✅ Harmony vocab loaded successfully: {type(enc).__name__}")
    print(f"   Vocab size: {enc.n_vocab if hasattr(enc, 'n_vocab') else 'unknown'}")
    print(f"")
    print(f"Cache contents of {TIKTOKEN_CACHE_DIR}:")
    for p in sorted(pathlib.Path(TIKTOKEN_CACHE_DIR).iterdir()):
        size_kb = p.stat().st_size // 1024
        print(f"   {p.name}  ({size_kb} KB)")
    print(f"\n✅ Done. Compute nodes can now load the vocab offline.")
except Exception as e:
    print(f"❌ Failed to load harmony vocab: {e}")
    print()
    print("If this is a network error, check that the login node has internet access.")
    print("If the cache dir is on a network filesystem shared with compute nodes,")
    print("run this script on a login node, not a compute node.")
    sys.exit(1)
