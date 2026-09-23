from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path


def _command(*args: str):
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_runtime_metadata():
    data = {
        "git_commit": _command("git", "rev-parse", "HEAD"),
        "git_dirty": bool(_command("git", "status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("torch", "transformers", "vllm", "openai", "numpy", "scipy")
        },
        "environment": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "HF_HOME", "VLLM_CACHE_DIR", "SPAR_SCRATCH")
            if os.environ.get(key) is not None
        },
    }
    try:
        import torch
        data["torch"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except Exception as exc:
        data["torch_error"] = repr(exc)
    return data


def write_run_metadata(path: str | Path, extra=None) -> None:
    payload = collect_runtime_metadata()
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
