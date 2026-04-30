import os
import unicodedata
import argparse
import pandas as pd
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.schema import EvalItem
from benchmarks.iheval import load_iheval
from prompts.registry import load_registry, render_prompt, iter_prompt_triples
from models.vllm_client import VLLMClient
from models.client import LLMClient
from scoring.hierarchy_scorer import parse_hierarchy_response

# ---------------------------------------------------------------------------
# Output quality validation (shared logic with run_phase1)
# ---------------------------------------------------------------------------

def _is_valid_output(text: str) -> tuple[bool, str]:
    """
    Return (is_valid, reason) for a raw model output string.

    Catches two known failure modes:
      1. Literal ``<tool_call>`` tokens — the model serialised a function call
         instead of producing a plain-text response.
      2. Garbled Unicode — >10 % of characters carry combining diacritics,
         which indicates the model hallucinated character substitutions on
         adversarial / malformed inputs.
    """
    if "<tool_call>" in text:
        return False, "tool_call_token"
    nfd = unicodedata.normalize("NFD", text)
    combining = sum(1 for c in nfd if unicodedata.category(c) == "Mn")
    if len(text) > 0 and combining / len(nfd) > 0.10:
        return False, "garbled_unicode"
    return True, ""

def _generate_one(client, item: EvalItem, family: str, variant: str, prompt_text: str, generator_model: str, clarity: str = None):
    """Single unit of work: one (item, prompt_family, prompt_variant) triple."""
    try:
        output, metadata = client.generate(
            system_prompt=prompt_text,
            user_prompt=item.input_text,
            model=generator_model,
            temperature=0.0
        )
    except Exception as e:
        print(f"Failed generation for {item.item_id} [{family}/{variant}]: {e}")
        return None

    valid, invalid_reason = _is_valid_output(output)
    scores = parse_hierarchy_response(output) if valid else {}

    record = item.model_dump()
    record["prompt_family"] = family
    record["prompt_variant"] = variant
    record["clarity_level"] = clarity
    record["model_output"] = output
    record["malformed_output"] = "" if valid else invalid_reason
    record["metadata"] = metadata
    for k, v in scores.items():
        record[f"{k}_score"] = v

    return record

def run_experiment(output_filepath: str, generator_model: str = "Qwen2.5-72B-Instruct", mock_mode: bool = False, limit: int = 0, max_workers: int = 32, data_dir: str = None, registry_version: str = "v1"):
    print("Loading Registry...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_dir is None:
        data_dir = os.path.join(base_dir, "artifacts", "datasets")
    data_dir = os.path.abspath(data_dir)
    print(f"Dataset directory: {data_dir}")

    _REGISTRY_FILES = {
        "v1": "registry.yaml",
        "v2": "registry_v2.yaml",
        "v3": "registry_v3.yaml",
    }
    if registry_version not in _REGISTRY_FILES:
        raise ValueError(f"Unknown registry_version '{registry_version}'. Choose from: {list(_REGISTRY_FILES)}")
    registry_path = os.path.join(base_dir, "prompts", _REGISTRY_FILES[registry_version])
    print(f"Registry: {registry_path}")
    registry = load_registry(registry_path)
    
    print("Loading Benchmarks...")
    items: List[EvalItem] = []
    
    iheval_path = os.path.join(data_dir, "iheval.csv")
    if mock_mode:
        iheval_path = os.path.join(base_dir, "tests", "fixtures", "iheval_mock.csv")

    if os.path.exists(iheval_path):
        items.extend(load_iheval(iheval_path))
    else:
        print(f"Warning: {iheval_path} not found. Run download_data.py")
        
    if not items:
        raise ValueError("No benchmark items loaded for Phase 2.")
        
    if limit and limit > 0:
        items = items[:limit]

    print(f"Loaded {len(items)} IHEval items. Setting up model...")
    if mock_mode:
         client = LLMClient(mock_mode=True, mock_response="I cannot ignore my previous instructions.", enable_cache=True)
    else:
         client = VLLMClient(model_name=generator_model, enable_cache=True)

    # Build the full list of tasks upfront.
    # iter_prompt_triples() handles both v1 (clarity=None) and v2/v3 schemas.
    tasks = []
    for item in items:
        for family, clarity, variant, prompt_text in iter_prompt_triples(registry):
            tasks.append((item, family, clarity, variant, prompt_text))

    total = len(tasks)
    print(f"Submitting {total} generation tasks with max_workers={max_workers}...")

    evaluated_records = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_one, client, item, family, variant, prompt_text, generator_model, clarity): (item.item_id, family, variant)
            for item, family, clarity, variant, prompt_text in tasks
        }
        for future in as_completed(futures):
            completed += 1
            if completed % 100 == 0 or completed == total:
                print(f"  Progress: {completed}/{total}")
            record = future.result()
            if record is not None:
                evaluated_records.append(record)

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df = pd.DataFrame(evaluated_records)
    df.to_csv(output_filepath, index=False)
    print(f"Done. Saved to {output_filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-model", type=str, default="Qwen2.5-72B-Instruct")
    parser.add_argument("--output-file", type=str, default="artifacts/phase2_results.csv")
    parser.add_argument("--limit", type=int, default=0, help="Limit items for smoke test")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--max-workers", type=int, default=32, help="Number of concurrent generation threads.")
    parser.add_argument("--data-dir", type=str, default=None, help="Absolute path to dataset directory (overrides default).")
    parser.add_argument("--registry-version", type=str, default="v1", choices=["v1", "v2", "v3"],
                        help="Prompt registry schema version to use (default: v1).")
    
    args = parser.parse_args()
    
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(base, args.output_file)
    run_experiment(out, args.generator_model, mock_mode=args.mock, limit=args.limit, max_workers=args.max_workers, data_dir=args.data_dir, registry_version=args.registry_version)

