import os
import argparse
import pandas as pd
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.schema import EvalItem
from benchmarks.xstest import load_xstest
from benchmarks.harmbench import load_harmbench
from benchmarks.iheval import load_iheval
from prompts.registry import load_registry, render_prompt
from models.vllm_client import VLLMClient
from models.client import LLMClient
from scoring.harmbench_scorer import parse_harmbench_response
from scoring.hierarchy_scorer import parse_hierarchy_response

BENCHMARK_SCORERS = {
    "HarmBench": parse_harmbench_response,
    "XSTest": parse_harmbench_response,
    "IHEval": parse_hierarchy_response,
}

def score_item(benchmark: str, response: str):
    scorer = BENCHMARK_SCORERS.get(benchmark, parse_harmbench_response)
    return scorer(response)

def _generate_one(client, item: EvalItem, family: str, variant: str, prompt_text: str, generator_model: str):
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

    scores = score_item(item.benchmark, output)

    record = item.model_dump()
    record["prompt_family"] = family
    record["prompt_variant"] = variant
    record["model_output"] = output
    record["metadata"] = metadata
    for k, v in scores.items():
        record[f"{k}_score"] = v

    return record

def run_experiment(output_filepath: str, generator_model: str = "Qwen2.5-72B-Instruct", mock_mode: bool = False, limit: int = 0, max_workers: int = 32):
    print("Loading Registry...")
    # Setup Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    registry_path = os.path.join(base_dir, "prompts", "registry.yaml")
    
    # Load registry
    registry = load_registry(registry_path)
    
    print("Loading Benchmarks...")
    items: List[EvalItem] = []
    
    # Real dataset paths (populated by benchmarks/download_data.py)
    xstest_real_path   = os.path.join(base_dir, "artifacts", "datasets", "xstest_prompts.csv")
    xstest_mock_path   = os.path.join(base_dir, "tests", "fixtures", "xstest_mock.csv")
    harmbench_path     = os.path.join(base_dir, "artifacts", "datasets", "harmbench_behaviors.csv")
    iheval_path        = os.path.join(base_dir, "artifacts", "datasets", "iheval.csv")

    if mock_mode:
        harmbench_path = os.path.join(base_dir, "tests", "fixtures", "harmbench_mock.csv")

    # Prefer real XSTest when available; fall back to mock fixture for tests
    if mock_mode:
        xstest_path = xstest_mock_path
    elif os.path.exists(xstest_real_path):
        xstest_path = xstest_real_path
    elif os.path.exists(xstest_mock_path):
        print("Warning: real xstest_prompts.csv not found; using mock fixture. "
              "Run benchmarks/download_data.py to get the full dataset.")
        xstest_path = xstest_mock_path
    else:
        xstest_path = None

    if xstest_path and os.path.exists(xstest_path):
        items.extend(load_xstest(xstest_path))
    else:
        print("Warning: No XSTest data found. Run benchmarks/download_data.py")

    if os.path.exists(harmbench_path):
        items.extend(load_harmbench(harmbench_path))
    else:
        print(f"Warning: {harmbench_path} not found. Run download_data.py")

    if os.path.exists(iheval_path):
        items.extend(load_iheval(iheval_path))
    else:
        print(f"Warning: {iheval_path} not found. Run download_data.py")
        
    if not items:
        raise ValueError("No benchmark items loaded.")

    if limit and limit > 0:
        items = items[:limit]

    print(f"Loaded {len(items)} items. Setting up model {generator_model}...")
    if mock_mode:
         client = LLMClient(mock_mode=True, mock_response="I cannot help with that.", enable_cache=True)
    else:
         client = VLLMClient(model_name=generator_model, enable_cache=True)

    # Build the full list of (item, family, variant, prompt_text) tasks upfront
    tasks = []
    for item in items:
        for family in registry.keys():
            for variant in registry[family]['variants'].keys():
                prompt_text = render_prompt(registry, family, variant)
                tasks.append((item, family, variant, prompt_text))

    total = len(tasks)
    print(f"Submitting {total} generation tasks with max_workers={max_workers}...")

    evaluated_records = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_one, client, item, family, variant, prompt_text, generator_model): (item.item_id, family, variant)
            for item, family, variant, prompt_text in tasks
        }
        for future in as_completed(futures):
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"  Progress: {completed}/{total}")
            record = future.result()
            if record is not None:
                evaluated_records.append(record)

    print(f"Saving {len(evaluated_records)} evaluations to {output_filepath}")
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df = pd.DataFrame(evaluated_records)
    df.to_csv(output_filepath, index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-model", type=str, default="Qwen2.5-72B-Instruct")
    parser.add_argument("--output-file", type=str, default="artifacts/phase1_results.csv")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of benchmark items to evaluate.")
    parser.add_argument("--mock", action="store_true", help="Use mock client and fixture paths for testing.")
    parser.add_argument("--max-workers", type=int, default=32, help="Number of concurrent generation threads.")
    
    args = parser.parse_args()
    
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(base, args.output_file)
    run_experiment(out, args.generator_model, mock_mode=args.mock, limit=args.limit, max_workers=args.max_workers)
