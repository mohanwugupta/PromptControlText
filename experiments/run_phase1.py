import os
import argparse
import pandas as pd
from typing import List

from core.schema import EvalItem
from benchmarks.xstest import load_xstest
from benchmarks.harmbench import load_harmbench
from prompts.registry import load_registry, render_prompt
from models.vllm_client import VLLMClient
from models.client import LLMClient
from scoring.harmbench_scorer import parse_harmbench_response

def run_experiment(output_filepath: str, generator_model: str = "Qwen2.5-72B-Instruct", mock_mode: bool = False, limit: int = 0):
    print("Loading Registry...")
    # Setup Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    registry_path = os.path.join(base_dir, "prompts", "registry.yaml")
    
    # Load registry
    registry = load_registry(registry_path)
    
    print("Loading Benchmarks...")
    items: List[EvalItem] = []
    
    # Optional logic: Use mock paths inside test or actual downloaded data paths
    xstest_path = os.path.join(base_dir, "tests", "fixtures", "xstest_mock.csv")
    harmbench_path = os.path.join(base_dir, "artifacts", "datasets", "harmbench_behaviors.csv")
    if mock_mode:
        harmbench_path = os.path.join(base_dir, "tests", "fixtures", "harmbench_mock.csv")

    if os.path.exists(xstest_path):
        items.extend(load_xstest(xstest_path))
    else:
        print(f"Warning: {xstest_path} not found.")
        
    if os.path.exists(harmbench_path):
        items.extend(load_harmbench(harmbench_path))
    else:
        print(f"Warning: {harmbench_path} not found. Run download_data.py")
        
    if not items:
        raise ValueError("No benchmark items loaded.")

    if limit and limit > 0:
        items = items[:limit]

    print(f"Loaded {len(items)} items. Setting up model {generator_model}...")
    if mock_mode:
         client = LLMClient(mock_mode=True, mock_response="I cannot help with that.", enable_cache=True)
    else:
         client = VLLMClient(model_name=generator_model, enable_cache=True)

    evaluated_records = []
    
    print("Generating responses...")
    for item in items:
        for family in registry.keys():
            for variant in registry[family]['variants'].keys():
                prompt_text = render_prompt(registry, family, variant)
                
                try:
                    output, metadata = client.generate(
                        system_prompt=prompt_text,
                        user_prompt=item.input_text,
                        model=generator_model,
                        temperature=0.0
                    )
                except Exception as e:
                    print(f"Failed generation for {item.item_id}: {e}")
                    continue
                    
                scores = parse_harmbench_response(output)
                
                record = item.model_dump()
                record["prompt_family"] = family
                record["prompt_variant"] = variant
                record["model_output"] = output
                record["metadata"] = metadata
                for k, v in scores.items():
                    record[f"{k}_score"] = v
                    
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
    
    args = parser.parse_args()
    
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(base, args.output_file)
    run_experiment(out, args.generator_model, mock_mode=args.mock, limit=args.limit)
