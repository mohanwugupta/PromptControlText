"""
PRD v2 §13 — Study 1: Policy routing on boundary items.

Within-item repeated-measures: each item run under all controller families,
each family at all clarity levels. Primary outcome: policy-routing accuracy.
"""
import os
import argparse
import pandas as pd
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.schema import EvalItem
from benchmarks.boundary_dataset import load_boundary_dataset
from prompts.registry import load_registry, render_prompt_v2
from models.vllm_client import VLLMClient
from models.client import LLMClient
from scoring.policy_classifier import classify_policy


def _generate_one(client, item: EvalItem, family: str, clarity: str,
                   variant: str, prompt_text: str, generator_model: str):
    """Single unit of work: one (item, family, clarity, variant) quad."""
    try:
        output, metadata = client.generate(
            system_prompt=prompt_text,
            user_prompt=item.input_text,
            model=generator_model,
            temperature=0.0,
        )
    except Exception as e:
        print(f"Failed generation for {item.item_id} [{family}/{clarity}/{variant}]: {e}")
        return None

    classified = classify_policy(output)

    record = item.model_dump()
    record["prompt_family"] = family
    record["clarity_level"] = clarity
    record["prompt_variant"] = variant
    record["model_output"] = output
    record["metadata"] = metadata
    record["classified_policy"] = classified
    record["routing_correct"] = int(classified == item.policy_label)
    return record


def run_study1(
    output_filepath: str,
    generator_model: str = "Qwen2.5-72B-Instruct",
    mock_mode: bool = False,
    limit: int = 0,
    max_workers: int = 32,
):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load v2 registry
    registry_path = os.path.join(base_dir, "prompts", "registry_v2.yaml")
    if mock_mode:
        registry_path = os.path.join(base_dir, "tests", "fixtures", "registry_v2_mock.yaml")
    registry = load_registry(registry_path)

    # Load boundary dataset
    boundary_path = os.path.join(base_dir, "artifacts", "datasets", "boundary_dataset.csv")
    if mock_mode:
        boundary_path = os.path.join(base_dir, "tests", "fixtures", "boundary_mock.csv")
    items = load_boundary_dataset(boundary_path)

    if not items:
        raise ValueError("No boundary items loaded for Study 1.")

    if limit and limit > 0:
        items = items[:limit]

    print(f"Study 1: {len(items)} boundary items loaded.")

    if mock_mode:
        client = LLMClient(mock_mode=True, mock_response="I cannot help with that.", enable_cache=True)
    else:
        client = VLLMClient(model_name=generator_model, enable_cache=True)

    # Build tasks: item × family × clarity × variant
    tasks = []
    for item in items:
        for family, fdata in registry.items():
            cl_levels = fdata.get("clarity_levels", {})
            for clarity, variants in cl_levels.items():
                for variant, prompt_text in variants.items():
                    tasks.append((item, family, clarity, variant, prompt_text))

    total = len(tasks)
    print(f"  Submitting {total} generation tasks (max_workers={max_workers})...")

    evaluated_records = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _generate_one, client, item, family, clarity, variant,
                prompt_text, generator_model,
            ): (item.item_id, family, clarity, variant)
            for item, family, clarity, variant, prompt_text in tasks
        }
        for future in as_completed(futures):
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"  Progress: {completed}/{total}")
            record = future.result()
            if record is not None:
                evaluated_records.append(record)

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df = pd.DataFrame(evaluated_records)
    df.to_csv(output_filepath, index=False)
    print(f"Study 1 done. Saved {len(df)} rows to {output_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRD v2 Study 1: Policy routing on boundary items")
    parser.add_argument("--generator-model", type=str, default="Qwen2.5-72B-Instruct")
    parser.add_argument("--output-file", type=str, default="artifacts/study1_results.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--max-workers", type=int, default=32)
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_study1(
        os.path.join(base, args.output_file),
        args.generator_model,
        mock_mode=args.mock,
        limit=args.limit,
        max_workers=args.max_workers,
    )
