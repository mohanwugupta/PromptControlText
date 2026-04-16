"""
Loader for the PRD v2 custom boundary dataset (Benchmark D).

PRD v2 §8: 120-180 annotated items with policy_label, ambiguity_level,
context_condition, benchmark_source, and rationale.
"""
import pandas as pd
from typing import List
from core.schema import EvalItem


def load_boundary_dataset(csv_path: str) -> List[EvalItem]:
    """Load the annotated boundary dataset into EvalItem objects."""
    df = pd.read_csv(csv_path)

    items = []
    for _, row in df.iterrows():
        item = EvalItem(
            item_id=str(row["item_id"]),
            benchmark="BoundarySet",
            domain=str(row["domain"]),
            input_text=str(row["input_text"]),
            gold_label=str(row["gold_label"]),
            policy_label=str(row["policy_label"]),
            ambiguity_level=str(row["ambiguity_level"]),
            context_condition=str(row["context_condition"]),
            metadata={"rationale": str(row["rationale"]), "benchmark_source": str(row["benchmark_source"])},
        )
        items.append(item)

    return items
