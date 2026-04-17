import pandas as pd
from typing import List, Tuple
from core.schema import EvalItem
from sklearn.model_selection import train_test_split

def load_xstest(csv_path: str) -> List[EvalItem]:
    """Load XSTest prompts CSV.

    Supports both the real schema (id, prompt, type, label, focus, note) and
    the legacy mock schema (id, type, prompt) used in tests.

    The real schema has a dedicated ``label`` column ('safe'/'unsafe').
    The mock schema encodes the label inside the ``type`` field
    ('safe_prompt' / 'unsafe_prompt') — we preserve that fallback so
    existing tests continue to pass.
    """
    df = pd.read_csv(csv_path)

    # Determine gold_label source
    if "label" in df.columns:
        # Real dataset: use the explicit label column
        def _gold(row):
            return "safe" if str(row["label"]).strip().lower() == "safe" else "unsafe"
    else:
        # Legacy mock: infer from type string
        def _gold(row):
            domain = str(row.get("type", ""))
            return "safe" if "safe" in domain and "unsafe" not in domain else "unsafe"

    items = []
    for _, row in df.iterrows():
        domain = str(row.get("type", ""))
        item = EvalItem(
            item_id=f"xstest_{row['id']}",
            benchmark="XSTest",
            domain=domain,
            input_text=str(row["prompt"]),
            gold_label=_gold(row),
        )
        items.append(item)
    return items

def split_dataset(items: List[EvalItem], dev_ratio: float = 0.5, seed: int = 42) -> Tuple[List[EvalItem], List[EvalItem]]:
    """Splits items into dev and holdout sets, stratified by domain/gold_label."""
    if not items:
        return [], []
    
    # Stratify based on a combination of benchmark and domain
    stratify_labels = [f"{i.benchmark}_{i.domain}" for i in items]
    
    try:
        dev_set, holdout_set = train_test_split(
            items, 
            train_size=dev_ratio, 
            stratify=stratify_labels, 
            random_state=seed
        )
    except ValueError:
        # Fallback if there are classes with only 1 member making stratification impossible
        dev_set, holdout_set = train_test_split(
            items, 
            train_size=dev_ratio, 
            random_state=seed
        )
        
    return dev_set, holdout_set
