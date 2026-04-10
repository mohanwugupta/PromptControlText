import pandas as pd
from typing import List, Tuple
from core.schema import EvalItem
from sklearn.model_selection import train_test_split

def load_xstest(csv_path: str) -> List[EvalItem]:
    df = pd.read_csv(csv_path)
    items = []
    for _, row in df.iterrows():
        # XSTest format typically has type like 'safe_prompt' or 'unsafe_prompt'
        domain = str(row.get('type', ''))
        gold_label = "safe" if "safe" in domain and "unsafe" not in domain else "unsafe"
        
        item = EvalItem(
            item_id=f"xstest_{row['id']}",
            benchmark="XSTest",
            domain=domain,
            input_text=str(row['prompt']),
            gold_label=gold_label
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
