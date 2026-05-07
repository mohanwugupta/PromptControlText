import pandas as pd
from typing import List
from core.schema import EvalItem

def load_iheval(csv_path: str) -> List[EvalItem]:
    df = pd.read_csv(csv_path)
    items = []
    for _, row in df.iterrows():
        domain = str(row.get('conflict_type', 'unknown'))
        gold_label = "conflict" if domain != "none" and domain != "unknown" else "non-conflict"
        
        item = EvalItem(
            item_id=str(row['item_id']),
            benchmark="IHEval",
            domain=domain,
            input_text=str(row['prompt']),
            gold_label=gold_label
        )
        items.append(item)
    return items
