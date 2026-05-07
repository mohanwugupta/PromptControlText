import pandas as pd
from typing import List
from core.schema import EvalItem

def load_harmbench(csv_path: str) -> List[EvalItem]:
    df = pd.read_csv(csv_path)
    items = []
    for _, row in df.iterrows():
        domain = str(row.get('FunctionalCategory', 'Harmful'))
        
        item = EvalItem(
            item_id=str(row['BehaviorID']),
            benchmark="HarmBench",
            domain=domain,
            input_text=str(row['Behavior']),
            gold_label="unsafe"  # HarmBench dataset primarily contains harmful instances.
        )
        items.append(item)
    return items
