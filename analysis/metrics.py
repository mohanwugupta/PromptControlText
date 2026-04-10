import pandas as pd
from typing import List

def aggregate_scores(df: pd.DataFrame, group_by: List[str]) -> pd.DataFrame:
    """
    Computes the mean of all score columns grouped by the specified columns.
    """
    score_cols = [c for c in df.columns if c.endswith("_score")]
    
    if not score_cols:
        return df.groupby(group_by).size().reset_index(name="count")
        
    agg_dict = {col: ['mean'] for col in score_cols}
    
    agg_df = df.groupby(group_by).agg(agg_dict).reset_index()
    
    # Flatten multi-level columns
    agg_df.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0] 
        for col in agg_df.columns
    ]
    
    return agg_df
