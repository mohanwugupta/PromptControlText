import pytest
import pandas as pd
from analysis.metrics import aggregate_scores

def test_aggregate_scores():
    # Construct synthetic evaluated items as a dataframe
    df = pd.DataFrame([
        {"prompt_family": "Refuse-first", "domain": "unsafe", "refusal_score": 1.0, "compliance_score": 0.0},
        {"prompt_family": "Refuse-first", "domain": "unsafe", "refusal_score": 1.0, "compliance_score": 0.0},
        {"prompt_family": "Refuse-first", "domain": "safe", "refusal_score": 0.5, "compliance_score": 0.5},
        {"prompt_family": "Clarify-first", "domain": "unsafe", "refusal_score": 0.0, "compliance_score": 0.0, "clarification_score": 1.0},
    ])
    
    agg = aggregate_scores(df, group_by=["prompt_family", "domain"])
    
    # Check Reufse-first unsafe
    rf_unsafe = agg[(agg["prompt_family"] == "Refuse-first") & (agg["domain"] == "unsafe")].iloc[0]
    assert rf_unsafe["refusal_score_mean"] == 1.0
    assert rf_unsafe["compliance_score_mean"] == 0.0
    
    # Check Refuse-first safe
    rf_safe = agg[(agg["prompt_family"] == "Refuse-first") & (agg["domain"] == "safe")].iloc[0]
    assert rf_safe["refusal_score_mean"] == 0.5
