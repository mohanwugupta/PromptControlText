import pytest
import os
import pandas as pd
from analysis.plots import plot_safety_boundary, plot_hierarchy_conflict

def test_plot_safety_boundary_generates_file(tmp_path):
    # Setup dummy phase 1 output
    csv_file = tmp_path / "phase1_results.csv"
    png_file = tmp_path / "safety_boundary.png"
    
    df = pd.DataFrame([
        {"prompt_family": "Refuse-first", "gold_label": "unsafe", "compliance_score": 0.1, "refusal_score": 0.9},
        {"prompt_family": "Refuse-first", "gold_label": "safe", "compliance_score": 0.5, "refusal_score": 0.5},
        {"prompt_family": "Clarify-first", "gold_label": "unsafe", "compliance_score": 0.3, "refusal_score": 0.3},
        {"prompt_family": "Clarify-first", "gold_label": "safe", "compliance_score": 0.9, "refusal_score": 0.1},
    ])
    df.to_csv(csv_file, index=False)
    
    # Execute
    plot_safety_boundary(str(csv_file), str(png_file))
    
    # Verify
    assert png_file.exists()
    assert png_file.stat().st_size > 0

def test_plot_hierarchy_conflict_generates_file(tmp_path):
    # Setup dummy phase 2 output
    csv_file = tmp_path / "phase2_results.csv"
    png_file = tmp_path / "hierarchy_conflict.png"
    
    df = pd.DataFrame([
        {"prompt_family": "Hierarchy-first", "gold_label": "conflict", "hierarchy_following_score": 0.9, "unsafe_continuation_score": 0.1},
        {"prompt_family": "Refuse-first", "gold_label": "conflict", "hierarchy_following_score": 0.5, "unsafe_continuation_score": 0.5},
    ])
    df.to_csv(csv_file, index=False)
    
    # Execute
    plot_hierarchy_conflict(str(csv_file), str(png_file))
    
    # Verify
    assert png_file.exists()
    assert png_file.stat().st_size > 0
