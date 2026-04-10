import pytest
import os
import pandas as pd
from benchmarks.xstest import load_xstest
from prompts.registry import load_registry, render_prompt
from models.client import LLMClient
from scoring.custom import parse_response
from analysis.metrics import aggregate_scores

def test_end_to_end_pipeline():
    # 1. Load Registry
    registry_path = os.path.join(os.path.dirname(__file__), "fixtures", "registry_mock.yaml")
    registry = load_registry(registry_path)
    
    # 2. Load Dataset
    xstest_path = os.path.join(os.path.dirname(__file__), "fixtures", "xstest_mock.csv")
    items = load_xstest(xstest_path)
    
    # 3. Setup Model Client (Mocked)
    client = LLMClient(mock_mode=True, mock_response="I cannot help with that.", enable_cache=True)
    
    evaluated_records = []
    
    # 4. Run loop
    for item in items:
        for family in ["Refuse-first", "Clarify-first"]:
            prompt_text = render_prompt(registry, family, "v1")
            
            output, metadata = client.generate(
                system_prompt=prompt_text,
                user_prompt=item.input_text,
                model="mock-model",
                temperature=0.0
            )
            
            scores = parse_response(output)
            
            # Store results
            record = item.model_dump()
            record["prompt_family"] = family
            record["prompt_variant"] = "v1"
            record["model_output"] = output
            record["metadata"] = metadata
            for k, v in scores.items():
                record[f"{k}_score"] = v
                
            evaluated_records.append(record)
            
    # 5. Analysis
    df = pd.DataFrame(evaluated_records)
    assert len(df) == 8 # 4 items * 2 families
    
    agg = aggregate_scores(df, group_by=["prompt_family", "domain"])
    assert len(agg) == 4 # 2 families * 2 domains (safe, unsafe)
    
    # Test passed end to end
