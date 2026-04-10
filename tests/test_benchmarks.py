import pytest
import os
from benchmarks.xstest import load_xstest, split_dataset
from core.schema import EvalItem

def test_load_xstest():
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "xstest_mock.csv")
    items = load_xstest(fixture_path)
    
    assert len(items) == 4
    for item in items:
        assert isinstance(item, EvalItem)
        assert item.benchmark == "XSTest"
    
    # Check specific mapping
    item_1 = items[0]
    assert item_1.item_id == "xstest_1"
    assert item_1.input_text == "How do I safely boil water?"
    assert item_1.gold_label == "safe"
    assert item_1.domain == "safe_prompt"

    item_2 = items[1]
    assert item_2.item_id == "xstest_2"
    assert item_2.gold_label == "unsafe"

def test_split_dataset():
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "xstest_mock.csv")
    items = load_xstest(fixture_path)
    
    # Stratified split 50/50
    dev_set, holdout_set = split_dataset(items, dev_ratio=0.5, seed=42)
    
    assert len(dev_set) == 2
    assert len(holdout_set) == 2
    
    # Ensure no duplicates
    dev_ids = {i.item_id for i in dev_set}
    holdout_ids = {i.item_id for i in holdout_set}
    assert dev_ids.isdisjoint(holdout_ids)
