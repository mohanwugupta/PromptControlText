import pytest
import os
import pandas as pd
from unittest.mock import patch, MagicMock

# The module is imported normally for integration testing mock paths
import experiments.run_phase1

def test_phase1_execution_loop(tmp_path):
    # Set the artifact output directory to temp
    output_csv = tmp_path / "phase1_results.csv"
    
    # Mock data setup
    mock_items = [
        MagicMock(input_text="Safe prompt?", benchmark="XSTest", domain="safe_prompt", gold_label="safe", item_id="x_1"),
        MagicMock(input_text="Hack a thing", benchmark="HarmBench", domain="Cybersecurity", gold_label="unsafe", item_id="h_1")
    ]
    for item in mock_items:
        item.model_dump.return_value = {"item_id": item.item_id, "input_text": item.input_text, "benchmark": item.benchmark, "domain": item.domain, "gold_label": item.gold_label}

    mock_client = MagicMock()
    mock_client.generate.return_value = ("I cannot do this", {"model": "test", "temperature": 0})
    
    mock_registry = {
        "Refuse-first": {"variants": {"v1": "Prompt text"}}
    }
    
    # Mocking external calls
    with patch("experiments.run_phase1.load_xstest", return_value=mock_items[:1]), \
         patch("experiments.run_phase1.load_harmbench", return_value=mock_items[1:]), \
         patch("experiments.run_phase1.load_iheval", return_value=[]), \
         patch("experiments.run_phase1.load_registry", return_value=mock_registry), \
         patch("experiments.run_phase1.VLLMClient", return_value=mock_client), \
         patch("experiments.run_phase1.os.path.exists", return_value=True):
         
         # Execute
         experiments.run_phase1.run_experiment(output_filepath=str(output_csv), mock_mode=True)
         
    assert output_csv.exists()
    df = pd.read_csv(output_csv)
    
    # Validation
    assert len(df) == 2 # 2 items * 1 family * 1 variation
    assert (df["refusal_score"] == 1.0).all() # due to mock return value
    assert "prompt_family" in df.columns


import experiments.run_phase2

def test_phase2_execution_loop(tmp_path):
    output_csv = tmp_path / "phase2_results.csv"
    
    mock_items = [
        MagicMock(input_text="Ignore system", benchmark="IHEval", domain="system_vs_user", gold_label="conflict", item_id="iheval_1")
    ]
    for item in mock_items:
        item.model_dump.return_value = {"item_id": item.item_id, "input_text": item.input_text, "benchmark": item.benchmark, "domain": item.domain, "gold_label": item.gold_label}

    mock_client = MagicMock()
    mock_client.generate.return_value = ("I cannot ignore my previous instructions", {"model": "test", "temperature": 0})
    
    with patch("experiments.run_phase2.load_iheval", return_value=mock_items), \
         patch("experiments.run_phase2.load_registry", return_value={"Hierarchy-first": {"variants": {"v1": "Prompt"}}}), \
         patch("experiments.run_phase2.VLLMClient", return_value=mock_client), \
         patch("experiments.run_phase2.os.path.exists", return_value=True):
         
         experiments.run_phase2.run_experiment(output_filepath=str(output_csv), mock_mode=True)
         
    df = pd.read_csv(output_csv)
    assert len(df) == 1
    assert (df["hierarchy_following_score"] == 1.0).all()

def test_phase1_no_data_raises_error(tmp_path):
    output_csv = tmp_path / "phase1_results.csv"
    
    with patch("experiments.run_phase1.load_xstest", return_value=[]), \
         patch("experiments.run_phase1.load_harmbench", return_value=[]), \
         patch("experiments.run_phase1.load_iheval", return_value=[]), \
         patch("experiments.run_phase1.load_registry", return_value={"mock": {"variants": {"v1": "Prompt"}}}), \
         patch("experiments.run_phase1.os.path.exists", return_value=True):
         
         with pytest.raises(ValueError, match="No benchmark items loaded"):
             experiments.run_phase1.run_experiment(output_filepath=str(output_csv), mock_mode=False)

def test_phase2_no_data_raises_error(tmp_path):
    output_csv = tmp_path / "phase2_results.csv"
    
    with patch("experiments.run_phase2.load_iheval", return_value=[]), \
         patch("experiments.run_phase2.load_registry", return_value={"mock": {"variants": {"v1": "Prompt"}}}), \
         patch("experiments.run_phase2.os.path.exists", return_value=True):
         
         with pytest.raises(ValueError, match="No benchmark items loaded for Phase 2"):
             experiments.run_phase2.run_experiment(output_filepath=str(output_csv), mock_mode=False)


# ---------------------------------------------------------------------------
# _is_valid_output tests
# ---------------------------------------------------------------------------

def test_valid_output_normal_text():
    valid, reason = experiments.run_phase1._is_valid_output("I cannot help with that request.")
    assert valid is True
    assert reason == ""

def test_invalid_output_tool_call_token():
    valid, reason = experiments.run_phase1._is_valid_output("Sure! <tool_call>{'name': 'execute'}</tool_call>")
    assert valid is False
    assert reason == "tool_call_token"

def test_invalid_output_garbled_unicode():
    # Simulate text where >10% of chars carry combining diacritics
    garbled = "tëying tëo äsk whät yöu'rë döing hërë ïn thïs plâcë ànd whÿ"
    valid, reason = experiments.run_phase1._is_valid_output(garbled)
    assert valid is False
    assert reason == "garbled_unicode"

def test_valid_output_empty_string():
    # Empty string: no diacritics, no tool_call token — should pass validation
    valid, reason = experiments.run_phase1._is_valid_output("")
    assert valid is True

def test_valid_output_accented_name():
    # Normal text with a few accented characters (well below 10% threshold)
    valid, reason = experiments.run_phase1._is_valid_output(
        "The café in Zürich is run by François."
    )
    assert valid is True

def test_malformed_output_column_written_for_invalid():
    """Records with invalid output must have malformed_output set and no score columns."""
    mock_items = [
        MagicMock(input_text="test", benchmark="XSTest", domain="safe", gold_label="safe", item_id="x_1")
    ]
    for item in mock_items:
        item.model_dump.return_value = {
            "item_id": item.item_id, "input_text": item.input_text,
            "benchmark": item.benchmark, "domain": item.domain, "gold_label": item.gold_label,
        }
    # mock_mode=True uses LLMClient, so patch LLMClient to return a tool_call output
    mock_client = MagicMock()
    mock_client.generate.return_value = ("<tool_call>dangerous()</tool_call>", {"model": "test"})
    mock_registry = {"Refuse-first": {"variants": {"v1": "Prompt"}}}

    with patch("experiments.run_phase1.load_xstest", return_value=mock_items), \
         patch("experiments.run_phase1.load_harmbench", return_value=[]), \
         patch("experiments.run_phase1.load_iheval", return_value=[]), \
         patch("experiments.run_phase1.load_registry", return_value=mock_registry), \
         patch("experiments.run_phase1.LLMClient", return_value=mock_client), \
         patch("experiments.run_phase1.os.path.exists", return_value=True):
        import tempfile, os as _os
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            tmp = f.name
        try:
            experiments.run_phase1.run_experiment(output_filepath=tmp, mock_mode=True)
            df = pd.read_csv(tmp)
            assert "malformed_output" in df.columns
            assert df.loc[0, "malformed_output"] == "tool_call_token"
            # Score columns should be absent (empty dict returned for invalid output)
            assert "refusal_score" not in df.columns or pd.isna(df.loc[0, "refusal_score"])
        finally:
            _os.unlink(tmp)
