"""
RED tests for PRD v2 plots.

PRD v2 analysis visualizations:
- Policy distribution heatmap per controller family
- Routing accuracy by context condition and clarity level
"""
import pytest
import os
import pandas as pd
from analysis.plots import plot_policy_distribution, plot_routing_accuracy


class TestPolicyDistributionPlot:
    def _make_df(self):
        return pd.DataFrame([
            {"prompt_family": "Refuse-first", "classified_policy": "refuse"},
            {"prompt_family": "Refuse-first", "classified_policy": "refuse"},
            {"prompt_family": "Refuse-first", "classified_policy": "answer"},
            {"prompt_family": "Clarify-first", "classified_policy": "clarify"},
            {"prompt_family": "Clarify-first", "classified_policy": "answer"},
            {"prompt_family": "Clarify-first", "classified_policy": "clarify"},
            {"prompt_family": "Minimal-safe-help", "classified_policy": "minimal_safe_help"},
            {"prompt_family": "Minimal-safe-help", "classified_policy": "refuse"},
            {"prompt_family": "Hierarchy-first", "classified_policy": "hierarchy_preserve"},
            {"prompt_family": "Hierarchy-first", "classified_policy": "refuse"},
        ])

    def test_produces_file(self, tmp_path):
        csv_file = tmp_path / "results.csv"
        png_file = tmp_path / "policy_dist.png"
        self._make_df().to_csv(csv_file, index=False)
        plot_policy_distribution(str(csv_file), str(png_file))
        assert png_file.exists()
        assert png_file.stat().st_size > 0


class TestRoutingAccuracyPlot:
    def _make_df(self):
        return pd.DataFrame([
            {"prompt_family": "Refuse-first", "clarity_level": "vague",
             "context_condition": "clean", "routing_correct": 1},
            {"prompt_family": "Refuse-first", "clarity_level": "explicit",
             "context_condition": "clean", "routing_correct": 1},
            {"prompt_family": "Refuse-first", "clarity_level": "vague",
             "context_condition": "ambiguous", "routing_correct": 0},
            {"prompt_family": "Refuse-first", "clarity_level": "explicit",
             "context_condition": "ambiguous", "routing_correct": 1},
            {"prompt_family": "Clarify-first", "clarity_level": "vague",
             "context_condition": "ambiguous", "routing_correct": 0},
            {"prompt_family": "Clarify-first", "clarity_level": "explicit",
             "context_condition": "ambiguous", "routing_correct": 1},
        ])

    def test_produces_file(self, tmp_path):
        csv_file = tmp_path / "results.csv"
        png_file = tmp_path / "routing_acc.png"
        self._make_df().to_csv(csv_file, index=False)
        plot_routing_accuracy(str(csv_file), str(png_file))
        assert png_file.exists()
        assert png_file.stat().st_size > 0
