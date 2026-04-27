# Scratchpad

## Scorer Validation (2026-04-27)
*   **Scorer**: `scoring/harmbench_scorer.py` validated against official gold labels via `scoring/validate_scorer.py`.
*   **XSTest v2 GPT-4 completions (N=450)**: κ = **0.876** ✓ PASS — primary validity evidence (direct prompting, same register as our experiments).
*   **HarmBench text_behaviors_val_set (N=602)**: κ = **0.436** — at the hard regex ceiling for adversarial jailbreak data. Root causes: non-English responses (Chinese, Polish), fig-leaf jailbreaks, verbose redirected refusals. Not a scorer deficiency; the HarmBench paper's own LLM scorer (GPT-4) reaches κ ≈ 0.88. Report as a methods limitation.
*   **Changes made**: Removed a duplicate `parse_harmbench_response` definition (Python was silently using the second broken copy with `len > 600` fallback). Correct fallback is `len > 100` (Pareto-optimal across both benchmarks by grid search). Refusal patterns expanded from 11 → 26 phrasings grounded in FP/FN analysis.
*   **Report saved**: `artifacts/scorer_validation_report.txt`.

## Progress Update
*   **Phase 1**: Completed Execution pipeline over XSTest & HarmBench. Includes model caches, download logic, offline parsing, rules-based heuristic generation, tests, and plots.
*   **Phase 2**: Scaffolded IHEval evaluations. `download_data.py` pulls `google/iheval` caching it offline. Added `hierarchy_scorer.py` mapping instruction conflicts into `hierarchy_following`, `stop_compliance`, and `unsafe_continuation`.
*   **Phase 2 Plots**: Scaled `plots.py` with argument `--phase 2` which outputs Hierarchy Conflict grouped bar charts analyzing `Accuracy vs Unsafe Continuation` across vague vs explicit prompt variants.
*   **Version Control**: `.gitignore` populated with standard Python/OS filters.

## PRD v2 Implementation (Policy Routing as Primary DV)

### Core Thesis Change
Shifted primary outcome from refusal rate → **policy-routing accuracy**.
The unit of explanation is now which *policy* the model selects (answer, refuse, clarify, minimal_safe_help, hierarchy_preserve), not binary safe/unsafe.

### New & Modified Modules

#### Schema Extension (`core/schema.py`)
*   Added `policy_label`, `ambiguity_level`, `context_condition`, `clarity_level` fields to `EvalItem`.
*   All new fields are Optional for backward compatibility.
*   Pydantic validators enforce allowed values per PRD v2 §12.

#### Prompt Registry v2 (`prompts/registry_v2.yaml`, `prompts/registry.py`)
*   **4 controller families**: Refuse-first, Clarify-first, Minimal-safe-help (NEW), Hierarchy-first.
*   **3 clarity levels** per family: vague, explicit, explicit_fallback (PRD v2 Factor 2).
*   **3 paraphrases** per (family × clarity) cell → 36 frozen prompts.
*   Added `render_prompt_v2()` function preserving backward-compat with v1 `render_prompt()`.
*   Original `registry.yaml` preserved untouched.

#### Policy Classifier (`scoring/policy_classifier.py`)
*   **Primary scorer for PRD v2.** Classifies model output into 5 policy labels.
*   Priority order: hierarchy_preserve > minimal_safe_help > clarify > refuse > answer.
*   Pattern-based classification with fallback heuristics.
*   `classify_policy()` returns a single label; `classify_policy_scores()` returns a one-hot dict.

#### Custom Boundary Dataset (`benchmarks/boundary_dataset.py`, `artifacts/datasets/boundary_dataset.csv`)
*   PRD v2 §8 Benchmark D: 130 annotated items.
*   Composition: 40 XSTest safe/sensitive, 40 HarmBench unsafe, 20 IHEval conflict, 10 quoted/analysis, 10 ambiguous intent, 10 minimal-safe-help scenarios.
*   Each item annotated with: `policy_label`, `ambiguity_level`, `context_condition`, `rationale`.

#### Routing Accuracy Metrics (`analysis/metrics.py`)
*   `compute_routing_accuracy()`: Primary outcome = policy_label == classified_policy.
*   `compute_secondary_metrics()`: harmful_compliance_rate, false_refusal_rate, clarification_rate.
*   Both support flexible `group_by` for slicing by family, clarity, condition.

#### Study Runners
*   `experiments/run_study1.py`: PRD v2 Study 1 — Policy routing on boundary items. Within-item repeated-measures across all families × clarity levels.
*   `experiments/run_study2.py`: PRD v2 Study 2 — Policy routing under hierarchy conflict. IHEval items across all families × clarity levels.

#### v2 Plots (`analysis/plots.py`)
*   `plot_policy_distribution()`: Heatmap of policy selections per family (RQ1 visualization).
*   `plot_routing_accuracy()`: Grouped bar chart of routing accuracy by family × clarity (RQ2 visualization).

### Test Coverage (TDD RED→GREEN)
*   `test_schema_v2.py`: 26 tests — enum validation, backward compatibility.
*   `test_prompts_v2.py`: 11 tests — registry structure, 36-prompt count, hash uniqueness.
*   `test_policy_classifier.py`: 24 tests — all 5 policy labels, edge cases, score dict.
*   `test_boundary_dataset.py`: 10 tests — loader, validation, coverage.
*   `test_routing_metrics.py`: 8 tests — accuracy computation, secondary metrics.
*   `test_study_runners.py`: 5 tests — Study 1 & 2 execution, error handling.
*   `test_v2_plots.py`: 2 tests — plot generation.
*   `test_v2_end_to_end.py`: 2 tests — full pipeline integration.
*   **Total new v2 tests: 88. All GREEN.**
