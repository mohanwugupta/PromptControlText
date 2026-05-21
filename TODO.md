# TODO List

## Completed (TDD Framework & Integrations)
- [x] **Setup**: Initialized project repo, `requirements.txt`, and primary directories (`benchmarks`, `prompts`, `models`, `scoring`, `experiments`, `analysis`, `tests`, `artifacts`, `slurm`).
- [x] **Milestone 1**: Pydantic `core/schema.py` mapping of unified `EvalItem` definitions.
- [x] **Milestone 2**: Created basic `benchmarks/xstest.py` ingestion parser.
- [x] **Milestone 3**: Deployed `prompts/registry.yaml` containing frozen prompt baselines alongside hash validators.
- [x] **Milestone 4**: Initialized baseline model clients with caching logic.
- [x] **Milestone 5**: Created naive evaluation heuristic scorers inside `scoring/custom.py`.
- [x] **Milestone 6**: Added complex multi-index nested pandas metrics aggregators in `analysis/metrics.py`.
- [x] **Milestone 7**: Ensured `tests/test_end_to_end.py` compiles the full data lifecycle.
- [x] **Slurm Integration**: Developed `slurm/run_prompt_controllers_72b.sh` mapped to `Qwen2.5-72B-Instruct` with offline caches and background vLLM daemon monitoring.
- [x] **Slurm Generator**: Extracted and mapped the scalable OpenAI `vllm_client.py` class into the execution tier.

## Completed (Phase 1 & Phase 2 Execution)
- [x] **Phase 1 Execution (Refusal Boundaries)**
  - [x] Integrate specific live endpoints for parsing full XSTest and HarmBench / JailbreakBench.
  - [x] Bridge the testing framework natively into a `experiments/run_phase1.py` runner called by the Slurm script.
- [x] **Phase 2 Scaffolding (Hierarchy-sensitive)**
  - [x] Ingest the IHEval substrate structure into `benchmarks/iheval.py`.
  - [x] Validate scoring behavior accurately identifies hierarchy overrides / stop compliance.
- [x] **Model Evaluators & Auditing**
  - [x] Replace the naive rules-based scoring heuristics with generalized evaluation interfaces (HarmBench compatible scorers).
  - [x] Add native manual reporting outputs/tables plotting.
- [x] **Plot Generation**
  - [x] Implement figure generation blocks summarizing Refusal vs. Compliance Boundaries across target prompts.

## Completed (PRD v2 — Policy Routing as Primary DV)
- [x] **Schema Extension**: Added `policy_label`, `ambiguity_level`, `context_condition`, `clarity_level` to `EvalItem` with Pydantic validators (26 tests GREEN).
- [x] **Prompt Registry v2**: Created `registry_v2.yaml` with 4 families × 3 clarity levels × 3 paraphrases = 36 frozen prompts. Added `render_prompt_v2()` (11 tests GREEN).
- [x] **Policy Classifier**: Implemented `scoring/policy_classifier.py` — primary scorer mapping model output → 5 policy labels (answer/refuse/clarify/minimal_safe_help/hierarchy_preserve) (24 tests GREEN).
- [x] **Custom Boundary Dataset**: Created `artifacts/datasets/boundary_dataset.csv` (130 annotated items) and `benchmarks/boundary_dataset.py` loader (10 tests GREEN).
- [x] **Routing Accuracy Metrics**: Added `compute_routing_accuracy()` and `compute_secondary_metrics()` to `analysis/metrics.py` (8 tests GREEN).
- [x] **Study 1 Runner**: Implemented `experiments/run_study1.py` — policy routing on boundary items, within-item repeated-measures across all families × clarity (3 tests GREEN).
- [x] **Study 2 Runner**: Implemented `experiments/run_study2.py` — policy routing under hierarchy conflict using IHEval (2 tests GREEN).
- [x] **v2 Plots**: Added `plot_policy_distribution()` (heatmap) and `plot_routing_accuracy()` (grouped bar) to `analysis/plots.py` (2 tests GREEN).
- [x] **v2 End-to-End Integration**: Full pipeline smoke test from boundary items → v2 prompts → model → policy classifier → routing accuracy (2 tests GREEN).

## Completed (PRD v3 — Mining LLM Responses for Emergent Policy Types)
- [x] **Response Table** (`mining/response_table.py`): Unified Phase 1 + Phase 2 loader, strict column contract, NaN back-fill for missing scorer columns (12 tests GREEN).
- [x] **Feature Extraction** (`mining/features.py`): Deterministic lexical + length + punctuation + phrase-group features with langdetect language flag (17 tests GREEN).
- [x] **Cluster Pipeline** (`mining/clustering.py`): TF-IDF text-only and hybrid KMeans; deterministic seeds; silhouette diagnostics; top-terms extraction; 2-D viz reduction (19 tests GREEN).
- [x] **Exemplar Selection** (`mining/exemplars.py`): Centroid-nearest and stratified-random per cluster; `build_exemplar_table()` for audit CSV (11 tests GREEN).
- [x] **Routing Sensitivity** (`mining/routing_sensitivity.py`): Per-item disagreement score (cluster entropy + policy entropy + length CV); ranked list; report builder (13 tests GREEN).
- [x] **Reports** (`mining/reports.py`): Full pipeline CLI producing all 9 artifact types under `artifacts/mining/<date>/`; manifest.json (9 tests GREEN).
- [x] **End-to-End** (`tests/test_mining_end_to_end.py`): Full pipeline on fixtures, all artifacts written with correct schema (6 tests GREEN).
- [x] **Total new v3 tests: 93 GREEN.**

## Completed (PRD v4.1 — Audit, Scorer Validation & Statistical Infrastructure)
- [x] **Audit Pipeline**: Built `audit/` modules — `freeze.py`, `tracker.py`, `audit_set.py`, `analysis.py`, `dashboard.py`. 42 new tests GREEN.
- [x] **Human Audit**: 161 hand-labeled responses (71 priority + 90 HarmBench), Cohen's κ=0.532 with v1 scorer.
- [x] **Scorer v2 (`scoring/custom.py`)**: Rewrote custom scorer with extended refusal/clarification phrase lists and `_strip_preamble()`. Validated at κ=0.865, accuracy=91.3% vs 161 human labels.
- [x] **Full Rescoring**: Applied v2 scorer to all 106,016 rows of `artifacts/mining/2026-04-28/mining_table.csv`. Backfilled `policy_label` column. Distribution: compliance=57.7%, refusal=38.0%, clarification=3.9% (safety benchmarks only).
- [x] **Statistical Infrastructure**: Created `analysis/statistical_model.py` — Wilson CIs, Cohen's h, clustered logistic regression, `routing_effect_table()`, `remap_iheval_labels()`, `compute_family_effects_summary_from_df()`.
- [x] **Updated `analysis/metrics.py`**: Added `load_mining_table()` with IHEval label remapping.
- [x] **Updated `analysis/plots.py`**: 6 publication-quality figures — `fig_policy_routing.png`, `fig_safety_boundary.png`, `fig_hierarchy_conflict.png`, `fig_refusal_by_benchmark.png`, `fig_routing_entropy.png`, `fig_cohens_h_refusal.png`. All saved under `artifacts/audit/figures/`.

## Completed (Phase 1 v2 Execution — Full PRD v2 Design)
- [x] **Phase 1 v2 Run**: Executed full PRD v2 design on Qwen2.5-72B-Instruct. `artifacts/phase1_results.csv` — 242,640 rows, 8 families × 3 clarity levels × 3 benchmarks (XSTest 32,400 + HarmBench 28,800 + IHEval 181,440).
- [x] **Policy Labels Backfilled**: Applied v2 scorer to all phase1 rows. `policy_label` column populated for all 242,640 rows.
- [x] **PRD v2 Analysis (RQ1–RQ3)**:
  - **RQ1 confirmed** — χ²=4,172.9 (p≈0): Families induce distinct policy distributions. Within-item switch rate=88.4% (751/850 items). Clarify-first uniquely elicits clarification (14.1% vs 0.8–2.0% for others).
  - **RQ2 confirmed** — χ²=698.1 (p≈0): Clarity level shifts routing. Vague controllers→highest compliance (63.8%). Explicit_fallback→highest refusal (42.6%). Clarification peaks at explicit (4.3%).
  - **RQ3 (IHEval)** — χ²=240.4 (p≈0): Families differ on hierarchy routing. Refuse-first has lowest unsafe_continuation (96.8%); Answer-first/Clarify-first highest (98.1–98.3%). Effect is modest — all families dominated by unsafe_continuation.
  - **HarmBench harmful compliance**: Lowest = Evidence-first (33.5%) and Refuse-first (34.8%). Highest = Source-isolation-first (44.7%) and Clarify-first (42.2%).
  - **False refusal (XSTest safe items)**: Near-zero across all families (1.0–2.9%). Clarify-first converts safe items to clarification requests (18.4%) — a distinct routing signature.

## Pending
- [ ] **Statistical confirmatory analysis**: Run `fit_all_policy_models()` (logistic regression with HC3 SEs) on phase1_results.csv for family and clarity main effects.
- [ ] **Clarity × ambiguity interaction (RQ2 refinement)**: Partition XSTest by safe/sensitive/unsafe and test whether explicit_fallback advantage concentrates on ambiguous items (H2).
- [ ] **Within-item repeated-measures**: Compute per-item disagreement entropy across 8 families to identify high-sensitivity items for focused analysis.
- [ ] **IHEval hierarchy_following_score review**: Currently nearly all IHEval rows →unsafe_continuation; investigate whether hierarchy_following_score column is being populated correctly by run_phase1.py.
- [ ] **Regenerate all 6 publication figures** against phase1_results.csv (currently wired to mining_table.csv).
- [ ] **Phase 3 Scaffolding (Cross-domain Transfer)**
  - Build ingestion pipelines for InjecAgent.
  - (Optional) Build ingestion pipelines for RAGTruth / AgentDojo.
- [ ] **Study 3 (Optional Appendix)**: Implement robustness study with attacked variants.
- [x] **Scorer Validation**: Validated `scoring/harmbench_scorer.py` against official gold labels. XSTest κ=0.876 ✓; HarmBench κ=0.436 (methods limitation). Report: `artifacts/scorer_validation_report.txt`.
- [ ] **Boundary Dataset Expansion**: Expand from 130 → 180 items per PRD v2 §8 target.
- [ ] **Slurm Scripts for v2**: Create `slurm/run_study1_72b.sh` and `slurm/run_study2_72b.sh`.
- [ ] Run freeze_artifacts on `artifacts/mining/2026-04-28` to create v3_baseline snapshot.
