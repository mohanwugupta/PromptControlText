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

## Pending (Execution & Downstream)
- [ ] **Phase 3 Scaffolding (Cross-domain Transfer)**
  - Build ingestion pipelines for InjecAgent.
  - (Optional) Build ingestion pipelines for RAGTruth / AgentDojo.
- [ ] **Study 1 Live Execution**: Run `run_study1.py` on cluster with Qwen2.5-72B-Instruct against full boundary dataset.
- [ ] **Study 2 Live Execution**: Run `run_study2.py` on cluster with full IHEval data.
- [ ] **Study 3 (Optional Appendix)**: Implement robustness study with attacked variants.
- [x] **Scorer Validation**: Validated `scoring/harmbench_scorer.py` against official gold labels. XSTest κ=0.876 ✓ (primary); HarmBench κ=0.436 (hard regex ceiling — report as methods limitation). Fixed duplicate function definition bug (broken `len > 600` fallback). Report: `artifacts/scorer_validation_report.txt`.
- [x] **Manual Audit**: Stratified manual audit of Study 1 & 2 outputs to validate automated scoring.
- [ ] **Boundary Dataset Expansion**: Expand from 130 → 180 items per PRD v2 §8 target.
- [ ] **Statistical Analysis**: Implement mixed-effects models for confirmatory analysis (logistic mixed-effects regression).
- [ ] **Slurm Scripts for v2**: Create `slurm/run_study1_72b.sh` and `slurm/run_study2_72b.sh`.
