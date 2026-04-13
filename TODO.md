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

## Pending (Execution & Downstream Implementations)
- [x] **Phase 1 Execution (Refusal Boundaries)**
  - [x] Integrate specific live endpoints for parsing full XSTest and HarmBench / JailbreakBench.
  - [x] Bridge the testing framework natively into a `experiments/run_phase1.py` runner called by the Slurm script.
- [x] **Phase 2 Scaffolding (Hierarchy-sensitive)**
  - [x] Ingest the IHEval substrate structure into `benchmarks/iheval.py`.
  - [x] Validate scoring behavior accurately identifies hierarchy overrides / stop compliance.
- [ ] **Phase 3 Scaffolding (Cross-domain Transfer)**
  - Build ingestion pipelines for InjecAgent.
  - (Optional) Build ingestion pipelines for RAGTruth / AgentDojo.
- [x] **Model Evaluators & Auditing**
  - [x] Replace the naive rules-based scoring heuristics with generalized evaluation interfaces (HarmBench compatible scorers).
  - [x] Add native manual reporting outputs/tables plotting.
- [x] **Plot Generation**
  - [x] Implement figure generation blocks summarizing Refusal vs. Compliance Boundaries across target prompts.
