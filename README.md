# Prompt-Level Safety Controllers for LLMs

This repository implements a scientifically defensible evaluation pipeline to measure whether short control prompts can shift the safety boundaries, refuse/comply policies, and hierarchy-following behavior of Large Language Models (LLMs). Follows a strict Test-Driven Development (RED→GREEN) framework.

## Project Structure
- `benchmarks/`: Connectors and parsers mapping safety datasets (e.g., XSTest) to a unified canonical schema (`EvalItem`).
- `core/`: Common Pydantic data schemas representing individual items, metrics, and prompts.
- `models/`: Interface layers for generation and chat endpoints. Features local mock caches for testing, and `vllm_client.py` for scalable background SLURM cluster generation using OpenAI compatible endpoints.
- `prompts/`: YAML-backed frozen prompt registry maintaining hash-locked baseline comparisons (Refuse-first, Clarify-first, Hierarchy-first).
- `scoring/`: Rule-based and model-driven evaluation parsers bridging raw output text into defined `compliance`, `refusal`, `clarification`, and `abstention` floating point metrics.
- `analysis/`: Cross-domain grouped aggregations leveraging `pandas` to flatten and compute experiment results.
- `slurm/`: Job scheduling scripts customized for 4x GPU Tensor Parallel runs across 72-Billion parameter models on the internal compute grid offline.
- `tests/`: Extensive Pytest suites establishing RED/GREEN constraints on dataset loaders, hashing, modeling, and aggregation correctness.

## Setup
### Local Testing
```bash
pip install -r requirements.txt
python -m pytest tests/
```

### Slurm Execution
Execute from the cluster environment after modifying model payload flags:
```bash
sbatch slurm/run_prompt_controllers_72b.sh
```
