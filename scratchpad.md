# Scratchpad

Use this document to track unstructured notes, intermediate thoughts, and scratchpad updates during development of the promptControlText project.

## Current Focus
- [x] Initial Project Setup & TDD Pipeline Constraints.
- [x] Slurm Script Generation & Extracted vLLM Generator Class.
- [ ] Transitioning into orchestrating real live tests mapping HarmBench/XSTest metrics across the SLURM endpoint.

## Notes & Intermediate Issues
- Model schema successfully maps to 4 GPU TP=4 inference with Qwen2.5-72B.
- Make sure `pytest` passes flawlessly before appending any new features to `models/` or `scoring/`.
- Conda Environment targeted: `promptControlText`
- Offline HuggingFace mode (`HF_HUB_OFFLINE`) is enforced via the slurm bash setups.
