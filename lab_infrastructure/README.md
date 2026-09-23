# SPAR Super Lab compute infrastructure

This folder is a standalone draft of the shared compute/code infrastructure for the SPAR Super Lab. It is staged inside PromptControlText because that repository contains the inference stack we want to extract. The intended long-term home is a dedicated infrastructure repository.

## Design

1. **RunPod** provides on-demand GPU Pods.
2. **Pixi** gives each research project a reproducible environment and lockfile.
3. **GitHub** stores code, configs, and reviews.
4. **RunPod Global Volume** stores durable shared assets and final outputs.
5. **Pod-local disk** is fast scratch for caches, activations, temporary shards, and other high-I/O work.
6. **vLLM + an OpenAI-compatible client** is the shared inference path.

The reusable inference pieces here were extracted from PromptControlText:
- `models/vllm_client.py`;
- `configs/model_registry.yaml`;
- `slurm/run_model_generic.sh`;
- the concurrency/checkpointing pattern in `experiments/run_phase1.py`.

The extracted version removes benchmark-specific logic and strengthens reproducibility metadata.

## Start here

1. [Quickstart](docs/quickstart.md)
2. [RunPod](docs/runpod.md)
3. [Pixi](docs/pixi.md)
4. [Storage](docs/storage.md)
5. [Research workflow](docs/research_workflow.md)
6. [Compute policy](docs/compute_policy.md)
7. [Mechanistic interpretability](docs/mech_interp.md)

The `template/` directory is a starter project that can be copied into a new research repository.

## Scope

We centralize inference because it is stable and broadly reused. We do **not** yet maintain a large shared mech-interp abstraction. Mechanistic-interpretability workflows vary enough that the safer initial pattern is to keep known-good recipes inside each project and promote utilities into shared code only after they recur across multiple projects.
