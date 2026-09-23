# Research workflow

## Project minimum

```text
README.md
pixi.toml
pixi.lock
configs/
src/ or project modules
experiments/
tests/
```

## Normal loop

1. Create a branch.
2. Check in the experimental config.
3. Run tests/smoke tests.
4. Run a small pilot.
5. Record metadata + Git commit.
6. Write outputs to a run-specific directory.
7. Inspect results.
8. Scale only after the pilot works.
9. Merge by pull request.

## Every important run should record

- model/checkpoint + revision;
- tokenizer revision;
- inference backend + version;
- Transformers/PyTorch versions;
- dtype/quantization;
- generation parameters;
- seed;
- GPU model;
- Git commit;
- experiment config;
- timestamp.

The starter `spar_inference.metadata` helper records much of the runtime automatically.

## Output directories

Prefer immutable run-specific directories:

```text
results/
  2026-09-23_qwen25-7b_prompt-v4_seed0/
    config.yaml
    metadata.json
    responses.jsonl
    summary.csv
```

Do not have multiple Pods overwrite one shared output file.

The same code path that runs 20 examples should run 200,000 examples; scaling should mostly be a config change.
