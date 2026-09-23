# Pixi

Each research repository owns its own environment. There is **no single lab-wide Python environment**.

Commit:

```text
pixi.toml
pixi.lock
```

## Core commands

```bash
pixi install
pixi run test
pixi run --locked test
pixi run <task>
pixi add numpy
pixi add --pypi openai
pixi update
```

For important runs, prefer `--locked`.

## Use tasks as the user interface

```toml
[tasks]
test = "pytest -q"
lint = "ruff check ."
serve = "bash scripts/serve_vllm.sh"
inference-smoke = "python examples/inference_smoke.py"
```

Then mentees run:

```bash
pixi run --locked test
pixi run --locked serve
pixi run --locked inference-smoke
```

## Why per-project environments

Different projects may require incompatible versions of vLLM, Transformers, TransformerLens, NNsight, SAE tooling, or CUDA-adjacent packages. Keeping dependencies project-local prevents one project's upgrade from breaking another.

If dependency stacks later diverge sharply, use Pixi's multiple environments/features rather than forcing everything together.
