# Mechanistic interpretability

Start **recipe-first**, not abstraction-first.

## Initial standard

Each mech-interp project should have one small end-to-end smoke test that:

1. loads a small model;
2. tokenizes one prompt;
3. runs one forward pass;
4. extracts a named hidden representation;
5. prints/saves the tensor shape;
6. writes the result to local scratch.

The starter example uses Hugging Face `output_hidden_states=True` so the baseline is transparent.

## Data path

```text
Global Volume -> stage inputs -> local scratch -> activations/analysis -> persist useful outputs -> Global Volume
```

## What becomes shared later

Promote utilities only after multiple projects need the same thing, e.g.:
- canonical token-position selection;
- standardized activation metadata;
- model-name -> module-name mapping;
- batched activation extraction;
- shard manifests/checkpointing;
- probe train/eval interfaces;
- common causal-intervention hooks.

## Every activation dataset should record

- model/checkpoint + revision;
- layer/module/hook point;
- token-selection rule;
- dtype;
- tensor shape;
- source item IDs;
- code Git commit.
