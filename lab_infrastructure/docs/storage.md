# Storage

## GitHub

Store code, configs, Pixi lockfiles, docs, and small fixtures.

## Global Volume

Use `/workspace` for durable shared assets:

```text
/workspace/
  shared/
    models/
    datasets/
    adapters/
  projects/
    <project>/
      durable-data/
      checkpoints/
      results/
```

Good uses: model weights, datasets, adapters, durable checkpoints, final outputs.

Do not treat the Global Volume as a POSIX parallel filesystem. Avoid concurrent writers to the same file and high-frequency small-file I/O.

## Pod-local scratch

```bash
export SPAR_SCRATCH=/root/scratch
mkdir -p "$SPAR_SCRATCH"
```

Use local disk for:
- Hugging Face/vLLM caches;
- temporary activations;
- intermediate shards;
- probe-training scratch;
- extracted hidden states.

Pattern:

```bash
cp /workspace/shared/datasets/data.parquet /root/scratch/
pixi run extract-activations --output /root/scratch/acts
mkdir -p /workspace/projects/my-project/results
cp -r /root/scratch/final_results /workspace/projects/my-project/results/
```

Anything important must be copied back before deleting the Pod.
