# 5-minute quickstart

## One-time local setup

```bash
curl -fsSL https://pixi.sh/install.sh | bash
pixi global install runpodctl
runpodctl update
runpodctl doctor
```

Each person should use their own RunPod credentials. Do not share API keys.

## Start a GPU

Inspect capacity first:

```bash
runpodctl gpu list
runpodctl datacenter list
runpodctl pod list
```

For interactive work, deploy the shared SPAR template in the RunPod console, attach `spar-super-lab-workspace`, request **one inexpensive appropriate GPU**, and name the Pod:

```text
<person>-<project>-<purpose>
```

For CLI creation, check the current syntax first:

```bash
runpodctl pod create --help
```

## Work on the Pod

Keep the repo and high-I/O working files on local Pod disk:

```bash
mkdir -p /root/projects /root/scratch
cd /root/projects
git clone <REPO_URL>
cd <REPO>

pixi run --locked test
pixi run --locked <project-task>
```

For the starter inference stack:

```bash
MODEL=<served-model-name> pixi run inference-smoke
```

## Storage rule

- GitHub: code/configs/docs
- Global Volume (`/workspace`): durable datasets, model assets, final results
- Local Pod disk: caches, activations, temporary shards, fast scratch

## Finish

Copy important outputs to persistent storage, then delete the Pod:

```bash
runpodctl pod list
runpodctl pod delete <pod-id>
```
