# RunPod

## Local CLI setup

```bash
pixi global install runpodctl
runpodctl update
runpodctl doctor
```

For scripts, use:

```bash
export RUNPOD_API_KEY="..."
```

Treat the key like a password. Use restricted keys with the minimum permissions needed.

## Useful commands

```bash
runpodctl gpu list
runpodctl datacenter list
runpodctl pod list
runpodctl pod list --all
runpodctl pod get <pod-id>
runpodctl pod logs <pod-id> --follow
runpodctl ssh info <pod-id>
runpodctl pod start <pod-id>
runpodctl pod stop <pod-id>
runpodctl pod delete <pod-id>
```

Use ordinary `ssh` for the shell after obtaining connection info.

## Pod creation

The CLI evolves, so always inspect:

```bash
runpodctl pod create --help
```

A typical current pattern is:

```bash
runpodctl pod create \
  --name mohan-persistence-pilot \
  --template-id <SPAR_TEMPLATE_ID> \
  --gpu-id "<GPU_ID>" \
  --gpu-count 1 \
  --wait
```

If your installed version supports a hard auto-termination option, use it for interactive Pods.

## Lab defaults

- One GPU unless the experiment needs more.
- Choose the cheapest GPU that fits the VRAM/throughput requirement.
- Name Pods `<person>-<project>-<purpose>`.
- Attach the SPAR Global Volume when durable shared data are needed.
- Put high-I/O scratch on Pod-local storage.
- Delete Pods when finished.

## Global Volume

The lab volume is `spar-super-lab-workspace`. Treat it as persistent shared/object-backed storage, not as Della-style fast scratch.
