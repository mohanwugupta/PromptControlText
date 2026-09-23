# Compute policy

## Default

- **1 GPU** per interactive/development Pod.
- Use the **cheapest GPU that fits the job**.
- Pilot on small models before large-model replication.
- Delete the Pod when finished.
- Use a runtime/termination cap when available.

## Ask in Slack first when

- expected compute cost is over about **$25**;
- requesting **4+ GPUs**;
- using H100/H200/B200-class GPUs when cheaper cards may suffice;
- running longer than **24 hours**;
- writing hundreds of GB or more of persistent data.

## Pod naming

```text
<person>-<project>-<purpose>
```

## Before leaving for the day

```bash
runpodctl pod list
```

If a Pod is not doing useful work, persist important outputs and delete it.

Do not preload the entire grant into RunPod. Keep a smaller prepaid balance with low-balance alerts and review spending weekly.
