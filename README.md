# Prompt Controllers as Policy Routers

This repository implements a scientifically defensible evaluation pipeline to measure whether short control prompts can steer LLMs to select the *correct safety policy* on ambiguous and adversarial inputs. The primary outcome is **policy-routing accuracy**: the fraction of items on which a controller induces the intended policy (answer, refuse, clarify, minimal_safe_help, or hierarchy_preserve).

The project follows a strict Test-Driven Development (RED→GREEN) framework. All 154 tests pass.

---

## Paper thesis

> Prompt-level controllers differ not just in how often a model refuses, but in *which policy* the model routes to. Controller clarity (vague vs. explicit vs. explicit-with-fallback) modulates routing accuracy, particularly on ambiguous and hierarchy-conflict items.

The two main research questions are:

- **RQ1** — Do controller families induce distinct policy distributions on the same item?
- **RQ2** — Does clarity level improve routing accuracy, especially under ambiguity and conflict?

---

## Policy taxonomy

| Label | Meaning |
|---|---|
| `answer` | Model complies and provides a direct response |
| `refuse` | Model declines the request |
| `clarify` | Model requests clarification before proceeding |
| `minimal_safe_help` | Model provides limited assistance with an explicit limitation or redirect |
| `hierarchy_preserve` | Model prioritises an explicit system instruction over the user turn |

---

## Controller families and clarity levels

Four families × three clarity levels × three paraphrases = **36 frozen prompts** (see `prompts/registry_v2.yaml`).

| Family | Vague | Explicit | Explicit + fallback |
|---|---|---|---|
| Refuse-first | ✓ | ✓ | ✓ |
| Clarify-first | ✓ | ✓ | ✓ |
| Minimal-safe-help | ✓ | ✓ | ✓ |
| Hierarchy-first | ✓ | ✓ | ✓ |

---

## Study design

### Study 1 — Policy routing on boundary items
- **Substrate**: 130-item annotated boundary dataset (`artifacts/datasets/boundary_dataset.csv`)
- **Design**: Within-item repeated-measures across all 36 prompts
- **Primary DV**: routing accuracy per family × clarity cell
- **Runner**: `experiments/run_study1.py`

### Study 2 — Policy routing under hierarchy conflict
- **Substrate**: IHEval conflict items
- **Design**: Same within-item repeated-measures structure
- **Primary DV**: hierarchy_preserve routing rate under each controller
- **Runner**: `experiments/run_study2.py`

### Boundary dataset composition (130 items)
| Subset | N | Source |
|---|---|---|
| XSTest safe/sensitive | 40 | Boundary around safe-but-sounds-harmful |
| HarmBench unsafe | 40 | Clear refusal cases |
| IHEval conflict | 20 | System vs user instruction conflict |
| Quoted / analysis | 10 | Requests about harmful content, not requests for it |
| Ambiguous intent | 10 | Genuinely underspecified requests |
| Minimal-safe-help scenarios | 10 | Cases where partial help is correct |

---

## Repository structure

```
core/schema.py             EvalItem Pydantic schema (policy_label, ambiguity_level,
                           context_condition, clarity_level fields added for v2)
prompts/registry_v2.yaml   36 frozen production prompts (FROZEN — do not edit)
prompts/registry.py        YAML loader + render_prompt_v2()
scoring/policy_classifier.py  Pattern-based 5-label policy classifier
benchmarks/boundary_dataset.py  Loader for the custom boundary dataset
benchmarks/{harmbench,iheval,xstest}.py  Standard benchmark loaders
analysis/metrics.py        compute_routing_accuracy(), compute_secondary_metrics()
analysis/stats.py          run_routing_lmm(), run_routing_glm(),
                           compute_bootstrap_ci(), format_results_table()
analysis/plots.py          plot_policy_distribution(), plot_routing_accuracy()
experiments/run_study1.py  Study 1 runner (boundary items)
experiments/run_study2.py  Study 2 runner (IHEval conflict items)
models/client.py           OpenAI-compatible generation client
models/vllm_client.py      vLLM endpoint client for cluster runs
slurm/                     SLURM job scripts for 72B model runs
artifacts/datasets/        boundary_dataset.csv and phase results CSVs
tests/                     155 pytest tests (all GREEN)
```

---

## Setup

### Local testing
```bash
pip install -r requirements.txt
python -m pytest tests/
```

### Cluster execution (Study 1)
```bash
python experiments/run_study1.py \
    --generator-model Qwen2.5-72B-Instruct \
    --output-file artifacts/study1_results.csv
```

### Cluster execution (Study 2)
```bash
python experiments/run_study2.py \
    --generator-model Qwen2.5-72B-Instruct \
    --output-file artifacts/study2_results.csv
```

### Confirmatory statistical analysis (after live runs)
```python
from analysis.stats import run_routing_lmm, format_results_table, compute_bootstrap_ci
import pandas as pd

df = pd.read_csv("artifacts/study1_results.csv")
result = run_routing_lmm(df)
print(format_results_table(result))
print(compute_bootstrap_ci(df))
```

---

## Statistical analysis plan (PRD v2 §17)

- **Primary model**: Linear mixed model (`run_routing_lmm`) — routing_correct ~ C(prompt_family) + C(clarity_level) + C(context_condition), random intercepts per item_id. This is a linear probability model (LPM) approximation of the GLMM; coefficient directions are cross-checked with `run_routing_glm` (logistic, no random effects).
- **Bootstrap**: `compute_bootstrap_ci` provides percentile-bootstrap 95% CIs for overall routing accuracy per PRD v2 §17.
- **Secondary DVs**: harmful compliance rate, false refusal rate, clarification rate (see `analysis/metrics.py`).
- **Pending**: manual audit of classifier labels and inter-rater reliability before results are treated as final.

---

## Key design decisions

- **Policy classifier is pattern-based with priority order**: hierarchy_preserve > minimal_safe_help > clarify > refuse > answer. This is intentionally conservative and fast; all automated classifications require manual audit of a stratified sample before paper submission.
- **Prompts are frozen after exploratory phase**: `registry_v2.yaml` is marked FROZEN. No prompt edits are permitted once study execution begins.
- **Within-item repeated measures**: each boundary item is evaluated under all 36 prompts, giving the primary contrast (family × clarity) maximum power at fixed item count.
