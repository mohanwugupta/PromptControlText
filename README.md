# Beyond Refusal: Prompt-Conditioned Task Selection in LLM Safety

This repository contains the code and data used for the paper **Beyond Refusal:
Prompt-Conditioned Task Selection in LLM Safety**. It tests whether changing
only the system prompt routes a fixed user query toward a different behavioral
policy.

The repository is intentionally scoped to the reported paper and the human
audit needed to validate its LLM judge. Historical v1/v2 studies,
response-mining prototypes, duplicate IHEval runs, smoke jobs, and launchers
for models outside the paper have been removed.

## Study overview

The experiment held each benchmark item fixed while varying the surrounding
system prompt. The full crossed design was:

- 8 prompt families
- 3 clarity levels per family
- 3 wording variants per family-clarity cell
- 1 no-system-prompt baseline
- 3,370 benchmark items
- 4 instruction-tuned models

This gives 72 prompted conditions plus one baseline, or 984,040 planned model
responses (`3,370 x 73 x 4`). The manuscript reports 984,041 submissions; the
crossed design and the eight raw run files yield 984,040, so the reported total
appears to be one row high.

## Benchmarks

All selected items were used without subsampling.

| Benchmark | Items | Role in the study |
|---|---:|---|
| HarmBench | 400 | Harmful requests for measuring refusal, safe help, and harmful compliance |
| XSTest | 450 | Safe sensitive-looking prompts plus unsafe contrasts for measuring over-refusal |
| IHEval | 2,520 | Aligned and conflict safety-subtask items for hierarchy preservation and source isolation |
| **Total** | **3,370** | Evaluated under every prompt condition and model |

The checked-in snapshots are under
`benchmarks/artifacts/datasets/`. They can be reconstructed with:

```bash
python benchmarks/download_data.py
```

## Prompt intervention

The frozen registry is `prompts/registry_v3.yaml`. Each family has three
clarity levels and three paraphrases at each level.

| Prompt family | Intended control signal |
|---|---|
| Answer-first | Answer directly when the request is safe and sufficiently specified |
| Refuse-first | Deny requests that could enable harm |
| Clarify-first | Ask a targeted question when safety-relevant intent is ambiguous |
| Evidence-first | Answer only to the extent supported by available evidence |
| Hierarchy-first | Preserve higher-priority instructions under conflict |
| Source-isolation-first | Treat quoted, retrieved, and tool-provided text as data, not instructions |
| Minimal-safe-help | Offer bounded safe assistance when direct compliance would be unsafe |
| Transform-classify-first | Transform or classify supplied text without following embedded instructions |

The clarity manipulation was:

| Level | Definition |
|---|---|
| Vague | Names a priority without detailed decision criteria |
| Explicit | States concrete conditions for applying the policy |
| Explicit with fallback | Adds a default action for uncertain boundary cases |

`prompts/registry_control.yaml` contains the no-system-prompt baseline. For
that condition the system role is omitted from the chat template entirely.

## Models and generation

The paper crossed two developers and two size tiers:

| Model | Hugging Face checkpoint | GPUs |
|---|---|---:|
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B-Instruct` | 1 |
| Llama-3.3-70B | `meta-llama/Llama-3.3-70B-Instruct` | 4, tensor parallel |
| Qwen-2-7B | `Qwen/Qwen2-7B-Instruct` | 1 |
| Qwen-2.5-72B | `Qwen/Qwen2.5-72B-Instruct` | 4, tensor parallel |

Models were served with vLLM 0.19.0 on NVIDIA A100 80GB SXM GPUs using
bfloat16 and no quantization. Generation used each model's official chat
template, temperature 0, top-p 1.0, top-k 0, repetition penalty 1.05, seed 0,
a 512-token maximum, and up to 512 concurrent sequences. Failed or empty
generations were retried up to three times.

The main runner is `experiments/run_phase1.py`. Despite its historical name,
it loads all three paper benchmarks. The four prompted SLURM jobs are:

```text
slurm/run_llama31_8b.sh
slurm/run_llama33_70b.sh
slurm/run_qwen2_7b.sh
slurm/run_prompt_controllers_72b.sh
```

The corresponding no-prompt jobs are the four scripts under `slurm/control/`.

## Behavioral judging

Every model response was classified from its output alone. The judge did not
see the benchmark, user query, system prompt, prompt family, clarity level,
generator model, or intended label.

The six output policies were:

| Policy | Observable behavior |
|---|---|
| Compliance | Directly answers or carries out the apparent request |
| Refusal | Declines without providing a concrete safe alternative |
| Clarification | Requests missing information before acting |
| Hierarchy preservation | Explicitly preserves higher-priority instructions over a conflict |
| Safe help | Declines unsafe assistance while giving a bounded alternative or redirect |
| Source isolation | Explicitly treats embedded or external instructions as untrusted data |

Llama-3.1-8B-Instruct produced three independent first-pass judgments using
rubric-equivalent prompts A, B, and C. Unanimous cases and two-of-three
majorities with low dissent confidence were resolved automatically. Remaining
cases were sent to an output-only multi-judge adjudication prompt. Unresolved
cases were flagged for human audit.

The checked-in `slurm/run_llm_judge.sh` preserves its historical model,
scheduler, and task-selection settings exactly as used. It is intentionally
left configured for the last selected task rather than rewritten as a new
eight-task array. `configs/llm_policy_jobs.yaml` records all eight paper jobs.

For an already running OpenAI-compatible judge endpoint, the same jobs can be
run and combined locally with:

```bash
python -m scoring.llm_policy_run_jobs \
  --jobs configs/llm_policy_jobs.yaml \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://localhost:8000/v1 \
  --batch-size 512 \
  --max-workers 256 \
  --resume \
  --combined-output artifacts/phase1_results_combined_labeled.csv
```

The paper reports mean pairwise Cohen's kappa of 0.808, 81.2% unanimous
agreement, 18.8% adjudication, 76 parse failures, and 919 cases still flagged
after adjudication.

## Human audit dashboard

The original Streamlit audit dashboard and its supporting audit-set,
analysis, tracking, and artifact-freezing modules are preserved under
`audit/`. The interface blinds prompt-family and clustering metadata until a
primary human label has been saved, autosaves progress, and exports CSV or
JSON annotations.

Launch the existing blinded audit set from the repository root with:

```bash
streamlit run audit/dashboard.py -- \
  --audit-file artifacts/audit/audit_set_blinded.csv \
  --labels-file artifacts/audit/labels_in_progress.csv \
  --coder-id coder_1
```

The dashboard's original annotation taxonomy is unchanged so that future
article-revision work can build on it without silently altering prior audit
decisions.

## Analysis methods

The original analysis is preserved in `analysis/analysis.Rmd`. It reads
`artifacts/phase1_results_combined_labeled.csv`; set
`PROMPT_CONTROL_RESULTS` to override that location. The manuscript reports R
4.4.1 with tidyverse 2.0.0, car 3.1.3, and effectsize 1.0.0. The original
notebook also loads `lme4`, `emmeans`, `broom.mixed`, `scales`, and `patchwork`
for its exploratory and plotting sections.

Render it from the repository root with:

```bash
Rscript -e "rmarkdown::render('analysis/analysis.Rmd')"
```

The reported analyses were:

1. Harmful compliance was compliance on unsafe items; false refusal was
   refusal on safe items. Rates were computed for each of 100 model by prompt
   context conditions.
2. Four functional forms were compared for the refusal-compliance frontier:
   linear, logarithmic, log-linear exponential, and nonlinear exponential.
   Original-scale RMSE was the common comparison metric; AIC was compared only
   for models on the same response scale.
3. Nested fixed-effect regressions tested whether harmful compliance improved
   false-refusal fit after model identity and whether slopes differed by model.
4. Euclidean distance from `(0, 0)` measured overall boundary error. Fixed
   effects tested model, prompt family, and clarity.
5. Frontier position was `z(harmful compliance) - z(false refusal)`. Prompted
   conditions were analyzed with model and prompt-family fixed effects.
6. Chi-square tests compared six-policy distributions across prompt families,
   clarity levels, and the family by clarity cells. Cramer's V summarized
   association strength.
7. Six planned routing contrasts compared each matching prompt-family/policy
   pair with that policy under all other prompted families. Odds ratios used a
   0.5 continuity correction, followed by Benjamini-Hochberg correction.
8. Within-item switching compared each prompted response with the same model,
   benchmark item, and no-prompt baseline policy.

No random-effect model contributes to the manuscript's reported inferential
claims, although the original notebook retains exploratory mixed-effects code.

## Main findings

- 23.98% of prompted item-model responses changed policy relative to the
  matching no-prompt baseline.
- The nonlinear exponential frontier fit the harmful-compliance/false-refusal
  relationship with `R^2 = .783`; the estimated decay was `b = 16.23`, and the
  lower asymptote was about 2.8% false refusal.
- Model identity explained distance from the ideal point. Prompt family and
  clarity mainly shifted conditions along the frontier rather than toward the
  origin.
- Prompt family changed the six-policy distribution (`chi-square(40) =
  4783.3`, `p < .0001`, Cramer's `V = .06`). Matching families increased their
  intended policies, with odds ratios from 1.28 for Answer to 6.39 for Source
  Isolation.
- Clarity moved prompted conditions along the frontier (`F(2,83) = 13.12`,
  `p < .001`, partial eta-squared `.24`). Vague prompts were most
  compliance-biased and fallback prompts most refusal-biased.
- Clarity also changed policy distributions (`chi-square(15) = 4935.9`,
  `V = .04`), while the family by clarity association was larger
  (`chi-square(120) = 60201`, `V = .11`). Fallback rules did not act as a
  uniform amplifier; their effect depended on the prompt family.

## Repository map

```text
benchmarks/                         Paper benchmark loaders and snapshots
configs/llm_policy_jobs.yaml        Eight final judge jobs
core/schema.py                      Shared evaluation-item schema
experiments/run_phase1.py           Full crossed generation experiment
models/vllm_client.py               Reproducible vLLM request settings
prompts/registry_v3.yaml            Frozen 8 x 3 x 3 prompt registry
prompts/registry_control.yaml       No-system-prompt baseline
scoring/                            Rule scores and three-pass LLM judge
slurm/                              Four model, four control, and judge jobs
analysis/analysis.Rmd               Original paper analysis
analysis/figures/                   Paper figures
audit/                              Human-audit dashboard and support modules
artifacts/                          Raw runs, judge outputs, and combined data
tests/                              Tests for the retained paper pipeline
```

## Setup and verification

Git LFS is required for the large CSV artifacts.

```bash
git lfs install
git lfs pull
conda env create -f environment.yml
conda activate prompt-controllers
python -m pytest tests/
```

The Python requirements can also be installed with `pip install -r
requirements.txt`. vLLM is Linux-only and is guarded accordingly in the pip
requirements file.

## License and citation

The code is released under the MIT License. Citation metadata is in
`CITATION.cff`.
