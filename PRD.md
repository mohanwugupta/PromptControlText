# PRD: Prompt-Level Safety Controllers for LLMs

## 1. Overview

**Project title**
**Prompt-Level Safety Controllers: Measuring how prompts shift refusal, hierarchy-following, and cross-domain safety behavior in LLMs**

**Core objective**
Build a scientifically defensible evaluation pipeline to test whether short control prompts can function as **lightweight safety controllers** that alter LLM behavior at the safety boundary.

**Why this matters**
Recent work on instruction hierarchy argues that safe behavior depends on correctly prioritizing higher-level instructions over lower-level or untrusted ones, especially when they conflict. IHEval operationalizes that problem with 3,538 examples across nine tasks and shows that models degrade sharply under conflict. HarmBench, JailbreakBench, XSTest, InjecAgent, and RAGTruth give complementary ways to measure harmful compliance, over-refusal, indirect prompt injection, and unsupported claims under retrieved evidence. ([arXiv][1])

**Primary research claim**
Prompts can induce distinct **safety-relevant behavioral policies**, but those policies differ in robustness, calibration, and transfer.

**Secondary research claim**
A meaningful fraction of apparent safety failures may reflect **weak or underspecified control hierarchy** rather than a stable dangerous objective.

---

## 2. Product goals

### Goal 1

Demonstrate that short prompts can shift the model into distinct safety-relevant behavioral modes.

### Goal 2

Test whether clearer hierarchy wording reduces failures under conflicting instructions.

### Goal 3

Measure whether prompt-based safety controls transfer across domains rather than acting as narrow benchmark patches.

### Goal 4

Produce a reusable, testable scientific codebase with frozen prompts, reproducible runs, auditable logs, and pre-specified analyses.

---

## 3. Non-goals

This project will **not**:

* claim to identify internal mechanistic representations
* claim evidence of deception, scheming, or self-preservation
* rely on cherry-picked demos
* optimize prompts directly on the final held-out evaluation set
* depend on one-off manual judgments without calibration

---

## 4. Research questions and hypotheses

### RQ1

Can brief prompts induce distinct safety-relevant behavioral modes?

**H1**
Prompt families will produce separable behavioral profiles on refusal, clarification, abstention, and override compliance.

### RQ2

Do safety failures shrink when instruction priority is made explicit?

**H2**
Prompts with explicit hierarchy and conflict-handling clauses will improve performance on instruction-conflict tasks relative to vague safety reminders.

### RQ3

Do prompt-based safety controls transfer across domains?

**H3**
Some controllers will generalize across multiple benchmarks, but transfer will be selective and often come with utility costs.

### RQ4

Are some prompt families better calibrated than others?

**H4**
“Refuse-first” prompts will reduce harmful compliance but raise false refusals; “clarify-first” and “hierarchy-first” prompts will show better safety-utility tradeoffs.

---

## 5. Scope and phase plan

## Phase 1: Safety modes at the refusal boundary

Use text-only safety benchmarks to test whether prompt families shift refusal and over-refusal behavior.

**Benchmarks**

* **XSTest** for exaggerated safety / over-refusal: 250 safe prompts and 200 unsafe contrast prompts. ([arXiv][2])
* **HarmBench** or **JailbreakBench** for harmful compliance and jailbreak robustness. HarmBench is a standardized evaluation framework for automated red teaming and robust refusal; JailbreakBench includes 100 harmful behaviors and standardized artifacts/scoring. ([arXiv][3])

**Deliverable**
Evidence that prompt families alter refusal calibration, not just style.

## Phase 2: Hierarchy-sensitive control

Use instruction-conflict tasks to test whether explicit priority ordering reduces failures.

**Benchmarks**

* **IHEval** as the main substrate. It covers system messages, user messages, conversation history, and tool outputs, with conflict and non-conflict cases. ([arXiv][1])
* Small custom conflict set for stop / override / do-not-use-tool / do-not-disclose cases.

**Deliverable**
Evidence that control wording matters specifically because it changes conflict resolution.

## Phase 3: Transfer across safety domains

Test the strongest prompt families from Phases 1–2 across richer safety settings.

**Benchmarks**

* **InjecAgent** for indirect prompt injection: 1,054 test cases spanning 17 user tools and 62 attacker tools. ([arXiv][4])
* Optional extension: **AgentDojo**, which includes 97 realistic tasks and 629 security test cases. ([arXiv][5])
* Optional extension: **RAGTruth**, with nearly 18,000 manually annotated RAG responses for unsupported or contradictory claims. ([arXiv][6])

**Deliverable**
A transfer matrix showing which controllers are broad, narrow, or costly.

---

## 6. Prompt families

Prompt families must be **pre-registered and frozen** before confirmatory evaluation.

### Family A: Refuse-first

Prioritize refusing potentially unsafe requests.
Expected signature: low harmful compliance, higher false refusal.

### Family B: Clarify-first

Ask brief clarification questions under ambiguity or conflict before acting.
Expected signature: lower unsupported action, higher clarification.

### Family C: Minimal-safe-help

Provide the safest compliant help possible without fulfilling harmful intent.
Expected signature: moderate refusal, moderate utility preservation.

### Family D: Hierarchy-first

Explicitly prioritize higher-level instructions, later safety constraints, stop requests, and untrusted tool outputs appropriately.
Expected signature: best performance on instruction-conflict and injection tasks.

**Scientific rule**
Each family must contain **at least 3 paraphrases**, and confirmatory claims are made over the **family average**, not one favorite prompt. This is required to avoid prompt cherry-picking.

---

## 7. Experimental design

## Common design choices

### Controlled variables

* same base model across all prompt conditions within a study
* same generation settings across conditions
* same item set across conditions
* frozen prompts
* fixed random seed for deterministic runs
* secondary robustness runs with a small number of stochastic repeats

### Recommended generation settings

* **Primary analysis**: temperature = 0
* **Robustness analysis**: temperature > 0 with 3–5 repeats per item

### Unit of analysis

Primary unit: **item-level outcome**
Secondary unit: **prompt-family mean across paraphrases**

### Design structure

Use a **within-item repeated-measures design** wherever possible: the same benchmark item is run under each prompt family. This improves power and reduces noise from item differences.

---

## 8. Outcome variables

## Phase 1 metrics

* harmful compliance rate
* refusal rate on unsafe items
* false refusal rate on safe items
* clarification rate on ambiguous items
* utility-preserving safe response rate

## Phase 2 metrics

* hierarchy-following accuracy
* override compliance
* stop compliance
* unsafe continuation rate
* conflict acknowledgment rate
* clarification under unresolved conflict

## Phase 3 metrics

* attack success rate under prompt injection
* policy violation rate
* tool misuse rate
* unsupported claim rate
* abstention / qualification rate
* task completion rate
* excessive passivity / over-refusal cost

**Composite metrics**

* safety benefit index
* utility cost index
* net calibration score

---

## 9. Statistical analysis plan

All primary analyses should be specified before confirmatory runs.

### Primary analyses

Use mixed-effects models where feasible:

* logistic mixed-effects regression for binary outcomes
* linear mixed-effects models for continuous outcomes
* random intercepts for item and prompt paraphrase
* random intercepts for model run if stochastic repeats are used

### Key contrasts

* each prompt family vs baseline
* hierarchy-first vs vague safety reminder
* clarify-first vs refuse-first
* phase-to-phase transfer effect by domain

### Uncertainty estimates

Report:

* effect sizes
* 95% confidence intervals
* exact p-values where appropriate
* bootstrap confidence intervals as a robustness check

### Multiple comparisons

Use a pre-specified correction method for secondary contrasts.

### Manual audit

A stratified manual audit of a subset of outputs is required to validate automated scoring, especially for:

* ambiguous refusals
* partial compliance
* minimal-safe-help behavior
* prompt-injection outputs

If using an LLM judge anywhere, it must first be validated against human labels on a held-out audit set.

---

## 10. Scientific defensibility rules

These are mandatory.

### Rule 1: Freeze prompt families

No editing prompts after viewing held-out confirmatory results.

### Rule 2: Separate dev and confirmatory evaluation

Use a development slice for debugging and prompt-family refinement, then freeze and run on held-out items.

### Rule 3: Preserve all transcripts

Store raw prompt, model output, parsed score, metadata, and evaluation version for every run.

### Rule 4: Avoid one-prompt conclusions

Claims must be based on prompt-family aggregates.

### Rule 5: Prefer benchmark-native scoring

Use official or benchmark-aligned scoring whenever available before inventing new scorers. HarmBench and JailbreakBench were built to standardize this problem, which is exactly why they are attractive starting points. ([arXiv][3])

### Rule 6: Document all exclusions

Every filtered item, parse failure, timeout, or evaluator disagreement must be logged.

### Rule 7: Treat transfer as the real test

A prompt that helps on one benchmark and fails elsewhere is a local patch, not a strong control signal.

---

## 11. TDD and RED→GREEN workflow

This project should be built under strict **test-driven development**.

## Development principle

For every new module:

1. write failing tests first (**RED**)
2. implement the smallest change that passes (**GREEN**)
3. refactor with tests still passing

No production code lands without a prior failing test, except trivial wiring.

## Required test layers

### A. Dataset loader tests

Write first.

* loads expected number of items
* preserves benchmark labels and metadata
* stratified split function is deterministic
* no duplicated item IDs after split
* benchmark-specific parsers map into a common schema

### B. Prompt registry tests

Write first.

* all prompt families load from versioned config
* each family has required paraphrase count
* prompt IDs are unique and frozen
* templates render exactly once with no missing variables
* prompt text hashes are stable

### C. Model wrapper tests

Write first.

* same input returns cached output if caching enabled
* metadata logs model name, version, temperature, seed, timestamp
* retries do not silently alter generation settings
* failed calls return typed errors, not null blobs

### D. Scoring tests

Write first.

* benchmark-native scorers produce valid outputs on known examples
* custom scorers match hand-labeled toy cases
* parser handles abstention, refusal, clarification, and partial compliance
* manual-audit export is correct

### E. Runner / experiment orchestration tests

Write first.

* full batch run produces one row per item × prompt × repeat
* interrupted runs can resume without duplication
* holdout sets are never included in dev runs
* randomization is reproducible

### F. Analysis tests

Write first.

* aggregate metrics match known toy datasets
* confidence intervals are reproduced on fixtures
* plots fail loudly on missing conditions
* statistical model code handles perfect separation gracefully

### G. End-to-end smoke tests

Write first.

* tiny fixture benchmark runs through the whole pipeline
* output tables, plots, and logs are all produced
* one intentionally bad prompt produces a predictable failure signal

---

## 12. RED→GREEN milestones by module

## Milestone 1: Core schema

**RED**

* tests for unified item schema fail

**GREEN**

* implement canonical schema:

  * `item_id`
  * `benchmark`
  * `domain`
  * `input_text`
  * `gold_label`
  * `prompt_family`
  * `prompt_variant`
  * `model_output`
  * `score`
  * `metadata`

## Milestone 2: Benchmark ingestion

**RED**

* tests for XSTest, HarmBench/JBB, and IHEval ingestion fail

**GREEN**

* implement importers and schema mapping

## Milestone 3: Prompt registry

**RED**

* tests for frozen prompt config fail

**GREEN**

* implement YAML/JSON prompt registry with hash locking

## Milestone 4: Model execution

**RED**

* tests for caching, retries, and metadata fail

**GREEN**

* implement API wrapper

## Milestone 5: Scoring

**RED**

* tests for refusal / compliance / clarification scoring fail

**GREEN**

* implement scorer adapters and audit workflow

## Milestone 6: Analysis

**RED**

* tests for metric computation and plotting fail

**GREEN**

* implement summaries, regressions, plots

## Milestone 7: Confirmatory guardrails

**RED**

* tests that holdout/dev leakage is blocked fail

**GREEN**

* enforce split isolation in code

---

## 13. Software architecture

Recommended modules:

* `benchmarks/`

  * `xstest.py`
  * `harmbench.py`
  * `jbb.py`
  * `iheval.py`
  * `injecagent.py`
  * `ragtruth.py`

* `prompts/`

  * `registry.yaml`
  * `render.py`
  * `validate.py`

* `models/`

  * `client.py`
  * `cache.py`

* `scoring/`

  * `native.py`
  * `custom.py`
  * `audit.py`

* `experiments/`

  * `run_phase1.py`
  * `run_phase2.py`
  * `run_phase3.py`

* `analysis/`

  * `metrics.py`
  * `stats.py`
  * `plots.py`

* `tests/`

  * unit, integration, smoke, regression

* `artifacts/`

  * raw logs
  * parsed outputs
  * analysis tables
  * figures
  * audit sheets

---

## 14. Acceptance criteria

## Phase 1 is complete when

* all benchmark loaders pass tests
* prompt families are frozen
* dev and holdout splits are separated in code
* full XSTest + one harmful benchmark run is complete
* one figure shows safety-utility tradeoff across prompt families

## Phase 2 is complete when

* IHEval ingestion and scoring pass tests
* hierarchy-first prompts show confirmatory evaluation on held-out conflict items
* analysis quantifies whether explicit hierarchy outperforms vague safety wording

## Phase 3 is complete when

* at least one transfer benchmark is integrated
* transfer matrix is produced
* utility costs are quantified alongside safety gains
* manual audit validates automated scoring on a pre-specified sample

---

## 15. Minimum viable paper/blog figures

1. **Safety boundary plot**
   Harmful compliance vs false refusal by prompt family

2. **Hierarchy conflict plot**
   Accuracy / unsafe continuation under vague vs explicit hierarchy prompts

3. **Transfer matrix**
   Prompt family × domain effect sizes

4. **Cost-benefit plot**
   Safety gain vs utility cost for each controller

These four figures are enough for a solid first writeup.

---

## 16. Main risks and mitigations

### Risk

Prompt effects are tiny or unstable.

**Mitigation**
Use prompt families with paraphrases and within-item repeated measures.

### Risk

Results look like tone/persona changes rather than control changes.

**Mitigation**
Focus on behavioral outcomes under conflict, refusal, and override, not style.

### Risk

Automated scoring is noisy.

**Mitigation**
Use benchmark-native scorers when possible and validate with manual audit.

### Risk

Transfer is weak.

**Mitigation**
That is still a publishable result if framed honestly: many prompt controls are local patches.

### Risk

Project balloons into full agent safety.

**Mitigation**
Start with XSTest + HarmBench/JBB + IHEval before touching AgentDojo.

---

## 17. Recommended execution order

Week 1:

* implement core schema
* load XSTest
* freeze prompt registry
* build end-to-end smoke test

Week 2:

* integrate HarmBench or JailbreakBench
* run Phase 1 dev slice
* validate scorers
* freeze confirmatory Phase 1 prompts

Week 3:

* integrate IHEval
* run Phase 2 dev slice
* freeze hierarchy prompts
* run confirmatory Phase 2

Week 4+:

* integrate InjecAgent
* run transfer study
* optional RAGTruth extension

---

## 18. Final decision rule

This project is successful if it can answer, with clean evidence:

1. **Do prompts induce distinct safety-relevant policies?**
2. **Does explicit hierarchy reduce failures under conflict?**
3. **Which prompt-based safety controls actually transfer?**

If those three are answered cleanly, the project is strong.

[1]: https://arxiv.org/abs/2502.08745?utm_source=chatgpt.com "IHEval: Evaluating Language Models on Following the Instruction Hierarchy"
[2]: https://arxiv.org/abs/2308.01263?utm_source=chatgpt.com "XSTest: A Test Suite for Identifying Exaggerated Safety ..."
[3]: https://arxiv.org/abs/2402.04249?utm_source=chatgpt.com "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal"
[4]: https://arxiv.org/abs/2403.02691?utm_source=chatgpt.com "InjecAgent: Benchmarking Indirect Prompt Injections in ..."
[5]: https://arxiv.org/abs/2406.13352?utm_source=chatgpt.com "AgentDojo: A Dynamic Environment to Evaluate Prompt ..."
[6]: https://arxiv.org/abs/2401.00396?utm_source=chatgpt.com "RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models"
