The goal is to use the outputs you already generated to discover whether the model is exhibiting a richer set of behavioral policies than the current refusal/compliance summaries reveal. That fits the current project state well: you already have the v2 policy-routing framework, the boundary dataset, the policy classifier, and study runners implemented, but live confirmatory analysis is still pending.  
It also fits the scoring reality: XSTest scoring looks strong, but HarmBench scoring is not publication-ready, so mining the raw responses themselves is scientifically safer than over-relying on one noisy binary scorer. 

---

# PRD v3: Mining LLM Responses for Emergent Policy Types

## 1. Overview

### Project title

**Mining LLM Responses for Emergent Policy Types: A Discovery Pipeline for Policy Routing at the Safety Boundary**

### Core objective

Build a scientifically defensible, test-driven discovery pipeline to mine existing LLM outputs for **emergent behavioral policy types**, identify **routing-sensitive items**, and determine whether the current theory of prompt-based policy routing is supported by the data already collected.

### Why this step exists

The current project has shown that prompt families move behavior at the coarse safety boundary, and the v2 codebase now treats **policy-routing accuracy** as the main scientific target. But before locking in a final policy taxonomy or confirmatory benchmark, we need to answer a more basic question:

> Do the model’s existing responses naturally organize into meaningful policy types, or are the apparent effects mostly a one-dimensional refusal/compliance shift?

This PRD is meant to answer that question.

---

## 2. Problem statement

The current analyses are too coarse. Existing plots mainly summarize:

* harmful compliance,
* false refusal,
* unsafe continuation,
* hierarchy-following.

Those are useful, but they do not yet tell us whether the model is selecting among multiple distinct policies such as:

* answer,
* refuse,
* clarify,
* minimal-safe-help,
* hierarchy-preserve.

At the same time, the project’s current scoring layer is asymmetric in reliability: XSTest scoring is strong, while HarmBench scoring remains too weak to support a paper as a primary quantitative pillar. 

So the next step is not to add more benchmarks.
It is to **mine the outputs you already have** and ask:

1. what policy-like response types actually emerge,
2. which items show prompt-dependent routing differences,
3. whether those emergent types align with the current v2 policy-routing theory.

---

## 3. Core thesis for this stage

If prompts act as control signals for policy selection, then existing model outputs should not just vary in “more refusal” versus “less refusal.” They should exhibit **clusters of behaviorally distinct policy types**, and the most informative items should be those where the same underlying item elicits different policy types under different controller prompts.

---

## 4. Product goals

### Goal 1

Discover whether model outputs naturally cluster into coherent behavioral policy types.

### Goal 2

Test whether the current proposed taxonomy

* answer
* refuse
* clarify
* minimal-safe-help
* hierarchy-preserve

is empirically supported, needs refinement, or is missing important categories.

### Goal 3

Identify **routing-sensitive items**: items whose outputs differ across prompt families in ways consistent with policy switching.

### Goal 4

Produce a small, high-value audited subset for the next confirmatory stage.

---

## 5. Non-goals

This PRD will not:

* make final confirmatory claims for the paper
* replace human ratings entirely
* rely on unsupervised clustering as ground truth
* treat cluster count as a scientific result in itself
* add new external benchmarks
* perform mechanistic interpretability
* use mined clusters directly as the final published ontology without audit

---

## 6. Inputs

## Required inputs

* `phase1_results.csv`
* `phase2_results.csv`
* current prompt metadata
* current output text
* existing scorer outputs and metadata
* existing v2 boundary dataset annotations where available

## Optional inputs

* item text
* benchmark source
* prompt family
* clarity level
* context condition
* existing v2 policy classifier outputs

---

## 7. Main research questions

### RQ1

Do the existing outputs form coherent policy-like clusters?

### RQ2

Do those clusters align with the current v2 policy taxonomy?

### RQ3

Are some prompt families enriched for specific policy clusters?

### RQ4

Which items show the strongest evidence of prompt-dependent policy routing?

### RQ5

Does the mined structure suggest revisions to the current theory?

---

## 8. Main hypotheses

### H1

The output space contains more than one meaningful safe-response mode; it is not purely one-dimensional refusal/compliance variation.

### H2

At least some clusters will map cleanly onto the current v2 policy candidates:

* refusal,
* clarification,
* minimal-safe-help,
* hierarchy-preserve.

### H3

Routing-sensitive items will be concentrated in ambiguous, quoted, and conflict-heavy contexts rather than in obviously unsafe prompts.

### H4

Prompt families will differ not only in mean refusal rate, but in **cluster enrichment**, indicating that they differentially route the model into distinct policy types.

---

## 9. Scientific defensibility principles

### Principle 1: discovery, not confirmation

This pipeline is exploratory. All outputs from it must be labeled as:

* exploratory,
* development-stage,
* hypothesis-generating.

### Principle 2: no unsupervised method is treated as ground truth

Clusters are tools for discovering structure, not final labels.

### Principle 3: raw text matters more than weak scorers

Because HarmBench scoring is currently unreliable relative to XSTest, mining should primarily use raw model outputs plus lightweight features, not just existing binary labels. 

### Principle 4: audit before interpretation

No cluster should be named or theorized about until representative examples are manually reviewed.

### Principle 5: routing-sensitive items are more important than global averages

The key scientific target is the set of items where prompt changes induce policy changes on the same underlying item.

---

## 10. Analysis plan

## Stage A: Response representation

Create a response representation table where each row is one model output.

### Text features

* TF-IDF of output text
* output length
* sentence count
* question mark count
* refusal phrase count
* hedging phrase count
* apology phrase count
* redirection phrase count

### Metadata features

Use for downstream analysis, but **do not include them in the clustering representation unless explicitly justified**:

* prompt family
* clarity level
* benchmark
* item_id
* existing scorer outputs
* context condition if present

### Existing numeric behavior features

Append for hybrid analyses:

* refusal score
* compliance score
* clarification score
* abstention score
* hierarchy_following score
* unsafe_continuation score
* stop_compliance score

These can be used for interpretation, not necessarily for the main clustering representation.

---

## 11. Discovery analyses

## Analysis 1: Text-only clustering

Cluster raw outputs using text representations only.

Purpose:

* check whether policy-like response modes emerge from language itself

Deliverables:

* cluster assignments
* cluster sizes
* top lexical features per cluster
* representative exemplars

## Analysis 2: Hybrid clustering

Cluster outputs using:

* text representation
* existing numeric behavior features

Purpose:

* see whether text and coarse scores together reveal more interpretable policy types

Deliverables:

* hybrid cluster assignments
* comparison to text-only clustering
* cluster stability metrics

## Analysis 3: Cluster enrichment by prompt family

Test whether prompt families differ in which clusters they produce.

Purpose:

* directly test the routing story

Deliverables:

* cluster × prompt-family contingency table
* normalized enrichment heatmap
* effect-size summaries

## Analysis 4: Routing-sensitive item detection

For each `item_id`, compare the outputs across prompt families and clarity levels.

Flag items where:

* outputs fall into different clusters,
* or differ strongly on the existing policy classifier,
* or exhibit high disagreement between text cluster and current policy label.

These are the **routing-sensitive items**.

Deliverables:

* ranked list of routing-sensitive items
* disagreement score per item
* exemplar output sets per item

## Analysis 5: Cluster-to-taxonomy mapping

After manual review of exemplars, assign provisional human-readable labels to clusters.

Candidate labels:

* direct refusal
* hedged refusal
* clarification
* safe redirection
* minimal-safe-help
* hierarchy-preserving refusal
* answer with safety preamble
* direct compliance
* mixed / ambiguous

Deliverables:

* provisional mapping from clusters to policy taxonomy
* note where the v2 taxonomy is supported, incomplete, or mis-specified

---

## 12. Required outputs

### Output 1

A **cluster report** with:

* cluster counts,
* top features,
* exemplar outputs,
* suggested policy interpretation.

### Output 2

A **routing-sensitive item report** with:

* item_id,
* benchmark,
* prompt-family outputs,
* cluster assignments,
* disagreement score.

### Output 3

A **taxonomy revision memo** stating:

* which v2 labels are supported,
* which need refinement,
* whether new labels are needed.

### Output 4

A **manual audit set** for the next phase:

* cluster exemplars,
* routing-sensitive items,
* ambiguous cases.

---

## 13. TDD / RED→GREEN plan

This discovery stage must still follow strict TDD.

## Milestone 1: Response-table builder

### RED

Tests fail because no unified mining table exists.

### GREEN

Implement a builder that loads Phase 1 and Phase 2 outputs into one unified row-level table.

### Acceptance

* one row per output
* required columns present
* no silent row drops
* benchmark and prompt metadata preserved

## Milestone 2: Feature extraction

### RED

Tests fail because response features are missing.

### GREEN

Implement deterministic feature extraction for:

* lexical features
* length features
* punctuation features
* scorer features

### Acceptance

* same input gives same feature vector
* missing text handled explicitly
* feature names stable across runs

## Milestone 3: Cluster pipeline

### RED

Tests fail because no clustering API exists.

### GREEN

Implement clustering module with:

* text-only mode
* hybrid mode
* configurable cluster counts
* deterministic seeds

### Acceptance

* cluster labels reproducible given fixed seed
* pipeline fails loudly on malformed inputs
* small toy data gives expected separable clusters

## Milestone 4: Exemplar extraction

### RED

Tests fail because cluster exemplars cannot be produced.

### GREEN

Implement exemplar selection for:

* cluster centroids or nearest points
* random audit samples per cluster

### Acceptance

* fixed seed reproducibility
* no duplicate exemplars when avoidable
* output table includes text and metadata

## Milestone 5: Routing-sensitive item detector

### RED

Tests fail because item-level disagreement is not computed.

### GREEN

Implement item-level routing disagreement metric.

### Acceptance

* within-item outputs compared correctly
* disagreement ranking stable
* tied cases handled explicitly

## Milestone 6: Reporting

### RED

Tests fail because reports and figures do not exist.

### GREEN

Implement:

* cluster report
* enrichment heatmap
* routing-sensitive item summary
* taxonomy memo scaffold

### Acceptance

* reports generated from one command
* outputs saved with versioned filenames
* figures reproducible

---

## 14. Required test suite

### `test_response_table.py`

* loads Phase 1 and Phase 2 outputs
* preserves row counts
* preserves item IDs and prompt metadata

### `test_feature_extraction.py`

* lexical features computed correctly on toy strings
* missing/empty outputs handled safely
* feature vectors deterministic

### `test_clustering_pipeline.py`

* deterministic seeds work
* toy corpora split into expected groups
* invalid cluster parameters fail loudly

### `test_exemplar_selection.py`

* exemplars come from correct clusters
* no duplicate exemplars when enough data exist

### `test_routing_sensitivity.py`

* known toy cases rank high when outputs differ
* identical outputs rank low
* item grouping is correct

### `test_cluster_enrichment.py`

* enrichment tables aggregate correctly
* normalized frequencies sum appropriately

### `test_mining_reports.py`

* report artifacts are written
* figure generation works
* tables contain required columns

### `test_mining_end_to_end.py`

* full pipeline runs on a tiny fixture set
* outputs all expected artifacts

---

## 15. Methodological choices

## Clustering strategy

Start simple and interpretable.

Primary pipeline:

* TF-IDF representation
* dimensionality reduction only for visualization, not necessarily for the main cluster assignment
* one or two simple clustering algorithms, not a zoo

Reason:

* this is a scientific discovery step, not a leaderboard on unsupervised NLP

## Cluster count

Do not fix one cluster count in advance as “truth.”

Instead:

* inspect a small range
* compare coherence and interpretability
* use manual review to choose a pragmatic working solution

## Prompt metadata

Do not use prompt family or clarity level as clustering inputs in the main unsupervised analysis.

Reason:

* that would contaminate the discovery step

Use prompt metadata only afterward to test enrichment.

## Language handling

Because scorer failures include non-English responses on HarmBench, detect language/script where possible and flag outputs that may need separate handling or exclusion. 

## Human audit

A cluster is not real for the paper until exemplars are manually inspected.

---

## 16. Figures for this stage

### Figure 1

Cluster size distribution

### Figure 2

2D projection of outputs colored by cluster

### Figure 3

Prompt-family enrichment heatmap by cluster

### Figure 4

Top routing-sensitive items with across-prompt outputs

### Figure 5

Cluster-to-policy mapping summary

These are discovery figures, not necessarily final paper figures.

---

## 17. Risks and mitigations

### Risk

Clusters reflect style or verbosity rather than policy.

**Mitigation**
Use manual exemplar audit and compare text-only vs hybrid clustering.

### Risk

Outputs are too dominated by refusal language to recover richer structure.

**Mitigation**
Analyze routing-sensitive items separately from the full corpus.

### Risk

HarmBench outputs distort the discovery due to scorer weakness or adversarial style.

**Mitigation**
Lean more heavily on raw text, flag multilingual/adversarial cases, and treat HarmBench-derived conclusions cautiously. 

### Risk

The discovered taxonomy does not match the current v2 labels.

**Mitigation**
That is a valuable finding. Revise the taxonomy rather than forcing the data to fit.

### Risk

This becomes another endless infrastructure project.

**Mitigation**
This PRD forbids new benchmark integrations and focuses only on mining existing outputs.

---

## 18. Acceptance criteria

This stage is complete when:

1. A unified mining table exists for existing outputs.
2. At least one interpretable clustering solution has been generated and manually inspected.
3. A set of routing-sensitive items has been identified.
4. A provisional cluster-to-policy taxonomy has been written.
5. A small audited subset has been selected for the next confirmatory phase.

---

## 19. Final decision rule

This discovery stage is successful if it can answer:

1. Do the outputs naturally contain richer policy types than simple refusal/compliance?
2. Which policy types actually appear?
3. Which items are most diagnostic of prompt-based policy routing?
4. Does the current v2 taxonomy hold up, or should it be revised?

If the answer to those questions is clear, the next confirmatory stage becomes much sharper.

---

## 20. Recommended execution order

### Week 1

* build unified response table
* write feature extraction tests
* implement deterministic feature pipeline

### Week 2

* implement text-only clustering
* generate exemplar reports
* inspect candidate cluster counts

### Week 3

* implement hybrid clustering
* generate enrichment analyses
* build routing-sensitive item detector

### Week 4

* manually inspect exemplars
* write taxonomy memo
* select audited subset for confirmatory labeling

---

## 21. What this stage should produce for the paper

This stage should not directly produce the final main result.
It should produce the **scientific bridge** between:

* the current coarse refusal/compliance findings,
  and
* the stronger policy-routing narrative.

If it works, it will let you say:

> Existing responses do not merely vary in overall safety level. They exhibit multiple emergent policy types, and controller prompts appear to enrich different policy modes on the same underlying items.

