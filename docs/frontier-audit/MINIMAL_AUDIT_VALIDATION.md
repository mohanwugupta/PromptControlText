# Minimal exploratory audit — validation evidence

Protocol `frontier-minimal-v1`: 10 practice responses plus a planned 30 representative responses (10/provider), two independent annotators, a three-hour total time cap per annotator, and zero targeted cases. The response-policy rubric remains `frontier-human-v1`; no canonical judge, generation design or historical audit was changed. This document records software validation, not human-study findings.

## Automated validation

```sh
.venv/bin/python -m pytest \
  tests/test_frontier_human.py tests/test_audit_set.py \
  tests/test_llm_policy_prompts.py tests/test_llm_policy_schema.py \
  tests/test_llm_policy_adjudicate.py -q
```

Result: **90 tests passed** (31 audit tests including parametrized cases, plus 59 existing regression tests). New checks cover benchmark-stratified sampling below the condition-count minimum, deterministic row-order invariance, inclusion weights summing to the eligible frame, recorded condition gaps, CLI defaults (10 practice / 10 per provider), immutable output creation, subset practice and all prior-practice exclusions, targeted exclusion provenance, rejection of foreign/mutated practice cases, complete and partial HTTP sealing, no editing after closure, zero-rating time-limit closure, reopening closed databases, paired-only adjudication, missingness reporting, and rejection of unsealed or improperly marked incomplete exports.

The original condition-stratified Python API defaults remain available. The CLI defaults change to benchmark strata and 10/provider; commands and manifests record the design explicitly. The targeted CLI requires an explicit size and is outside this minimal protocol.

## Real archive and private kit verification

The final frame independently verifies all **10,950** frozen cases and their response mappings: **10,432 text responses** and **518 nontext blocks**. The new 10-response practice packet is a deterministic subset of the original 30, using the same historical frame and seed `frontier-practice-v1`. All response strings and request mappings match. All original practice responses and exact text matches remain excluded; the eligible scored text frame has **10,364** requests. Allocation feasibility was checked for 10/provider across benchmark strata without drawing or freezing the scored sample or inspecting comparative labels.

Practice bundle checksum:

`a9ed22eb90225f125a80a8fa6cb67ad746e3f4a8e4ae90b841e7ba5a6d605811`

The private `practice-kit-minimal-v1.zip` contains only the blinded packet, minimal dashboard code, guide and frozen rubric files, with no ratings, coordinator manifest, credentials or identity information. Archive bytes were checked against the source kit. Existing practice packets/databases remain untouched.

## Browser verification with synthetic fixtures only

A separate two-response synthetic dashboard was tested in the in-app browser. A synthetic rating was saved, the next response appeared, and the new **Close at time limit** button required a second explicit confirmation. It then displayed **1 / 2 saved · Sealed at time limit**, disabled editing and preserved the unanswered response as missing. A separate synthetic database tested both responses followed by **Seal completed pass → Confirm seal**, which displayed **2 / 2 saved · Sealed**. Screenshots are retained privately under `.local/frontier-audit/minimal-ui/`. This completes the earlier outstanding normal-seal interaction check. No genuine human labels were created.

## Limits

The three-hour limit is a procedural limit tracked with an external cumulative session timer; the dashboard response timer is not a reliable total-session clock and does not automatically stop a person at three hours. The explicit closure control safely preserves partial work. Instructions include setup, practice, calibration, scored annotation and export within the time budget; adjudication is a separate role.

At most 30 scored pairs provide limited exploratory evidence. Benchmark coverage is recorded, while coverage of all 73 conditions/provider is intentionally relinquished. Per-provider and rare-policy estimates can be extremely imprecise or undefined. Time-limit missingness can be related to response difficulty; weights do not remove that bias. The analysis retains assigned denominators, missingness, uncertainty bounds and original independent exports, and adjudicates only paired ratings. Calibration/timing and a recorded freeze decision precede the actual scored draw. Canonical frontier judging, actual human collection, adjudication and study findings remain unfinished.
