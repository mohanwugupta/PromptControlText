# Preparation validation — October 8, 2026

This document records software validation, **not human-study findings**.

## Automated checks

Command (repository virtual environment, Python 3.9):

```sh
.venv/bin/python -m pytest \
  tests/test_frontier_human.py tests/test_audit_set.py \
  tests/test_llm_policy_prompts.py tests/test_llm_policy_schema.py \
  tests/test_llm_policy_adjudicate.py -q
```

Result: **83 tests passed**. This includes 24 new synthetic audit test cases and 59 existing audit/judge regression cases. Pandas and jsonschema were installed in the local virtual environment to run existing regression tests; the new dashboard and analysis have no third-party runtime dependencies.

Checks cover deterministic selection, input-order invariance, request/text practice separation, provider/condition coverage and inclusion weights; unique identities for identical response strings; strict blinded-bundle fields; independent database bindings and resume; rejection of overwrites and direct updates/deletes; exact evidence, allowed labels, confidence and invalid values; incomplete/sealed exports; source-bound separate adjudication including unresolved outcomes; canonical checkpoint/prompt validation and deduplicated-text-to-request mapping; known kappa/confusion examples, absent classes, unequal weights, missingness, empty queues, parent clustering and degenerate intervals; localhost HTTP data isolation, host checks, write-token checks, forbidden routes, duplicate-save rejection, exports and sealing.

A real published-frame build independently reconstructed frozen case IDs and matched all 10,105 published completed cases and response hashes at the input snapshot. It contained 9,695 eligible text responses and 410 separately counted nontext blocks. A 30-response provisional practice packet was generated with seed `frontier-practice-v1`; neither the scored representative sample nor the targeted queue was frozen. Every original response string was preserved. The packet has zero human ratings. Practice bundle checksum:

`0d11c1e081559db368c8d6af6e1caeac7a9d303cd38fa618aaf330fb82fae4b2`

Historical audit code, frozen main config/manifest, and canonical judge prompts/schema were checked byte-for-byte against the base commit and remain unchanged.

## Visible dashboard checks

The in-app browser displayed the synthetic two-response test packet. It showed response text and opaque audit IDs, with no prompt, provider, benchmark, sampling reason or model prediction. A synthetic rating was entered through visible controls and saved; the next response appeared. Reload resumed at the unrated response. Navigating back showed the saved rating and disabled editing controls. Both synthetic rows were saved. The separate uncertainty flag and six label choices were visible. A first-pass unresolved control was found visible during the initial check and fixed; the corrected first-pass screen hides it.

The first native browser confirmation interrupted automation. It was replaced with an inline two-click seal control, avoiding native dialogs. A fresh page with the revised control was visually inspected, but the browser automation session could not complete its click interaction afterward. HTTP tests verified complete-pass sealing and rejection of incomplete-pass sealing. The revised inline seal interaction should receive a coordinator click-through before distributing the packet. No genuine human ratings were used or invented for these checks.

## Limits and remaining decisions

This validates tooling, not taxonomy validity, annotation quality, statistical power or judge accuracy. Actual independent annotators, a distinct adjudicator, time allocation, final sample size and the recorded refusal/fulfillment interpretation must be settled during calibration. Canonical frontier judge outputs were unavailable during preparation; the adapter and targeted path were tested using clearly synthetic fixtures. No paid judging or infrastructure was started. Private coordinator maps, annotator identities, ratings, and future notes are excluded from publication. Cluster intervals are approximate and may be unstable/undefined for sparse classes or few parents.
