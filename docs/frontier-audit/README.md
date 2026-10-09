# Human-validation preparation

The current reduced-workload plan is **10 practice responses, 30 representative responses (10/provider), and no targeted packet**. The total time cap is three hours per annotator. Stop at the cap and use **Close at time limit** to preserve a sealed partial pass with missing responses. Both annotators independently label the same full packet. This is a small exploratory check, not strong evidence of judge reliability. These are workload defaults, not a power calculation; time practice and record the final scored size before drawing the sample.

Read [ANNOTATOR_GUIDE.md](ANNOTATOR_GUIDE.md) for unchanged six-policy criteria, [COORDINATOR.md](COORDINATOR.md) for commands, and [ANNOTATOR_INSTRUCTIONS.md](ANNOTATOR_INSTRUCTIONS.md) for step-by-step annotation instructions. [MINIMAL_AUDIT_VALIDATION.md](MINIMAL_AUDIT_VALIDATION.md) records validation of this revision; [VALIDATION.md](VALIDATION.md) preserves the original preparation evidence. [REPORT_TEMPLATE.md](REPORT_TEMPLATE.md) has no study findings filled in.

The new private kit is `.local/frontier-audit/practice-kit-minimal-v1.zip`. Give each annotator a clean copy. The old 30-response kit and any ratings remain untouched. All previously distributed practice responses and exact text matches remain excluded from evaluation. Runtime kits, mappings, identities and notes stay private.

Inside the extracted kit folder, with Python 3.9+:

```sh
python3 -m audit.frontier.dashboard --bundle bundle.json \
  --database .local/ratings.sqlite --coder rater_a --port 8765
```

Use `rater_b` in the second annotator's own copy. Keep Terminal open and visit `http://127.0.0.1:8765` in the browser on that same computer. This address means the annotator's computer, not the coordinator's. Windows users may replace `python3` with `py -3`. Relaunch with the same database and pseudonym to resume. Saved ratings are permanent; adjudication is separate.

The benchmark-stratified lighter sampler replaces the requirement to cover every condition per provider. It records coverage gaps and inclusion weights. The original condition-stratified design remains explicitly selectable. The 10,950-case generation archive, canonical judge/rubric, and legacy audit behavior are unchanged. Human collection, frontier judging and final sample freezing are separate tasks. The scheduler stays paused.
