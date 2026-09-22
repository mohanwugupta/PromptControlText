"""Offline integrity, response-quality, and generation-cost report for the pilot."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/frontier_pilot'


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def main():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    tasks = {t['task_id']: t for t in manifest['tasks']}
    private = (OUT / 'responses.jsonl').exists()
    attempts = read_rows(OUT / ('responses.jsonl' if private else 'usage.jsonl'))
    ledger = {e['task_id']: e for e in read_rows(OUT / 'ledger.jsonl')}
    latest = {r['task_id']: r for r in attempts}
    blocked = {(r['benchmark'], r['item_id']) for r in attempts if r.get('error_code') == 'cyber_policy'}
    assert all(r['task_id'] in tasks for r in attempts)
    for name, expected in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    paid = [r for r in attempts if 'usage' in r]
    assert len(paid) == len({r['task_id'] for r in paid}), 'Duplicate billed completion'
    estimated = sum(r['estimated_cost_usd'] for r in paid)
    accounted = sum(e['budget_charge_usd'] for e in ledger.values()) if ledger else json.loads((OUT / 'budget_accounting.json').read_text())['accounted_usd']
    assert accounted <= manifest['budget_usd'] + 1e-8
    models = sorted({t['model'] for t in tasks.values()})
    clean = []
    for tid, task in tasks.items():
        r = latest.get(tid, {})
        status = r.get('status') or ('skipped_provider_block' if (task['benchmark'], task['item_id']) in blocked else 'not_attempted')
        clean.append({**task, **r, 'status': status})
    fields = ['task_id', 'model', 'response_model', 'benchmark', 'item_id', 'gold_label',
              'domain', 'prompt_family', 'clarity_level', 'prompt_variant',
              'reasoning_effort', 'max_output_tokens', 'input_text', 'system_prompt',
              'status', 'model_output', 'estimated_cost_usd', 'elapsed_seconds', 'error_code']
    with (OUT / ('pilot_results.csv' if private else 'pilot_metadata.csv')).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(clean)
    lines = ['# Frontier API pilot report', '',
             'This is a technical feasibility and cost pilot, not a safety-performance result.', '',
             f"Source repository commit: `{manifest['repository_commit']}`. All recorded source hashes verified.",
             '', f"Planned: {len(tasks)} requests on 24 items (8 per benchmark), two models, four conditions per item.",
             'The four conditions are an omitted-system baseline and three assigned prompts. The sparse design covers the 72 registry prompts once per model before exclusions.',
             '', f'Estimated generation charges from returned token usage: **${estimated:.4f}**.',
             f'Conservative budget accounting including unresolved/rejected-request reservations: **${accounted:.4f} / $10**.',
             'These are local estimates, not a provider invoice. No paid judge calls are included.', '',
             '## Response quality', '',
             '| Model | Completed | Incomplete | Empty returned finals | API blocks | Other failures | Skipped variants | Not attempted | Estimated usage cost |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for model in models:
        rr = [r for r in clean if r['model'] == model]
        got = [r for r in rr if 'usage' in r]
        statuses = collections.Counter(r['status'] for r in rr)
        lines.append(f"| {model} | {statuses['completed']} | {statuses['incomplete']} | {sum(not (r.get('model_output') if private else r.get('output_chars')) for r in got)} | {sum(r.get('error_code') == 'cyber_policy' for r in rr)} | {sum(r['status'] in {'http_error', 'transport_or_parse_error'} and r.get('error_code') != 'cyber_policy' for r in rr)} | {statuses['skipped_provider_block']} | {statuses['not_attempted']} | ${sum(r.get('estimated_cost_usd', 0) for r in got):.4f} |")
    lines += ['', 'Both models used low reasoning effort and a 2,048-token combined reasoning/output cap. No temperature or seed was supplied. This differs from the original non-reasoning 512-token protocol.', '',
              '## Costs by benchmark', '',
              '| Model | Benchmark | Returned responses | Input tokens | Output tokens | Reasoning tokens | Mean cost/response | Mean latency (s) |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for model in models:
        for benchmark in ['HarmBench', 'XSTest', 'IHEval']:
            rr = [r for r in paid if r['model'] == model and r['benchmark'] == benchmark]
            if not rr:
                continue
            lines.append(f"| {model} | {benchmark} | {len(rr)} | {sum(r['usage']['input_tokens'] for r in rr)} | {sum(r['usage']['output_tokens'] for r in rr)} | {sum(r['usage'].get('output_tokens_details', {}).get('reasoning_tokens', 0) for r in rr)} | ${statistics.mean(r['estimated_cost_usd'] for r in rr):.5f} | {statistics.mean(r['elapsed_seconds'] for r in rr):.1f} |")
    lines += ['', '## Preliminary generation-only projections', '',
              'Extrapolations use each model’s observed pooled mean. Item selection, missing API-blocked cases, truncation, output lengths, and sparse prompt assignment limit these estimates. They are not guaranteed quotes.', '',
              '| Model | 100 items × 25 conditions | 200 items × 25 conditions | 100 items × 73 conditions | 600 items × 73 conditions |',
              '|---|---:|---:|---:|---:|']
    for model in models:
        rr = [r for r in paid if r['model'] == model]
        if rr:
            mean = statistics.mean(r['estimated_cost_usd'] for r in rr)
            lines.append(f'| {model} | ${mean*2500:.2f} | ${mean*5000:.2f} | ${mean*7300:.2f} | ${mean*43800:.2f} |')
    lines += ['', 'The 25-condition option retains eight families × three clarity levels plus baseline, with one preselected wording variant per cell. It sacrifices paraphrase replication and must be reported as a reduced extension. A full budget must also cover judging, failed requests, reruns, and contingency.', '',
              '## Interpretation limits and next steps', '',
              '- Provider policy errors are API-level blocks, not generated refusals. The affected item’s remaining variants are skipped across both models; no prompt rewriting is used to evade the block.',
              '- Prior credit-exhaustion attempts are retained in the attempt history but do not count as generated responses.',
              '- Inspect incomplete and mixed responses before selecting a production output cap. Increasing the cap can increase cost.',
              '- No LLM or human labels have been assigned; these data cannot yet support policy-switching or SDT estimates.',
              '- A limited assistant spot-check of four benign XSTest baseline outputs (two items, both models) found relevant final answers. This is not a blinded human audit or an accuracy estimate.',
              '- HarmBench contextual inputs omit ContextString, matching the existing loader. IHEval inputs are flattened and may omit the native instruction hierarchy. Resolve these dataset issues before claiming benchmark validity.',
              '- Sparse assignments confound item and prompt condition. Use a fully crossed selected subset for any confirmatory extension.',
              '- Keep API policy blocks and missing data in reports; do not silently convert them to refusal or discard them from denominators.',
              '- Reuse original-model outputs only where item text, prompt conditions, and scoring definitions match.', '',
              '## Budget recommendation', '',
              'For the $250 total budget, first consider 100 stratified items × all 73 conditions × both models. This preserves wording replication and leaves room for judging, reruns, and contingency. The pilot suggests roughly $92 for generation alone at standard prices, but nonrandom API exclusions and short flattened IHEval inputs may underestimate production costs. Fix the dataset construction and scoring protocol before a larger run. No production run has been started.', '',
              '## Files', '',
              '- `manifest.json`: exact planned requests, sampling, provenance, prices, and settings.',
              '- `usage.jsonl`: public attempt metadata and token usage, excluding output text and provider identifiers.',
              '- `responses.jsonl`: private append-only request history and final texts (gitignored).',
              '- `pilot_results.csv`: one row per planned task, including missing and blocked rows.',
              '- `ledger.jsonl`: private persistent reservations and settlements (gitignored); `budget_accounting.json` publishes the aggregate.', '']
    (OUT / 'PILOT_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'planned':len(tasks), 'returned':len(paid), 'completed':sum(r['status']=='completed' for r in clean),
                      'estimated_cost_usd':estimated, 'accounted_usd':accounted,
                      'report':str(OUT/'PILOT_REPORT.md')}))


if __name__ == '__main__':
    main()
