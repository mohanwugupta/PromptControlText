"""36 cheap judgments of 12 existing responses; $1 cap, no new generation.

Reuses the generation pilot's durable reservation and transport controls.
Only --execute sends requests. Raw judge inputs/evidence remain gitignored.
"""
import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import run_api_pilot as api
from scoring.llm_policy_prompts import build_judge_user_message, load_default_prompts
from scoring.llm_policy_schema import load_schema, validate_judge_output

OUT = ROOT / 'artifacts/judge_micro_pilot'
MODEL = 'gpt-5.4-nano'
SEED = 20260922
BASE_PAYLOAD = api.payload


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload(task):
    body = BASE_PAYLOAD(task)
    body['text'] = {'format': {'type': 'json_schema', 'name': 'policy_judgment',
                             'strict': True, 'schema': load_schema()}}
    return body


def reservation(task):
    # Includes schema, framing, input text, and a large additional margin.
    upper_input = len(json.dumps(payload(task), ensure_ascii=False).encode()) + 2048
    return (upper_input * 0.20 * 1.25 + task['max_output_tokens'] * 1.25) / 1e6


def validate_resume_state():
    if api.read_jsonl(OUT / 'usage.jsonl') and not (OUT / 'ledger.jsonl').exists():
        raise RuntimeError('Published judge archive has no private ledger; refusing duplicate paid requests.')


def configure():
    api.OUT = OUT
    api.MODELS = [MODEL]
    api.PRICES = {MODEL: (0.20, 1.25)}
    api.CAP = 1.0
    api.payload = payload
    api.reservation = reservation
    api.validate_resume_state = validate_resume_state


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'manifest.private.json'
    if path.exists():
        manifest = json.loads(path.read_text())
        for name, expected in manifest['source_sha256'].items():
            assert digest_file(ROOT / name) == expected, f'Changed source: {name}'
        return manifest
    if (OUT / 'manifest.json').exists():
        raise RuntimeError('Published archive has no private inputs. Refusing to recreate paid tasks.')
    source = ROOT / 'artifacts/frontier_pilot/responses.jsonl'
    rows = [r for r in api.read_jsonl(source) if r.get('status') == 'completed' and r.get('model_output')]
    groups = defaultdict(list)
    for r in rows:
        groups[(r['benchmark'], r['model'])].append(r)
    assert len(groups) == 6
    rng = random.Random(SEED)
    selected = []
    for group in sorted(groups):
        # Distinct items per stratum; randomized items then randomized available condition.
        by_item = defaultdict(list)
        for r in groups[group]:
            by_item[r['item_id']].append(r)
        for item in rng.sample(sorted(by_item), 2):
            selected.append(rng.choice(sorted(by_item[item], key=lambda r:r['task_id'])))
    prompts = load_default_prompts()
    tasks = []
    for r in selected:
        for variant in ['A', 'B', 'C']:
            t = dict(model=MODEL, source_task_id=r['task_id'], source_output_sha256=hashlib.sha256(r['model_output'].encode()).hexdigest(),
                     benchmark=r['benchmark'], item_id=r['item_id'], source_model=r['model'],
                     prompt_family='output-only-policy-judge', clarity_level='not-applicable', prompt_variant=variant,
                     system_prompt=prompts[variant], input_text=build_judge_user_message(r['model_output']),
                     reasoning_effort='none', max_output_tokens=512)
            t['task_id'] = api.digest(t)
            tasks.append(t)
    sources = ['artifacts/frontier_pilot/responses.jsonl', 'scoring/llm_policy_schema_v1.json'] + [f'scoring/llm_policy_judge_prompt_{v}_v1.txt' for v in 'ABC']
    manifest = dict(seed=SEED, design='12 existing responses: two per benchmark/source-model stratum; three prompt variants each',
                    model=MODEL, budget_usd=1.0, input_usd_per_million=.20, output_usd_per_million=1.25,
                    reasoning_effort='none', max_output_tokens=512, schema=load_schema(),
                    source_sha256={n:digest_file(ROOT/n) for n in sources}, tasks=tasks)
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def report(manifest):
    rows = api.read_jsonl(OUT/'responses.jsonl')
    latest = {r['task_id']:r for r in rows}
    tasks = {t['task_id']:t for t in manifest['tasks']}
    public = []
    private = []
    votes = defaultdict(dict)
    for tid,r in latest.items():
        t=tasks[tid]
        label, error = validate_judge_output(r.get('model_output','')) if r['status']=='completed' else (None, 'API response not completed')
        row={k:r[k] for k in ['task_id','model','response_model','status','usage','estimated_cost_usd','elapsed_seconds','error_code'] if k in r}
        row.update(source_task_id=t['source_task_id'],judge_prompt=t['prompt_variant'],valid_schema=label is not None)
        if label:
            row.update(primary_label=label['primary_label'],secondary_label=label['secondary_label'],confidence=label['confidence'])
            votes[t['source_task_id']][t['prompt_variant']]=label['primary_label']
        public.append(row)
        private.append(dict(task_id=tid,judgment=label,validation_error=error))
    (OUT/'judgments.private.json').write_text(json.dumps(private,indent=2)+'\n')
    (OUT/'usage.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in public))
    public_manifest={k:v for k,v in manifest.items() if k!='tasks'}
    public_manifest['tasks']=[{k:v for k,v in t.items() if k not in {'input_text','system_prompt'}} for t in manifest['tasks']]
    (OUT/'manifest.json').write_text(json.dumps(public_manifest,indent=2)+'\n')
    complete=[v for v in votes.values() if len(v)==3]
    total=sum(r.get('estimated_cost_usd',0) for r in rows)
    summary=dict(planned=36,attempted=len(latest),valid_judgments=sum(r['valid_schema'] for r in public),
                 fully_judged_responses=len(complete),unanimous_responses=sum(len(set(v.values()))==1 for v in complete),
                 estimated_cost_usd=total,accounted_usd=api.Ledger(OUT).total(),budget_usd=1.0,
                 statuses=dict(Counter(r['status'] for r in public)),label_counts=dict(Counter(r['primary_label'] for r in public if r['valid_schema'])))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Cheapest-judge micro-pilot','',manifest['design']+'.','',
           'GPT-5.4 nano; no reasoning; 512 output-token cap; standard API; original A/B/C prompts and strict JSON schema.',
           'Judges see only assistant response text, without benchmark, source model, original prompt, or expected labels.','',
           f"Valid judgments: {summary['valid_judgments']}/36. Fully judged responses: {len(complete)}/12.",
           f"Unanimous A/B/C primary labels: {summary['unanimous_responses']}/{len(complete)}.",
           f"Estimated usage charge: ${total:.6f}; conservative ledger: ${summary['accounted_usd']:.6f} / $1.",'',
           '| Source response ID | A | B | C |','|---|---|---|---|']
    for sid in dict.fromkeys(t['source_task_id'] for t in manifest['tasks']):
        v=votes[sid];lines.append('| '+sid[:12]+' | '+' | '.join(v.get(k,'missing/invalid') for k in 'ABC')+' |')
    lines += ['', '## Interpretation', '',
              'This checks cost, schema compatibility, and within-model prompt agreement. Three prompts on one model are correlated votes, not independent judges or human ground truth.',
              'No human accuracy estimate, policy-performance claim, or production approval follows from this tiny sample. Only completed pilot responses were eligible; selection is not representative of all benchmark outputs.',
              'The original A/B/C prompts differ in their rules for mixed refusal-and-answer responses, so disagreements may reflect rubric differences as well as model instability.',
              'Raw response text and judge evidence remain private. Public metadata contains labels, costs, and provenance only. No adjudication or larger-model comparison has been run.', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    configure()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=prepare()
        if args.execute:
            api.execute(manifest,args.limit)
        print(json.dumps(report(manifest),indent=2))
