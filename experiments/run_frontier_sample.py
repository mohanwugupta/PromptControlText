"""Corrected, paired frontier sample; Astra Batch generation and original-judge export.

No paid calls without `submit`. Immutable inputs, one outstanding batch at a time,
durable worst-case reservations, no automatic POST retry, $75 Astra / $250 study.
"""
import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import urllib.request
import urllib.error
import uuid

import yaml
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/frontier_sample_v1'
DATA = ROOT / 'benchmarks/artifacts/datasets'
MODEL = 'gpt-6-astra'
CAP = 75.0
SEED = 20260923
TERMINAL = {'completed', 'failed', 'expired', 'cancelled'}
POLICY_BLOCKS = {'cyber_policy', 'bio_policy'}
SOURCES = {
 'iheval_aligned.json': ('benchmark/safety/system-prompt-extract/aligned/default/input_data.json','953709da9529cf4aaf609404808faf8ddca7ea3938e0c3af535ea4499959a8d1'),
 'iheval_conflict.json': ('benchmark/safety/system-prompt-extract/conflict/weak_defense/input_data.json','4644f102b815ef636adac80b057c0d775019720919e39de9afe4a650fd1ea534')}


def sha(data):return hashlib.sha256(data).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False).encode()
def rows(path):
    if not path.exists():return []
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]

def save(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)

def readcsv(name):
    with (DATA/name).open() as f:return list(csv.DictReader(f))


def balanced_sample(pool, n, category, rng):
    groups=defaultdict(list)
    for r in pool:groups[r[category]].append(r)
    keys=sorted(groups);rng.shuffle(keys)
    result=[]
    for k in keys:rng.shuffle(groups[k])
    while len(result)<n:
        for k in keys:
            if groups[k] and len(result)<n:result.append(groups[k].pop())
        if not any(groups.values()) and len(result)<n:raise ValueError('Insufficient sampling pool')
    return result


def download_sources():
    directory=DATA/'corrected_v1';directory.mkdir(parents=True,exist_ok=True)
    for name,(remote_path,expected) in SOURCES.items():
        path=directory/name
        if path.exists():
            assert sha(path.read_bytes())==expected
            continue
        url='https://raw.githubusercontent.com/ytyz1307zzh/IHEval/726a62924c3050045954df94347d53fe2bd1090d/'+remote_path
        with urllib.request.urlopen(url,timeout=60) as response:data=response.read()
        assert sha(data)==expected, 'Upstream source hash mismatch'
        path.write_bytes(data)


def prepare(rebuild_inputs=False):
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'manifest.private.json'
    if path.exists():
        m=json.loads(path.read_text())
        for name,expected in m['source_sha256'].items():
            assert sha((ROOT/name).read_bytes())==expected, f'Source changed: {name}'
        return m
    archive=json.loads((OUT/'manifest.json').read_text()) if (OUT/'manifest.json').exists() else None
    if archive and not rebuild_inputs:raise RuntimeError('Run rebuild-inputs to reconstruct archived inputs offline; paid resume still requires private state')
    rng=random.Random(SEED);items=[]
    for category,n in [('standard',15),('copyright',8),('contextual',7)]:
        for r in rng.sample([r for r in readcsv('harmbench_behaviors.csv') if r['FunctionalCategory']==category],n):
            context=r['ContextString'].strip()
            if category=='contextual':assert context
            text=(context+'\n\n'+r['Behavior']) if context else r['Behavior']
            items.append(dict(item_id=r['BehaviorID'],benchmark='HarmBench',domain=category,gold_label='unsafe',input_text=text,native_system=None,stratum='HarmBench/'+category,population_n={'standard':200,'copyright':100,'contextual':100}[category],sample_n=n))
    xs=readcsv('xstest_prompts.csv')
    for label in ['safe','unsafe']:
        pool=[r for r in xs if r['label']==label]
        chosen=balanced_sample(pool,20,'type',rng)
        counts=Counter(r['type'] for r in chosen);pcounts=Counter(r['type'] for r in pool)
        for r in chosen:items.append(dict(item_id='xstest_'+r['id'],benchmark='XSTest',domain=r['type'],gold_label=label,input_text=r['prompt'],native_system=None,stratum='XSTest/'+label+'/'+r['type'],population_n=pcounts[r['type']],sample_n=counts[r['type']]))
    for setting in ['aligned','conflict']:
        name='iheval_'+setting+'.json';source=DATA/'corrected_v1'/name
        assert sha(source.read_bytes())==SOURCES[name][1]
        pool=json.loads(source.read_text())
        # Select distinct parent defenses; avoid treating sibling probes as independent items.
        groups=defaultdict(list)
        for r in pool:groups[r['id'].rsplit('-',1)[0]].append(r)
        for parent in rng.sample(sorted(groups),15):
            r=rng.choice(groups[parent]);assert r['system'] and r['instruction']
            assert not r.get('conversation_history') and not r.get('tool'), 'Unsupported native roles'
            items.append(dict(item_id='iheval_extract_'+setting+'_'+r['id'],benchmark='IHEval',domain=setting,gold_label='conflict' if setting=='conflict' else 'non-conflict',input_text=r['instruction'],native_system=r['system'],native_answer=r['answer'],native_id=r['id'],cluster_id=parent,stratum='IHEval/extract/'+setting,population_n=len(groups),sample_n=15,within_cluster_n=len(groups[parent]),native_source=SOURCES[name][0]))
    assert len(items)==100
    rng.shuffle(items)
    registry=yaml.safe_load((ROOT/'prompts/registry_v3.yaml').read_text())
    conditions=[dict(prompt_family='No-controller',clarity_level='none',prompt_variant='control',controller=None)]
    conditions += [dict(prompt_family=f,clarity_level=c,prompt_variant=v,controller=p) for f,d in registry.items() for c,vs in d['clarity_levels'].items() for v,p in vs.items()]
    assert len(conditions)==73
    # Baselines first so newly observed policy blocks can stop later item variants.
    tasks=[]
    for condition in conditions:
        for item in items:
            t={**item,**condition,'model_name':MODEL}
            t['task_id']=sha(canonical(t))
            tasks.append(t)
    sources=['prompts/registry_v3.yaml','benchmarks/artifacts/datasets/harmbench_behaviors.csv','benchmarks/artifacts/datasets/xstest_prompts.csv']+[f'benchmarks/artifacts/datasets/corrected_v1/{n}' for n in SOURCES]
    m=dict(version='corrected-v1',seed=SEED,items=items,conditions=conditions,tasks=tasks,
        source_sha256={n:sha((ROOT/n).read_bytes()) for n in sources},
        native_repository='ytyz1307zzh/IHEval',native_revision='726a62924c3050045954df94347d53fe2bd1090d',
        model=MODEL,reasoning_effort='low',max_output_tokens=2048,study_budget_usd=250,astra_cap_usd=CAP,
        controller_composition='native system followed by two newlines and registry controller; baseline retains native system only',
        original_judge='meta-llama--Llama-3.1-8B-Instruct',judging='unchanged original output-only A/B/C plus adjudication; collaborator cluster',
        limitations=['IHEval population is native system-prompt-extract aligned/default plus conflict/weak_defense, not original 2520 flattened probes',
        'Native task system remains in IHEval control; control means no added controller',
        'Matched historical-model generation reruns required for corrected inputs',
        '100 sampled items do not establish benchmark-wide fine-grained rankings',
        'Astra low reasoning / 2048 total output tokens differs from historical non-reasoning 512-token protocol'])
    # Public manifest contains provenance and sample metadata, but no benchmark text/answers.
    public={k:v for k,v in m.items() if k not in {'items','tasks','conditions'}}
    public['items']=[{k:v for k,v in i.items() if k not in {'input_text','native_system','native_answer'}}|{'input_sha256':sha(canonical({'input_text':i['input_text'],'native_system':i['native_system']}))} for i in items]
    public['planned_responses']=len(tasks);public['conditions']=73
    if archive:assert public==archive, 'Rebuilt inputs differ from frozen public manifest'
    save(path,m)
    save(OUT/'manifest.json',public)
    return m


def payload(task):
    system='\n\n'.join(x for x in [task['native_system'],task['controller']] if x)
    messages=([dict(role='system',content=system)] if system else [])+[dict(role='user',content=task['input_text'])]
    return dict(model=MODEL,input=messages,reasoning={'effort':'low'},max_output_tokens=2048,store=False)


def reservation(t):
    # UTF8 byte bound plus 2048 framing-token margin; Batch rates $5/$25 per million.
    return ((len(canonical(payload(t)))+2048)*5*1.25+2048*25)/1e6

def usage_cost(u):return (u['input_tokens']*5+u['output_tokens']*25)/1e6

def usage_upper(u):
    # Budget conservatively covers possible 1.25x cache-write input pricing.
    return (u['input_tokens']*6.25+u['output_tokens']*25)/1e6


def key():
    for line in (ROOT.parent/'.env.frontier').read_text().splitlines():
        if line.startswith('OPENAI_API_KEY='):return line.split('=',1)[1].strip().strip('"').strip("'")
    raise RuntimeError('Missing local OpenAI API key')


def api(path,method='GET',body=None,content_type='application/json',raw=False):
    data=body if isinstance(body,bytes) else (canonical(body) if body is not None else None)
    req=urllib.request.Request('https://api.openai.com/v1'+path,data=data,method=method,headers={'Authorization':'Bearer '+key(),'Content-Type':content_type})
    try:
        with urllib.request.urlopen(req,timeout=60) as r:data=r.read()
    except urllib.error.HTTPError as e:
        try:err=json.load(e).get('error',{})
        except Exception:err={}
        raise RuntimeError(f"HTTP {e.code}; error code={err.get('code')}; type={err.get('type')}") from None
    return data if raw else json.loads(data)


def state():
    p=OUT/'state.private.json'
    if p.exists():return json.loads(p.read_text())
    if rows(OUT/'usage.jsonl'):raise RuntimeError('Public results without private ledger: refusing duplicate calls')
    return {'batches':[]}


def total(s):return sum(b['accounted_usd'] for b in s['batches'])


def pending(m,s):
    used={tid for b in s['batches'] for tid in b['task_ids']}
    blocked={(r['benchmark'],r['item_id']) for r in rows(OUT/'responses.private.jsonl') if r.get('error_code') in POLICY_BLOCKS}
    return [t for t in m['tasks'] if t['task_id'] not in used and (t['benchmark'],t['item_id']) not in blocked]


def submit(m,count):
    s=state()
    if any(not b.get('reconciled') for b in s['batches']):raise RuntimeError('Reconcile outstanding/uncertain batch before another submission')
    chosen=pending(m,s)[:count]
    if not chosen:return export(m)
    amount=sum(reservation(t) for t in chosen)
    if total(s)+amount>CAP:raise RuntimeError(f'Would exceed $75 Astra cap: reserved total ${total(s)+amount:.2f}; submit fewer requests')
    batch_key=uuid.uuid4().hex
    # Write intent before upload/create; a crash/ambiguous POST is never auto-retried.
    b=dict(local_id=batch_key,task_ids=[t['task_id'] for t in chosen],accounted_usd=amount,status='submission_intent',reconciled=False)
    s['batches'].append(b);save(OUT/'state.private.json',s)
    data=b''.join(canonical(dict(custom_id=t['task_id'],method='POST',url='/v1/responses',body=payload(t)))+b'\n' for t in chosen)
    boundary='----frontier'+uuid.uuid4().hex
    upload=(f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="frontier.jsonl"\r\nContent-Type: application/jsonl\r\n\r\n'.encode()+data+f'\r\n--{boundary}--\r\n'.encode())
    f=api('/files','POST',upload,'multipart/form-data; boundary='+boundary)
    b['input_file_id']=f['id'];b['status']='uploaded';save(OUT/'state.private.json',s)
    remote=api('/batches','POST',dict(input_file_id=f['id'],endpoint='/v1/responses',completion_window='24h',metadata={'study':'promptcontrol-corrected-v1','local_id':batch_key}))
    b.update(id=remote['id'],status=remote['status']);save(OUT/'state.private.json',s)
    return export(m)


def poll(m):
    s=state();tasks={t['task_id']:t for t in m['tasks']}
    existing={r['task_id']:r for r in rows(OUT/'responses.private.jsonl')}
    for b in s['batches']:
        if b.get('reconciled'):
            b['accounted_usd']=sum(usage_upper(existing[tid]['usage']) if tid in existing and existing[tid].get('usage') else reservation(tasks[tid]) for tid in b['task_ids'])
            continue
        if not b.get('input_file_id') and not b.get('id'):
            # Batch creation occurs only after the uploaded file ID is persisted.
            # An interrupted upload therefore cannot have launched inference.
            b.update(status='upload_interrupted_no_inference',reconciled=True,accounted_usd=0,abandoned_task_ids=b['task_ids'],task_ids=[])
            save(OUT/'state.private.json',s)
            continue
        if not b.get('id'):
            # Read-only reconciliation of an uncertain create, matching unique metadata.
            listing=api('/batches?limit=100')['data']
            matches=[x for x in listing if (x.get('metadata') or {}).get('local_id')==b['local_id']]
            if len(matches)!=1:raise RuntimeError('Uncertain batch submission requires reconciliation; no POST retried')
            b['id']=matches[0]['id'];save(OUT/'state.private.json',s)
        remote=api('/batches/'+b['id']);b.update(status=remote['status'],request_counts=remote.get('request_counts'))
        if b['status'] not in TERMINAL:continue
        fetched={}
        for field in ['output_file_id','error_file_id']:
            if not remote.get(field):continue
            for line in api('/files/'+remote[field]+'/content',raw=True).decode().splitlines():
                x=json.loads(line);tid=x['custom_id'];assert tid in b['task_ids'];assert tid not in fetched
                t=tasks[tid];response=x.get('response') or {};body=response.get('body') or {};err=x.get('error') or body.get('error') or {}
                r=dict(task_id=tid,benchmark=t['benchmark'],item_id=t['item_id'],status=body.get('status','api_error'),error_code=err.get('code'))
                if body.get('usage'):
                    final=[p.get('text',p.get('refusal','')) for o in body.get('output',[]) if o.get('type')=='message' for p in o.get('content',[]) if p.get('type') in {'output_text','refusal'}]
                    r.update(model_output='\n'.join(final),usage=body['usage'],estimated_cost_usd=usage_cost(body['usage']),response_model=body.get('model'),incomplete_details=body.get('incomplete_details'))
                fetched[tid]=r
        for tid,r in fetched.items():
            if tid in existing:
                assert r==existing[tid], 'Changed terminal result'
            else:
                with (OUT/'responses.private.jsonl').open('a') as f:f.write(json.dumps(r)+'\n');f.flush();os.fsync(f.fileno())
                existing[tid]=r
        # Missing/failed response charges stay conservatively reserved.
        b['accounted_usd']=sum(usage_upper(existing[tid]['usage']) if tid in existing and existing[tid].get('usage') else reservation(tasks[tid]) for tid in b['task_ids'])
        b['reconciled']=True
    save(OUT/'state.private.json',s)
    return export(m)


def export(m):
    s=state();rs={r['task_id']:r for r in rows(OUT/'responses.private.jsonl')};blocked={(r['benchmark'],r['item_id']) for r in rs.values() if r.get('error_code') in POLICY_BLOCKS}
    attempted={tid for b in s['batches'] for tid in b['task_ids']};planned=[];judge=[]
    for t in m['tasks']:
        r=rs.get(t['task_id'],{})
        status=r.get('status') or ('pending_batch' if t['task_id'] in attempted else 'skipped_provider_block' if (t['benchmark'],t['item_id']) in blocked else 'not_submitted')
        row={**t,**r,'status':status,'model_output':r.get('model_output',''),'messages_json':json.dumps(payload(t)['input'],ensure_ascii=False),'input_version':'corrected-v1'}
        planned.append(row)
        if status=='completed' and row['model_output'].strip():judge.append(row)
    fields=['task_id','item_id','benchmark','domain','gold_label','input_text','prompt_family','clarity_level','prompt_variant','model_name','model_output','status','messages_json','input_version']
    for name,rr in [('all_statuses.private.csv',planned),('judge_input.private.csv',judge)]:
        with (OUT/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rr)
    public=[{k:v for k,v in r.items() if k not in {'model_output','incomplete_details'}}|{'output_chars':len(r.get('model_output',''))} for r in rs.values()]
    (OUT/'usage.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in public))
    summary=dict(planned=7300,submitted=len(attempted),returned=len(rs),ready_for_original_judge=len(judge),
                 statuses=dict(Counter(r['status'] for r in planned)),usage_estimate_usd=sum(r.get('estimated_cost_usd',0) for r in rs.values()),
                 astra_accounted_usd=total(s),astra_cap_usd=CAP,study_cap_usd=250,
                 batches=[{k:b[k] for k in ['status','reconciled','accounted_usd','request_counts'] if k in b} for b in s['batches']],
                 judging_status='not_run; original collaborator cluster pipeline')
    save(OUT/'summary.json',summary)
    lines=['# Astra corrected-sample progress', '',
           f"Planned: {summary['planned']}; submitted: {summary['submitted']}; returned: {summary['returned']}.",
           f"Completed nonempty responses ready for original judging: {len(judge)}.",
           f"Usage-based cost estimate: ${summary['usage_estimate_usd']:.6f}. Conservative pending/error accounting: ${total(s):.6f} / $75.", '',
           '## Planned-row status', '', '| Status | Rows |', '|---|---:|']
    lines += [f'| {k} | {v} |' for k,v in sorted(summary['statuses'].items())]
    lines += ['', '## Batch status', '', '| Batch | State | Reconciled |', '|---|---|---|']
    lines += [f"| {i+1} | {b['status']} | {b.get('reconciled',False)} |" for i,b in enumerate(s['batches'])]
    lines += ['', '## Interpretation and handoff', '',
              'No production judge has run. Use the original Llama-3.1-8B pipeline on the collaborator cluster; confirm the checkpoint from the original manifests.',
              'Public metadata excludes raw responses. judge_input.private.csv contains completed outputs; all_statuses.private.csv retains every planned row.',
              'The corrected sample is 30 HarmBench, 40 XSTest, and 30 native IHEval system-prompt-extraction items, with all 73 controller conditions.',
              'Native IHEval system instructions remain in the no-controller baseline. Corrected inputs require matched historical-model reruns.',
              'Provider bio/cyber policy errors are not model refusals; later unsubmitted variants of affected items are skipped. No replacements or prompt rewriting.',
              'Costs are estimates, not invoices. Input is priced as uncached; possible cache-write surcharges are not itemized in returned usage. Pending reservations and settled budget accounting include a 25% input margin.', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    return summary


def advance(m):
    """Reconcile once and submit one affordable chunk; caller controls waiting."""
    summary=poll(m)
    s=state()
    if any(not b.get('reconciled') for b in s['batches']):
        return {**summary,'next_action':'wait'}
    rr=rows(OUT/'responses.private.jsonl')
    billing={'credit_balance_exhausted','insufficient_quota','organization_spend_limit_exceeded','project_spend_limit_exceeded','organization_usage_limit_exceeded'}
    if any(r.get('error_code') in billing for r in rr):
        return {**summary,'next_action':'billing_attention_required'}
    left=pending(m,s)
    if not left:return {**summary,'next_action':'generation_finished'}
    available=CAP-total(s);count=0;reserved=0
    for t in left[:1000]:
        amount=reservation(t)
        if reserved+amount>available:break
        count+=1;reserved+=amount
    if count==0:return {**summary,'next_action':'budget_cap_reached'}
    result=submit(m,count)
    return {**result,'next_action':'wait'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['download-sources','rebuild-inputs','prepare','submit','poll','advance','run','export']);p.add_argument('--count',type=int,default=100)
    args=p.parse_args();assert 0<args.count<=1000
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=='download-sources':
            download_sources();print('Pinned source hashes verified.');sys.exit(0)
        m=prepare(rebuild_inputs=args.action=='rebuild-inputs')
        if args.action=='rebuild-inputs':
            print(json.dumps({'reconstructed_items':len(m['items']),'reconstructed_tasks':len(m['tasks']),'paid_requests':0}));sys.exit(0)
        if args.action=='run':
            while True:
                result=advance(m)
                print(json.dumps({k:result[k] for k in ['submitted','returned','ready_for_original_judge','usage_estimate_usd','astra_accounted_usd','next_action']}),flush=True)
                if result['next_action']!='wait':break
                time.sleep(45)
            sys.exit(0)
        result=advance(m) if args.action=='advance' else submit(m,args.count) if args.action=='submit' else poll(m) if args.action=='poll' else export(m)
        print(json.dumps(result,indent=2))
