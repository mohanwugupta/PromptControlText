"""Prepare approved main-study outputs from a verified checkpoint; never push secrets."""
import hashlib,json,shutil,sys
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from dotenv import dotenv_values
from frontier.main_run import MainLedger,main_cases,reserved_terminal_block
from frontier.batch_api import result_row
root=Path.cwd();local=root/'.local/frontier/main-20261008';dest=root/'artifacts/frontier/runs/2026-10-08-main'
checkpoint=json.loads((local/'latest-checkpoint.json').read_text());src=Path(checkpoint['path']);manifest=json.load(open(root/'artifacts/frontier/manifest.json'));config=json.load(open(root/'configs/frontier-main.json'))
ledger=MainLedger(src/'live-backup.sqlite',manifest,config);assert ledger.summary()==checkpoint['summary']
expected={c['request_id']:c for c in main_cases(manifest,config)}
rows=[json.loads(s) for s in (src/'verified/results.jsonl').read_text().splitlines()];assert len(rows)==len({r['request_id'] for r in rows})
for r in rows:assert r['case']==expected[r['request_id']]
secrets=[v for v in dotenv_values(root/'.env',interpolate=False).values() if v and len(v)>8]
files=[]
def write(path,content,immutable=False):
 assert not any(s in content for s in secrets),f'Credential found in {path}'
 path.parent.mkdir(parents=True,exist_ok=True)
 if immutable and path.exists():assert path.read_text()==content,f'Existing batch changed: {path}'
 else:path.write_text(content)
 files.append(str(path.relative_to(root)))
meta=[];completed=defaultdict(list)
for r in rows:
 result=r['result'];meta.append({'request_id':r['request_id'],'provider':r['provider'],'batch_key':r['batch_key'],'state':r['state'],'item_id':r['case']['item']['item_id'],'stratum':r['case']['item']['stratum'],'condition_id':r['case']['condition']['condition_id'],'request_body_sha256':r['case']['request_body_sha256'],'accounted_usd':r['cost'],'outcome':result.get('outcome') if result else None,'response_text_sha256':hashlib.sha256(result.get('text','').encode()).hexdigest() if result else None})
 if result is not None:completed[r['batch_key']].append(r)
raw_count=0
for key,rr in completed.items():
 response_path=dest/'responses'/f'{key}.jsonl'
 content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr)
 if response_path.exists() and response_path.read_text()!=content:
  # Published batches remain immutable. Only the reviewed completion-state
  # correction may differ; current state lives in requests.jsonl.
  old={r['request_id']:r for r in map(json.loads,response_path.read_text().splitlines())}
  assert set(old)=={r['request_id'] for r in rr}
  for row in rr:
   prior=old[row['request_id']]
   if prior!=row:
    assert prior['state']=='needs_review' and row['state']=='done'
    assert {**prior,'state':'done'}==row and reserved_terminal_block(row['case'],row['result'])
  content=response_path.read_text()
 write(response_path,content,True)
 raw=src/'provider-results'/f'{key}.jsonl'
 if not raw.exists():raw=dest/'provider-results'/f'{key}.jsonl'
 if raw.exists():
  parsed=[result_row(rr[0]['provider'],json.loads(line)) for line in raw.read_text().splitlines() if line.strip()]
  by_id={r['request_id']:r['result'] for r in rr}
  assert len(parsed)==len(by_id) and {k for k,_ in parsed}==set(by_id)
  for identity,result in parsed:
   saved={k:v for k,v in by_id[identity].items() if k!='cost_basis'};assert result==saved,(key,identity)
  write(dest/'provider-results'/raw.name,raw.read_text(),True);raw_count+=len(parsed)
write(dest/'requests.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in meta))
summary=checkpoint['summary'];count=sum(map(len,completed.values()));supervisor=json.loads((src/'supervisor-status.json').read_text())
control=json.loads((local/'control-state.json').read_text())
phase='complete' if summary['complete'] else ('stopped_'+supervisor['state'] if supervisor['state'] in ('needs_review','budget_stopped','deadline') or control.get('terminated') else 'running')
if supervisor['state']=='prepared_to_resume':phase='prepared_to_resume'
record={'as_of_utc':datetime.fromtimestamp(checkpoint['retrieved_epoch'],timezone.utc).isoformat(),'phase':'complete' if summary['complete'] else 'running','checkpoint_sha256':checkpoint['sha256'],'main':summary,'supervisor':supervisor,'published_full_response_records':count,'published_original_provider_records_in_this_export':raw_count,'generation_complete':summary['complete'],'judging_started':False,'raw_publication_authorization':'User explicitly approved publishing all records from the 10,950-case main study as they finish, including benchmark prompts, answers and synthetic IHEval test codes. API credentials/private configuration are excluded.','verification':{'frozen_case_mapping':True,'unique_request_ids':True,'sqlite_matches_checkpoint_summary':True,'original_provider_rows_match_normalized_results':True,'exported_request_records':len(rows),'secret_scan_passed':True},'source_commit':'8db7bdd680c36b5f434b7cc6b5f564c96b22c88a','continuation_commit':'e17b1398a09a96785664838ffcef259e16d8c211'}
record['phase']=phase
record['worker_terminated']=bool(control.get('terminated'))
record['completed_cases']=sum(summary['done_by_provider'].values())
record['review_cases']=sum(r['state']=='needs_review' for r in rows)
record['unsubmitted_cases']=summary['expected_requests']-len(rows)
record['read_only_recovery']=checkpoint.get('recovery')
record['terminal_block_resolution']=checkpoint.get('resolution')
record['budget_breakdown_usd']={
 'usage_estimates':round(sum(r['cost'] for r in rows if r['result'] and r['result'].get('cost_basis')=='usage_at_conservative_batch_rates'),6),
 'retained_unknown_usage_reservations':round(sum(r['cost'] for r in rows if r['result'] and r['result'].get('cost_basis')=='reserved_unknown'),6),
 'pending_reservations':round(sum(r['cost'] for r in rows if r['result'] is None),6)}
write(dest/'progress.json',json.dumps(record,indent=2)+'\n')
if checkpoint.get('resolution'):
 write(dest/'resume-resolution.json',json.dumps(checkpoint['resolution'],indent=2)+'\n',True)
cleanup=json.loads((local/'cleanup.json').read_text()) if (local/'cleanup.json').exists() else None
if cleanup:
 public_cleanup={k:v for k,v in cleanup.items() if k!='checkpoint_path'}
 write(dest/'cleanup.json',json.dumps(public_cleanup,indent=2)+'\n')
if checkpoint.get('recovery'):
 write(dest/'recovery.json',json.dumps(checkpoint['recovery'],indent=2)+'\n')
 write(dest/'operations/recover_existing.py',(local/'recover_existing.py').read_text())
write(dest/'operations/export_public.py',Path(__file__).read_text())
readme=(dest/'README.md').read_text();start=readme.index('Fifteen complete generated records') if 'Fifteen complete generated records' in readme else readme.index('Full generated records available at this checkpoint');end=readme.index('The user explicitly approved transferring')
section=f'''Full generated records available at this checkpoint are in [responses/](responses/):
**{count:,} records**. The user explicitly approved public release of all main-study
records as they finish, including the synthetic test codes from the public IHEval
benchmark. [provider-results/](provider-results/) preserves {raw_count:,} original
provider result rows verified against normalized records in this export.
[requests.jsonl](requests.jsonl) accounts for all **{len(rows):,} cases** submitted or
reserved at this snapshot: {record['completed_cases']:,} completed,
{record['review_cases']:,} needing review, and {summary['outcomes'].get('pending',0):,} pending.
Another {record['unsubmitted_cases']:,} frozen cases have not been submitted.
Each finished response maps to its frozen item, condition and request hash.
[operations/](operations/) archives the deployment, backup and export scripts;
[execution-provenance.json](execution-provenance.json) records execution hashes.
Credentials, SSH keys, private account/workspace configuration, temporary bundles
and duplicate local database backups remain excluded from the public repo.

'''
write(dest/'README.md',readme[:start]+section+readme[end:])
prov_path=dest/'execution-provenance.json';prov=json.loads(prov_path.read_text())
for script_path in files:
 if '/operations/' in script_path and script_path.endswith('.py'):
  prov['archived_script_sha256'][script_path]=hashlib.sha256((root/script_path).read_bytes()).hexdigest()
write(prov_path,json.dumps(prov,indent=2)+'\n')
files=list(dict.fromkeys(files));(local/'approved-sync-files.json').write_text(json.dumps(files,indent=2)+'\n')
payload=json.dumps([{'path':p,'mode':'100644','type':'blob','content':(root/p).read_text()} for p in files],ensure_ascii=True)
assert not any(s in payload for s in secrets)
(local/'approved-sync-payload.json').write_text(payload)
print(json.dumps({'files':len(files),'full_records':count,'original_provider_records':raw_count,'metadata_rows':len(rows),'characters':len(payload),'as_of_utc':record['as_of_utc'],'private_credential_scan':'passed'}))
