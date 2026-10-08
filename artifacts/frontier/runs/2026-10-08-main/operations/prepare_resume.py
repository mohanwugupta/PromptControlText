"""Apply the reviewed terminal-block state correction to a copy of the existing ledger."""
import hashlib, json, shutil, sqlite3, sys, tarfile, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from frontier.main_run import MainLedger, main_cases, validate_main
root=Path.cwd(); local=root/'.local/frontier/main-20261008'
old=json.loads((local/'latest-checkpoint.json').read_text()); source=Path(old['path'])
assert hashlib.sha256((source/'checkpoint.tar.gz').read_bytes()).hexdigest()==old['sha256']
manifest=json.loads((root/'artifacts/frontier/manifest.json').read_text())
config=validate_main(json.loads((root/'configs/frontier-main.json').read_text()),manifest,json.loads((root/'configs/frontier-pilot.json').read_text()))
dest=local/'checkpoints'/('resume-ready-'+str(time.time_ns())); dest.mkdir()
with sqlite3.connect('file:'+str(source/'live-backup.sqlite')+'?mode=ro',uri=True) as a:
 with sqlite3.connect(dest/'live-backup.sqlite') as b:a.backup(b)
ledger=MainLedger(dest/'live-backup.sqlite',manifest,config)
assert ledger.summary()==old['summary']
expected={c['request_id']:c for c in main_cases(manifest,config)}
before={r['request_id']:dict(r) for r in ledger.db.execute('SELECT * FROM requests')}
for key,row in before.items():assert json.loads(row['case_json'])==expected[key]
cost_before=ledger.total()
resolved=ledger.reconcile_terminal_blocks()
review=json.loads((root/'artifacts/frontier/runs/2026-10-08-main/stop-report.json').read_text())
assert set(resolved)=={r['request_id'] for r in review['review_cases']}
after={r['request_id']:dict(r) for r in ledger.db.execute('SELECT * FROM requests')}
assert set(before)==set(after)
for key,row in after.items():assert row==({**before[key],'state':'done'} if key in resolved else before[key])
assert ledger.total()==cost_before
assert not ledger.db.execute("SELECT 1 FROM batches WHERE state IN ('active','submitting','needs_review')").fetchone()
ledger.checkpoint(dest/'verified');summary=ledger.summary();ledger.db.close()
shutil.copytree(source/'provider-results',dest/'provider-results')
shutil.copy2(source/'supervisor-status.json',dest/'worker-stop-supervisor-status.json')
now=time.time()
resolution={'resolved_epoch':now,'source_checkpoint_sha256':old['sha256'],
 'request_ids':resolved,'change':'needs_review to done for explicit terminal Gemini prompt blocks',
 'request_count_before':len(before),'request_count_after':len(after),
 'main_accounted_usd_before':old['summary']['main_accounted_usd'],'main_accounted_usd_after':summary['main_accounted_usd'],
 'result_records_changed':False,'reservations_released':False,'new_submissions':0,
 'reason':'Completion state and unknown billing are separate; full per-case reservations remain held.'}
(dest/'resolution.json').write_text(json.dumps(resolution,indent=2)+'\n')
(dest/'supervisor-status.json').write_text(json.dumps({'state':'prepared_to_resume','updated_epoch':now,'new_worker_launched':False})+'\n')
with tarfile.open(dest/'checkpoint.tar.gz','w:gz') as t:
 for name in ['live-backup.sqlite','verified','provider-results','supervisor-status.json','worker-stop-supervisor-status.json','resolution.json']:t.add(dest/name,arcname=name)
checkpoint={'path':str(dest),'retrieved_epoch':now,'sha256':hashlib.sha256((dest/'checkpoint.tar.gz').read_bytes()).hexdigest(),'summary':summary,'resolution':resolution}
(dest/'source-checkpoint.json').write_text(json.dumps(old,indent=2)+'\n')
(local/'latest-checkpoint.json').write_text(json.dumps(checkpoint,indent=2)+'\n')
with tarfile.open(local/'resume-ledger.tar.gz','w:gz') as t:
 t.add(dest/'live-backup.sqlite',arcname='main.sqlite')
 t.add(dest/'provider-results',arcname='provider-results')
 t.add(dest/'verified/status.json',arcname='status.json')
 t.add(dest/'resolution.json',arcname='resume-resolution.json')
ready={'checkpoint':checkpoint,'restore_bundle_sha256':hashlib.sha256((local/'resume-ledger.tar.gz').read_bytes()).hexdigest(),
 'ledger_sha256':hashlib.sha256((dest/'live-backup.sqlite').read_bytes()).hexdigest()}
(local/'resume-ready.json').write_text(json.dumps(ready,indent=2)+'\n')
print(json.dumps({'resolved':len(resolved),'completed':sum(summary['done_by_provider'].values()),'unsubmitted':10950-len(after),'accounted_usd':summary['main_accounted_usd'],'cost_unchanged':True,'no_new_submissions':True}))
