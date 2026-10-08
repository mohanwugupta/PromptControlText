"""Restore the verified existing ledger to the empty replacement worker; no API keys or generation."""
import hashlib,json,shlex,subprocess
from pathlib import Path
root=Path.cwd();out=root/'.local/frontier/main-20261008'
state=json.loads((out/'control-state.json').read_text());ready=json.loads((out/'resume-ready.json').read_text())
assert not state['terminated'] and state['pod_name']=='PromptControlText-main-batch-resume-1'
bundle=(out/'resume-ledger.tar.gz').read_bytes();assert hashlib.sha256(bundle).hexdigest()==ready['restore_bundle_sha256']
remote='''import hashlib,io,json,sqlite3,sys,tarfile
from pathlib import Path
from frontier.main_run import MainLedger,main_cases
root=Path.cwd();out=root/'.local/frontier/main-20261008';out.mkdir(parents=True,exist_ok=True)
assert not (out/'main.sqlite').exists(),'Existing ledger must not be overwritten'
data=sys.stdin.buffer.read();assert hashlib.sha256(data).hexdigest()==BUNDLE_SHA
with tarfile.open(fileobj=io.BytesIO(data)) as t:
 for m in t.getmembers():
  assert not m.name.startswith('/') and '..' not in m.name.split('/') and not m.issym() and not m.islnk()
 t.extractall(out)
assert hashlib.sha256((out/'main.sqlite').read_bytes()).hexdigest()==LEDGER_SHA
m=json.loads((root/'artifacts/frontier/manifest.json').read_text());c=json.loads((root/'configs/frontier-main.json').read_text())
ledger=MainLedger(out/'main.sqlite',m,c);expected={r['request_id']:r for r in main_cases(m,c)}
for row in ledger.db.execute('SELECT * FROM requests'):assert json.loads(row['case_json'])==expected[row['request_id']]
assert ledger.summary()==json.loads((out/'status.json').read_text())
assert len(list(ledger.db.execute('SELECT * FROM requests')))==1615
print(json.dumps({'restored_requests':1615,'completed':sum(ledger.summary()['done_by_provider'].values()),'main_accounted_usd':ledger.summary()['main_accounted_usd'],'frozen_case_mapping':True,'paid_requests_submitted':0}))
'''.replace('BUNDLE_SHA',repr(ready['restore_bundle_sha256'])).replace('LEDGER_SHA',repr(ready['ledger_sha256']))
args=['ssh','-i',str(root/'.local/frontier/id_ed25519'),'-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes',
 '-o','UserKnownHostsFile='+str(root/'.local/frontier/known_hosts'),'-o','ConnectTimeout=15','-p',str(state['ssh_port']),
 'root@'+state['ssh_host'],'cd /workspace/PromptControlText && .venv/bin/python -c '+shlex.quote(remote)]
r=subprocess.run(args,input=bundle,capture_output=True,timeout=90)
print(r.stdout.decode());print(r.stderr.decode());
if r.returncode:raise SystemExit(r.returncode)
(out/'restore-verification.json').write_bytes(r.stdout)
