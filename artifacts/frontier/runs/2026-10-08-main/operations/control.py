import hashlib,io,json,os,subprocess,sys,tarfile,time,urllib.request,urllib.error
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from scripts.credentials import credential_environment
from frontier.providers import NoRedirect
ROOT=Path.cwd();OUT=ROOT/'.local/frontier/main-20261008';STATE=OUT/'control-state.json'
env=credential_environment(ROOT/'.env')
headers={'Authorization':'Bearer '+env['RUNPOD_API_KEY'],'User-Agent':'PromptControlText-frontier/1'}
def api(path='',method='GET',body=None):
 req=urllib.request.Request('https://rest.runpod.io/v1/pods'+path,method=method,headers={**headers,'Content-Type':'application/json'},data=json.dumps(body).encode() if body else None)
 with urllib.request.build_opener(NoRedirect).open(req,timeout=45) as r:
  b=r.read();return json.loads(b) if b else {'http_status':r.status}
def save(s):
 p=STATE.with_suffix('.tmp');p.write_text(json.dumps(s,indent=2)+'\n');p.replace(STATE)
def ssh(s,cmd,data=None,timeout=90):
 address=s['ssh_host'];port=str(s['ssh_port']);assert all(x.isdigit() for x in address.split('.'))
 args=['ssh','-i',str(ROOT/'.local/frontier/id_ed25519'),'-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(ROOT/'.local/frontier/known_hosts'),'-o','ConnectTimeout=15','-p',port,'root@'+address,cmd]
 return subprocess.run(args,input=data,capture_output=True,timeout=timeout)
action=sys.argv[1]
if action=='list':
 pods=api();print(json.dumps([{'id':p['id'],'name':p.get('name'),'costPerHr':p.get('costPerHr'),'desiredStatus':p.get('desiredStatus')} for p in pods]))
elif action=='create':
 assert not STATE.exists(),'Existing worker state: inspect before creating'
 created=time.time()
 pod=api(method='POST',body={'computeType':'CPU','cloudType':'SECURE','cpuFlavorIds':['cpu3c'],'cpuFlavorPriority':'custom','vcpuCount':2,'name':'PromptControlText-main-batch','templateId':'runpod-ubuntu-2404','containerDiskInGb':10,'volumeInGb':0,'ports':['22/tcp'],'countryCodes':['US']})
 (OUT/'created-pod.json').write_text(json.dumps(pod,indent=2))
 s={'pod_id':pod['id'],'created_local_epoch':created,'deadline_epoch':created+3*3600-30,'terminated':False,'source_commit':'8db7bdd680c36b5f434b7cc6b5f564c96b22c88a'};save(s)
 watchdog=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','frontier.runpod_watchdog','--state',str(STATE)],stdout=(OUT/'watchdog.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
 awake=subprocess.Popen(['caffeinate','-i','-w',str(watchdog.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
 s.update(watchdog_pid=watchdog.pid,caffeinate_pid=awake.pid);save(s)
 print(json.dumps({'pod_id':pod['id'],'costPerHr':pod.get('costPerHr'),'deadline_epoch':s['deadline_epoch'],'watchdog_pid':watchdog.pid}))
elif action=='status':
 s=json.loads(STATE.read_text());p=api('/'+s['pod_id']);(OUT/'latest-pod.json').write_text(json.dumps(p,indent=2))
 print(json.dumps({k:p.get(k) for k in ['id','name','costPerHr','desiredStatus','publicIp','portMappings','vcpuCount','memoryInGb','imageName']}))
 if p.get('publicIp') and p.get('portMappings',{}).get('22'):
  s.update(ssh_host=p['publicIp'],ssh_port=p['portMappings']['22']);save(s)
elif action=='bundle':
 files=['configs/frontier-pilot.json','configs/frontier-main.json','artifacts/frontier/manifest.json','requirements-frontier.txt','scripts/credentials.py','prompts/registry.py']+[str(p) for p in Path('frontier').glob('*.py')]
 hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}
 provenance={'source_commit':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),'files':hashes}
 for name in files:assert subprocess.check_output(['git','show',provenance['source_commit']+':'+name])==Path(name).read_bytes(),'Uncommitted deployment source'
 (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 with tarfile.open(OUT/'source.tar.gz','w:gz') as tar:
  for p in files:tar.add(p,arcname=p)
  tar.add(OUT/'provenance.json',arcname='provenance.json')
 print(json.dumps({'files':len(files),'source_sha256':hashlib.sha256((OUT/'source.tar.gz').read_bytes()).hexdigest()}))
elif action=='deploy':
 s=json.loads(STATE.read_text())
 cmd='mkdir -p /workspace/PromptControlText && cd /workspace/PromptControlText && tar -xzf - && uv venv .venv && uv pip install --python .venv/bin/python -r requirements-frontier.txt && .venv/bin/python -m frontier.main_run'
 r=ssh(s,cmd,(OUT/'source.tar.gz').read_bytes(),180);print(r.stdout.decode());print(r.stderr.decode());raise SystemExit(r.returncode)
elif action=='credentials':
 s=json.loads(STATE.read_text());assert not s.get('terminated');assert s.get('ssh_host')
 names=['OPENAI_API_KEY','ANTHROPIC_API_KEY','GEMINI_API_KEY','ANTHROPIC_WORKSPACE_ID'];vals={n:env[n] for n in names if env.get(n)};assert len(vals)==4
 remote="cd /workspace/PromptControlText && .venv/bin/python -c 'import json,os,sys; from pathlib import Path; d=json.load(sys.stdin); p=Path(\".env\"); fd=os.open(str(p),os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600); os.fchmod(fd,0o600); os.write(fd,(\"\\n\".join(k+\"=\"+json.dumps(v) for k,v in d.items())+\"\\n\").encode()); os.close(fd); print(\"Provider credential file saved with mode 600\")'"
 r=ssh(s,remote,json.dumps(vals).encode());print(r.stdout.decode());print('SSH exit',r.returncode);raise SystemExit(r.returncode)
elif action=='launch':
 s=json.loads(STATE.read_text());assert s.get('purpose')=='frontier_main' and s.get('pod_name')=='PromptControlText-main-batch-resume-1' and not s.get('terminated')
 assert s['deadline_epoch']-time.time()>2*3600+600,'Insufficient time before cleanup deadline'
 remote="""import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path('/workspace/PromptControlText');os.chdir(root)
p=json.loads(Path('provenance.json').read_text())
assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in p['files'].items())
assert Path('.env').stat().st_mode & 0o777 == 0o600
out=Path('.local/frontier/main-20261008');out.mkdir(parents=True,exist_ok=True)
fd=os.open(str(out/'launch-once.json'),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
script=sys.stdin.buffer.read();Path('launch-main.sh').write_bytes(script)
log=(out/'progress.jsonl').open('ab')
proc=subprocess.Popen(['bash','launch-main.sh'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'pid':proc.pid,'source_commit':p['source_commit']}
os.write(fd,json.dumps(record).encode());os.close(fd)
print(json.dumps(record))
"""
 import shlex
 cmd='cd /workspace/PromptControlText && .venv/bin/python -c '+shlex.quote(remote)
 r=ssh(s,cmd,(OUT/'launch.sh').read_bytes(),30)
 (OUT/'launch-result.json').write_bytes(r.stdout)
 print(r.stdout.decode());print(r.stderr.decode());raise SystemExit(r.returncode)
elif action in ('retrieve','retrieve-final'):
 s=json.loads(STATE.read_text());assert not s.get('terminated')
 remote="""import io,json,sqlite3,tarfile,time
from pathlib import Path
out=Path('/workspace/PromptControlText/.local/frontier/main-20261008')
if not (out/'main.sqlite').exists(): raise SystemExit('No ledger yet')
backup=out/('live-backup-'+str(time.time_ns())+'.sqlite')
src=sqlite3.connect(out/'main.sqlite');dst=sqlite3.connect(backup);src.backup(dst);dst.close();src.close()
import sys
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for name in ['live-backup.sqlite','progress.jsonl','status.json','launch-once.json','exit-code','supervisor-status.json','supervisor-launch.json','supervisor.log']:
  p=backup if name=='live-backup.sqlite' else out/name
  if p.exists():t.add(p,arcname=name)
 if INCLUDE_RAW:
  for p in (out/'provider-results').glob('*.jsonl'):t.add(p,arcname='provider-results/'+p.name)
backup.unlink()
""".replace('INCLUDE_RAW',str(action=='retrieve-final'))
 import shlex
 r=ssh(s,'cd /workspace/PromptControlText && .venv/bin/python -c '+shlex.quote(remote),timeout=60)
 if r.returncode:print('Retrieve failed',r.returncode,r.stderr.decode()[:300]);raise SystemExit(1)
 stamp=str(time.time_ns());dst=OUT/'checkpoints'/stamp;dst.mkdir(parents=True,exist_ok=True)
 archive=dst/'checkpoint.tar.gz';archive.write_bytes(r.stdout)
 with tarfile.open(archive) as t:
  assert all(m.isfile() and not m.name.startswith('/') and '..' not in m.name.split('/') for m in t.getmembers());t.extractall(dst)
 from frontier.main_run import MainLedger,main_cases
 m=json.load(open(ROOT/'artifacts/frontier/manifest.json'));c=json.load(open(ROOT/'configs/frontier-main.json'))
 ledger=MainLedger(dst/'live-backup.sqlite',m,c)
 expected={x['request_id']:x for x in main_cases(m,c)}
 for row in ledger.db.execute('SELECT * FROM requests'):assert json.loads(row['case_json'])==expected[row['request_id']]
 ledger.checkpoint(dst/'verified')
 summary=ledger.summary();(OUT/'latest-checkpoint.json').write_text(json.dumps({'path':str(dst),'retrieved_epoch':time.time(),'sha256':hashlib.sha256(r.stdout).hexdigest(),'summary':summary},indent=2)+'\n')
 print(json.dumps(summary))
 if (dst/'exit-code').exists():print('worker_exit_code',int((dst/'exit-code').read_text()))
elif action=='supervise':
 s=json.loads(STATE.read_text());assert s['purpose']=='frontier_main'
 remote="""import hashlib,json,os,subprocess,sys
from pathlib import Path
os.chdir('/workspace/PromptControlText');data=json.load(sys.stdin)
p=Path('frontier/main_watch.py');p.write_text(data['source']);assert hashlib.sha256(p.read_bytes()).hexdigest()==data['sha256']
out=Path('.local/frontier/main-20261008')
fd=os.open(str(out/'supervisor-launch.json'),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
record={k:data[k] for k in ['deadline','source_commit','sha256']}
log=(out/'supervisor.log').open('ab')
proc=subprocess.Popen(['.venv/bin/python','-u','-m','frontier.main_watch','--deadline',str(data['deadline'])],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record['pid']=proc.pid;os.write(fd,json.dumps(record).encode());os.close(fd)
print(json.dumps(record))
"""
 import shlex
 src=(ROOT/'frontier/main_watch.py').read_text()
 data={'source':src,'sha256':hashlib.sha256(src.encode()).hexdigest(),'deadline':s['deadline_epoch'],'source_commit':s['source_commit']}
 r=ssh(s,'cd /workspace/PromptControlText && .venv/bin/python -c '+shlex.quote(remote),json.dumps(data).encode(),30)
 (OUT/'supervisor-launch.json').write_bytes(r.stdout);print(r.stdout.decode());print(r.stderr.decode());raise SystemExit(r.returncode)
elif action=='delete':
 s=json.loads(STATE.read_text());assert s.get('purpose')=='frontier_main' and s.get('pod_name')=='PromptControlText-main-batch-resume-1'
 r=json.load(open(OUT/'latest-checkpoint.json'));assert time.time()-r['retrieved_epoch']<120,'Fresh verified checkpoint required'
 p=api('/'+s['pod_id']);assert p['name']==s['pod_name']
 deleted=api('/'+s['pod_id'],method='DELETE')
 try:api('/'+s['pod_id']);raise RuntimeError('Pod still exists')
 except urllib.error.HTTPError as e:assert e.code==404
 assert s['pod_id'] not in {p['id'] for p in api()}
 s.update(terminated=True,terminated_epoch=time.time(),absence_http_status=404,absent_from_list=True)
 save(s)
 result={'pod_id':s['pod_id'],'deleted_utc_epoch':s['terminated_epoch'],'get_status':404,'absent_from_list':True,'compute_estimate_usd':round((s['terminated_epoch']-s['created_local_epoch'])/3600*.06,6),'checkpoint_path':r['path'],'checkpoint_sha256':r['sha256']}
 (OUT/'cleanup.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
