"""Back up the existing worker; clean it up after its controller stops."""
import json,subprocess,time
from pathlib import Path
root=Path.cwd();out=root/'.local/frontier/main-20261008';control=out/'control.py'
def run(action):
 r=subprocess.run([str(root/'.venv/bin/python'),str(control),action],cwd=root,capture_output=True,timeout=110)
 if r.returncode:
  print(json.dumps({'monitor':'action_failed','action':action,'exit':r.returncode}),flush=True)
  return False
 return True
while True:
 s=json.loads((out/'control-state.json').read_text())
 if s.get('terminated'):
  print('{"monitor":"worker_absent"}',flush=True);break
 if not run('retrieve'):
  if time.time()>s['deadline_epoch']+300:break
  time.sleep(60);continue
 record=json.loads((out/'latest-checkpoint.json').read_text());path=Path(record['path'])
 sup=path/'supervisor-status.json';status=json.loads(sup.read_text()).get('state') if sup.exists() else None
 print(json.dumps({'monitor':'checkpoint_saved','done':record['summary']['done_by_provider'],'accounted_usd':record['summary']['main_accounted_usd'],'supervisor':status}),flush=True)
 if status in ('complete','needs_review','deadline') or time.time()>s['deadline_epoch']-90:
  if run('retrieve-final') and run('delete'):
   print(json.dumps({'monitor':'verified_cleanup','reason':status or 'deadline'}),flush=True);break
  time.sleep(30);continue
 time.sleep(min(180,max(1,s['deadline_epoch']-time.time()-90)))
