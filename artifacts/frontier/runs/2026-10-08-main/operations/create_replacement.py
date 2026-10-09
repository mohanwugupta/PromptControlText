"""Create one replacement CPU worker, preserving the original study deadline/budget."""
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from frontier.providers import NoRedirect
from scripts.credentials import credential_environment
root=Path.cwd(); out=root/'.local/frontier/main-20261008'; state=out/'control-state.json'
old=json.loads(state.read_text());assert old['pod_id']=='43dht91rp868ts' and old['terminated']
cleanup=json.loads((out/'workers/43dht91rp868ts/cleanup.json').read_text())
assert cleanup['get_status']==404 and cleanup['absent_from_list']
assert (out/'resume-ready.json').exists() and old['deadline_epoch']-time.time()>3*3600
env=credential_environment(root/'.env')
headers={'Authorization':'Bearer '+env['RUNPOD_API_KEY'],'User-Agent':'PromptControlText-frontier/1','Content-Type':'application/json'}
opener=urllib.request.build_opener(NoRedirect)
def api(path='',method='GET',body=None):
 req=urllib.request.Request('https://rest.runpod.io/v1/pods'+path,method=method,headers=headers,data=json.dumps(body).encode() if body else None)
 with opener.open(req,timeout=45) as response:
  data=response.read();return json.loads(data) if data else {}
pods=api();assert old['pod_id'] not in {p['id'] for p in pods}
name='PromptControlText-main-batch-resume-1'
assert not any(p.get('name')==name for p in pods)
intent=out/'resume-create-intent.json'
with intent.open('x') as f:json.dump({'name':name,'created_epoch':time.time(),'prior_worker':old['pod_id']},f)
created=time.time()
pod=api(method='POST',body={'computeType':'CPU','cloudType':'SECURE','cpuFlavorIds':['cpu3c'],
 'cpuFlavorPriority':'custom','vcpuCount':2,'name':name,'templateId':'runpod-ubuntu-2404',
 'containerDiskInGb':10,'volumeInGb':0,'ports':['22/tcp'],'countryCodes':['US']})
(out/'created-pod.json').write_text(json.dumps(pod,indent=2)+'\n');(out/'created-pod.json').chmod(0o600)
if pod.get('costPerHr')!=.06 or pod.get('vcpuCount')!=2:
 api('/'+pod['id'],'DELETE');raise SystemExit('Unexpected worker specification; deleted replacement')
source_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()
current={'pod_id':pod['id'],'pod_name':name,'created_local_epoch':created,'deadline_epoch':old['deadline_epoch'],
 'study_original_created_epoch':old['created_local_epoch'],'prior_worker':old['pod_id'],
 'prior_main_compute_estimate_usd':cleanup['compute_estimate_usd'],'terminated':False,'source_commit':source_commit,
 'purpose':'frontier_main','compute_usd_per_hour':.06,'infrastructure_reserve_usd':5}
assert current['prior_main_compute_estimate_usd']+(current['deadline_epoch']-created)/3600*.06<=2.88
def save():
 tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(current,indent=2)+'\n');tmp.replace(state)
save()
watch=subprocess.Popen([str(root/'.venv/bin/python'),'-u','-m','frontier.runpod_watchdog','--main-study','--state',str(state)],
 stdout=(out/'main-watchdog.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
awake=subprocess.Popen(['caffeinate','-i','-w',str(watch.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
current.update(watchdog_pid=watch.pid,caffeinate_pid=awake.pid);save()
print(json.dumps({'pod_id':pod['id'],'cost_per_hour':pod['costPerHr'],'original_deadline_preserved':True,
 'deadline_epoch':current['deadline_epoch'],'infrastructure_reserve_usd':5,'watchdog_pid':watch.pid}))
