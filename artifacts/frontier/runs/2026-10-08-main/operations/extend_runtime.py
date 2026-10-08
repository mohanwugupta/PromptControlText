import json,os,signal,subprocess,time
from pathlib import Path
root=Path.cwd();out=root/'.local/frontier/main-20261008';path=out/'control-state.json';s=json.loads(path.read_text())
assert s['pod_id']=='43dht91rp868ts' and not s['terminated']
old=s['watchdog_pid'];oldawake=s['caffeinate_pid'];oldcommand=subprocess.check_output(['ps','-p',str(old),'-o','command=']).decode()
assert 'frontier.runpod_watchdog' in oldcommand and str(path) in oldcommand
s.update(purpose='frontier_main',compute_usd_per_hour=.06,infrastructure_reserve_usd=5,deadline_epoch=s['created_local_epoch']+48*3600,continuation_source_commit='e17b1398a09a96785664838ffcef259e16d8c211')
path.write_text(json.dumps(s,indent=2)+'\n')
watch=subprocess.Popen([str(root/'.venv/bin/python'),'-u','-m','frontier.runpod_watchdog','--main-study','--state',str(path)],stdout=(out/'main-watchdog.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(1)
assert watch.poll() is None
assert 'armed' in (out/'main-watchdog.log').read_text()
awake=subprocess.Popen(['caffeinate','-i','-w',str(watch.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
s.update(watchdog_pid=watch.pid,caffeinate_pid=awake.pid);path.write_text(json.dumps(s,indent=2)+'\n')
os.kill(old,signal.SIGTERM)
try:
 a=subprocess.check_output(['ps','-p',str(oldawake),'-o','command=']).decode()
 if 'caffeinate' in a and str(old) in a:os.kill(oldawake,signal.SIGTERM)
except (ProcessLookupError,subprocess.CalledProcessError):pass
print(json.dumps({'worker':s['pod_id'],'deadline_epoch':s['deadline_epoch'],'new_watchdog_pid':watch.pid,'old_watchdog_stopped':True,'maximum_compute_usd':2.88,'infrastructure_reserve_usd':5}))
