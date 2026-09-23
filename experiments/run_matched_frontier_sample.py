"""Cluster-only matched generation using the corrected frozen message manifest.

Uses an already-running vLLM server. No remote machine/GPU is provisioned.
Failed tasks stay explicit and are not silently retried on resume.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.run_frontier_sample import payload
from models.vllm_client import VLLMClient


class SingleAttemptVLLM(VLLMClient):
    @property
    def client(self):
        # Existing wrapper counts max_retries as attempts; disable SDK retries separately.
        return super().client.with_options(max_retries=0)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--base-url',default='http://localhost:8000/v1')
    p.add_argument('--max-tokens',type=int,required=True,help='Explicitly choose and report the historical-model output cap.')
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();m=json.loads(a.manifest.read_text());a.output_dir.mkdir(parents=True,exist_ok=True)
    with (a.output_dir/'.run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config={'input_manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'model':a.model,'max_tokens':a.max_tokens,'temperature':0.0,'base_url':a.base_url}
        cp=a.output_dir/'config.json'
        if cp.exists():assert json.loads(cp.read_text())==config,'Cannot change configuration during resume'
        else:cp.write_text(json.dumps(config,indent=2)+'\n')
        rp=a.output_dir/'responses.jsonl'
        old=[json.loads(s) for s in rp.read_text().splitlines()] if rp.exists() else []
        done={r['task_id']:r for r in old};write_lock=threading.Lock()
        client=SingleAttemptVLLM(model_name=a.model,base_url=a.base_url,max_retries=1,enable_cache=False)
        def run(t):
            messages=payload(t)['input'];system=messages[0]['content'] if messages[0]['role']=='system' else None
            r={'task_id':t['task_id'],'model_name':a.model}
            try:
                text,meta=client.generate(system_prompt=system,user_prompt=t['input_text'],model=a.model,temperature=0.0,max_tokens=a.max_tokens)
                finish=meta.get('finish_reason')
                r.update(model_output=text,metadata=meta,status='incomplete' if finish=='length' else 'completed')
            except Exception as e:r.update(status='error',error_type=type(e).__name__)
            with write_lock:
                with rp.open('a') as f:f.write(json.dumps(r)+'\n');f.flush();os.fsync(f.fileno())
                done[t['task_id']]=r
                print(json.dumps({'recorded':len(done),'status':r['status']}),flush=True)
        with ThreadPoolExecutor(max_workers=a.workers) as pool:list(pool.map(run,[t for t in m['tasks'] if t['task_id'] not in done]))
        fields=['task_id','item_id','benchmark','domain','gold_label','input_text','prompt_family','clarity_level','prompt_variant','model_name','model_output','status']
        with (a.output_dir/'judge_input.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
            for t in m['tasks']:
                r=done.get(t['task_id'],{})
                if r.get('status')=='completed' and r.get('model_output','').strip():w.writerow({**t,**r})


if __name__=='__main__':main()
