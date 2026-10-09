"""Recover already-submitted provider jobs from the final worker ledger using GET only."""
import hashlib
import json
import shutil
import sqlite3
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from frontier.batch_api import Client
from frontier.main_run import MainLedger, main_cases, step, validate_main
from scripts.credentials import credential_environment

root = Path.cwd()
local = root / '.local/frontier/main-20261008'
prior = json.loads((local / 'latest-checkpoint.json').read_text())
source = Path(prior['path'])
assert source.joinpath('live-backup.sqlite').is_file()
assert hashlib.sha256((source / 'checkpoint.tar.gz').read_bytes()).hexdigest() == prior['sha256']
manifest = json.loads((root / 'artifacts/frontier/manifest.json').read_text())
config = validate_main(json.loads((root / 'configs/frontier-main.json').read_text()), manifest,
                       json.loads((root / 'configs/frontier-pilot.json').read_text()))
dest = local / 'checkpoints' / ('recovered-' + str(time.time_ns()))
dest.mkdir()
with sqlite3.connect('file:' + str(source / 'live-backup.sqlite') + '?mode=ro', uri=True) as src:
    with sqlite3.connect(dest / 'live-backup.sqlite') as dst:
        src.backup(dst)
shutil.copy2(source / 'supervisor-status.json', dest / 'supervisor-status.json')
shutil.copytree(source / 'provider-results', dest / 'provider-results')
ledger = MainLedger(dest / 'live-backup.sqlite', manifest, config)
assert ledger.summary() == prior['summary']
before_ids = {r[0] for r in ledger.db.execute('select request_id from requests')}
calls = []

class ReadOnlyClient(Client):
    def request(self, provider, path, method='GET', data=None, content_type='application/json', raw=False):
        if method != 'GET' or data is not None:
            raise RuntimeError('Recovery permits GET only')
        calls.append({'provider': provider, 'path': path, 'method': method})
        return super().request(provider, path, method, data, content_type, raw)

status = step(ledger, list(main_cases(manifest, config)),
              ReadOnlyClient(credential_environment(root / '.env'), 60),
              dest / 'provider-results', submit=False)
assert {r[0] for r in ledger.db.execute('select request_id from requests')} == before_ids
ledger.checkpoint(dest / 'verified')
summary = ledger.summary()
ledger.db.close()
provenance = {'kind': 'read_only_provider_recovery', 'source_checkpoint_sha256': prior['sha256'],
              'source_retrieved_epoch': prior['retrieved_epoch'], 'recovered_epoch': time.time(),
              'http_calls': calls, 'new_submissions': 0, 'original_worker_supervisor_preserved': True,
              'result': status}
(dest / 'recovery.json').write_text(json.dumps(provenance, indent=2) + '\n')
with tarfile.open(dest / 'checkpoint.tar.gz', 'w:gz') as archive:
    for name in ('live-backup.sqlite', 'verified', 'provider-results', 'supervisor-status.json', 'recovery.json'):
        archive.add(dest / name, arcname=name)
checkpoint = {'path': str(dest), 'retrieved_epoch': provenance['recovered_epoch'],
              'sha256': hashlib.sha256((dest / 'checkpoint.tar.gz').read_bytes()).hexdigest(),
              'summary': summary, 'recovery': provenance}
(dest / 'source-checkpoint.json').write_text(json.dumps(prior, indent=2) + '\n')
tmp = local / 'latest-checkpoint.json.tmp'
tmp.write_text(json.dumps(checkpoint, indent=2) + '\n')
tmp.replace(local / 'latest-checkpoint.json')
print(json.dumps({'recovery_directory': str(dest), 'read_only_http_calls': len(calls), 'status': status,
                  **{k: v for k, v in summary.items() if k != 'batches'}}))
