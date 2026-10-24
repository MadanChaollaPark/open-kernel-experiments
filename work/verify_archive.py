"""Offline evidence checks. Does not import GPU libraries or launch jobs."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
manifest = json.loads((ROOT / 'outputs/best_manifest.json').read_text())
entries = (manifest['verified_candidates'] + manifest['previous_scored']
           + list(manifest['verified_checks'].values()))
for entry in entries:
    source = ROOT / entry['source']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == entry['local_file_sha256'], source
    receipt = json.loads((ROOT / entry['receipt']).read_text())
    assert receipt['id'] == entry['official_job'], entry
    assert receipt['content_hash'] == entry['server_content_identifier'], entry
    assert receipt['mode'] == entry['mode'], entry
    assert receipt['status'] == 'done', entry
    assert receipt['result']['passed'] is True, entry
    assert receipt['result']['speedup'] == entry['speedup'], entry
    assert not receipt['result']['flags'], entry
for path in (ROOT / 'experiments/sources').glob('*.py'):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == path.stem, path
for record in (ROOT / 'results/local').glob('*.json'):
    data = json.loads(record.read_text())
    if isinstance(data, dict):
        for ref in data.get('source_files', {}).values():
            assert hashlib.sha256((ROOT / ref['path']).read_bytes()).hexdigest() == ref['sha256'], record
# Historical failed snapshots may intentionally contain invalid generated code.
for folder in ['outputs', 'work']:
    for path in (ROOT / folder).glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
print(f'OK: {len(entries)} verified official entries; source snapshots and record references match; published entrypoints parse.')
