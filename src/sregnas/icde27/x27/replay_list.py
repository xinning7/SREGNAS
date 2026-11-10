
import hashlib
import json
from pathlib import Path

def selected(cell):
    path=Path(cell['x27_replay_manifest']);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=cell['x27_replay_sha256']:
        raise ValueError('X27 replay manifest changed')
    data=json.loads(raw)
    if data['dataset']!=cell['dataset'] or len(data['rows'])!=cell['count']:
        raise ValueError('X27 replay dataset/count mismatch')
    if cell['count']!=400 or cell.get('pool_indices') is not None:
        raise ValueError('X27 cross replay requires ordered Q400 and round-robin mapping')
    from sregnas.core.architecture import Architecture
    for row in data['rows']:
        arch=json.loads(row['architecture_json'])
        if Architecture(arch).fingerprint()!=row['fingerprint']:
            raise ValueError('X27 architecture/fingerprint mismatch')
    return data['rows']
