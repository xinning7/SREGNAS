
import fcntl
import json
import os
from pathlib import Path

_cache = {}

def lookup(filename, key):
    from sregnas.icde27 import ledger
    if os.environ.get('SREGNAS_X22_DEFER_RECORDS') == '1':
        from sregnas.icde27.x22.deferred import flush
        flush()
    path = Path(os.environ[ledger.ENV]) / filename
    if not path.exists():
        return []
    with path.open('rb') as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        st = os.fstat(stream.fileno())
        cache_key = (os.getpid(), str(path))
        state = _cache.get(cache_key)
        signature = (st.st_dev, st.st_ino)
        if state is None or state['signature'] != signature or st.st_size < state['offset']:
            state = dict(signature=signature, offset=0, groups={})
            _cache[cache_key] = state
        stream.seek(state['offset'])
        for line in stream:
            if not line.endswith(b'\n'):
                break
            row = json.loads(line)
            group = row['identity'] if filename == 'launch_ledger.jsonl' else (row['parent_candidate'], row['partition_id'])
            state['groups'].setdefault(group, []).append(row)
            state['offset'] = stream.tell()
        return list(state['groups'].get(key, ()))
