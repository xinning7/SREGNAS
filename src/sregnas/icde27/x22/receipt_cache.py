
import atexit
import hashlib
import json
import os
from pathlib import Path

_cache={}
_registered=False

def load(path,expected):
    global _registered
    path=Path(path)
    if not _registered:
        _registered=True;atexit.register(finalize);receipt('started')
    stat=path.stat();signature=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
    key=(str(path),expected)
    cached=_cache.get(key)
    if cached is None or cached[0]!=signature:
        content=path.read_bytes()
        if hashlib.sha256(content).hexdigest()!=expected:
            receipt('failed',error='partition manifest changed after graph preparation')
            raise AssertionError('partition manifest changed after graph preparation')
        cached=(signature,json.loads(content));_cache[key]=cached
    return cached[1]

def receipt(status,**fields):
    root=os.environ.get('SREGNAS_ICDE27_RECORD_DIR')
    if not root:return
    path=Path(root)/'x22_framework'/f'partition_receipts_{os.getpid()}.json'
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(status=status,paths=len(_cache),**fields))+'\n');tmp.replace(path)

def finalize():
    if not _registered:return
    try:
        for path,expected in _cache:
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
                raise AssertionError('partition manifest changed after graph preparation')
        receipt('complete',assertion='PASS')
    except BaseException as exc:
        receipt('failed',error=repr(exc));raise

def assertions(out):
    rows=[json.loads(p.read_text()) for p in (Path(out)/'x22_framework').glob('partition_receipts_*.json')]
    return {'x22_complete_partition_receipts':all(r.get('status')=='complete' and r.get('assertion')=='PASS' for r in rows)}
