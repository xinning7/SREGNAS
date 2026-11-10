
from sregnas.paths import PROJECT_ROOT as _artifact_root
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(str(_artifact_root))
OUT = ROOT/'results/icde27/x27'
CODE = Path(os.environ.get('SREGNAS_X27_CODE', str(ROOT/'.runtime/x27-code/release')))
PYTHON = __import__("sys").executable
T0 = datetime.fromisoformat('2026-11-06T04:56:21+00:00').timestamp()
END = T0 + 6*3600
DATASETS = ('Flickr','arXiv','Reddit','Products')

def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path, delimiter=','):
    with Path(path).open() as f: return list(csv.DictReader(f, delimiter=delimiter))
def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)
def append(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
def table(path, values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for v in values for k in v))
    with path.open('w') as f:
        w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader()
        for v in values:w.writerow({k:json.dumps(x,ensure_ascii=False) if isinstance(x,(dict,list)) else x for k,x in v.items()})
def environment(gpu=None, code=CODE):
    from sregnas.icde27.x25.common import environment as previous, FLAGS
    env=previous(code,flags=FLAGS)

    env['NUMBA_CACHE_DIR']=str(OUT/'numba_cache')
    env.update(CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
        SREGNAS_ALLOW_GPU='0' if gpu is None else '1',
        NUMBA_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    if gpu is not None:env['SREGNAS_APPROVED_GPUS']=str(gpu)
    return env
