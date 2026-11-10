from sregnas.paths import PROJECT_ROOT as _artifact_root, GPU_LOCK as _artifact_lock
import csv
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time

ROOT=Path(str(_artifact_root))
OUT=ROOT/'results/icde27/x17'
PYTHON=__import__("sys").executable
LOCK=Path(str(_artifact_lock))
START=1793630817.0
EXCLUSIVE_END=START+36000
VERSION='pi_softmax_max_v1'
DATASETS=('Flickr','arXiv','Reddit','Products')
SOFTMAX={'gat','sym-gat','cos','linear','gene-linear','generalized_linear'}


def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def rows(path,delimiter=','):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f,delimiter=delimiter))


def table(path,data,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(dict.fromkeys(k for r in data for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fields,lineterminator='\n',extrasaction='ignore');w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in data)
    assert len(rows(path))==len(data)


def cpu_only():
    os.environ.update(CUDA_VISIBLE_DEVICES='',SREGNAS_ALLOW_GPU='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    import torch
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)


def mdtable(data,fields):
    def fmt(v):return (f'{v:.2f}' if isinstance(v,float) else str(v if v is not None else '')).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(fields)+' |','| '+' | '.join('---' for _ in fields)+' |']+['| '+' | '.join(fmt(r.get(k,'')) for k in fields)+' |' for r in data])


def section(title,body):
    p=ROOT/'docs/audit/x17_report.md';text=p.read_text();head,old=text.split('## '+title+'\n',1)
    tail='\n## '+old.split('\n## ',1)[1] if '\n## ' in old else ''
    p.write_text(head+'## '+title+'\n\n'+body.rstrip()+'\n'+tail)


def affected(architecture,backend='canonical'):
    from sregnas.icde27.space import canonical_layers
    return backend not in ('fastgnas-native','native') and any(l['encoding_op'] in SOFTMAX and l['aggregation_op']=='max' for l in canonical_layers(architecture))
