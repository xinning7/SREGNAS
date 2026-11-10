from sregnas.paths import PROJECT_ROOT as _artifact_root, GPU_LOCK as _artifact_lock
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path

ROOT = Path(str(_artifact_root))
LEGACY_OUT = ROOT / 'results/icde27/x18'
OUT = LEGACY_OUT / 'v2'
PYTHON = __import__("sys").executable
LOCK = Path(str(_artifact_lock))
START = datetime(2026,11,3,6,0,16,tzinfo=timezone.utc).timestamp()
VERSION = 'x18_v2'
DATASETS = ('Flickr','arXiv','Reddit','Products')
FI = dict(zip(DATASETS,('metis','metis','random','random')))
if (OUT/'fi_choices.json').exists():FI.update(json.loads((OUT/'fi_choices.json').read_text()))


def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def rows(path,delimiter=','):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f,delimiter=delimiter))


def table(path,data,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(dict.fromkeys(k for r in data for k in r)) or ['version','status']
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fields,lineterminator='\n',extrasaction='ignore');writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in data)
    assert len(rows(path))==len(data)


def jsonrows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


def environment(code,variant=None,dataset=None):
    env=dict(os.environ,PYTHONPATH=str(code/'src'),SREGNAS_ROOT=str(code),
        SREGNAS_DATA_DIR=os.environ.get('SREGNAS_DATA_DIR', str(ROOT / 'data')),SREGNAS_ALLOW_GPU='1',
        SREGNAS_APPROVED_GPUS='0,1,2,3,4,5,6,7',SREGNAS_CPU_THREADS='1',
        SREGNAS_CANONICAL_NORMALIZATION_FIX='1',SREGNAS_SAMPLED_GRAD_CLIP='1.0',
        SREGNAS_PI_SOFTMAX_MAX_FIX='1',SREGNAS_ICDE27_TIE_RULE='1',SREGNAS_ICDE27_NATIVE_RNG='1',
        SREGNAS_X18_VERSION=VERSION,
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',SREGNAS_X17_E2E_CAP='1')
    for key in ('CUDA_VISIBLE_DEVICES','RAY_ADDRESS','SREGNAS_ICDE27_RECORD_DIR','SREGNAS_RUN_COMMIT',
                'SREGNAS_PROXY_STEPS','SREGNAS_PROXY_BATCH_TARGETS','SREGNAS_PROXY_AP_TARGETS','SREGNAS_S0_PARALLEL',
                'SREGNAS_QUALITY_ONLY','SREGNAS_X_QUEUE_AMENDMENT','SREGNAS_X16_LOCK_FD',
                'SREGNAS_ICDE27_CAMPAIGN_DEADLINE_EPOCH','SREGNAS_X18_CSR_KEY'):
        env.pop(key,None)
    env.pop('SREGNAS_X18_PARALLEL_PERMUTATIONS',None)
    env.pop('SREGNAS_X18_FAST_CSR',None)
    env.pop('SREGNAS_X18_MEMORY_PREP',None)
    for key in list(env):
        if key.startswith('SREGNAS_X19_'):env.pop(key)
    if variant:
        env['SREGNAS_X18_MEMORY_PREP']='1'
        e,b=configuration(variant,dataset)
        env['SREGNAS_PROXY_STEPS']=str(e)
        if b is not None:
            env['SREGNAS_PROXY_BATCH_TARGETS']=str(b)
            env['SREGNAS_PROXY_AP_TARGETS']='1'
    return env


def configuration(variant,dataset):
    return {'V0':(100,None),'V1':(50,None),'V2':(50,256 if dataset=='Products' else 512),
            'V3':(50,2048),'V4':(25,256 if dataset=='Products' else 512)}[variant]


def mdtable(data,fields):
    def fmt(v):
        if isinstance(v,str):
            import re
            if re.fullmatch(r'-?\d+\.\d+(?:[eE][+-]?\d+)?',v):v=float(v)
        return (f'{v:.2f}' if isinstance(v,float) else str(v if v is not None else '')).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(fields)+' |','| '+' | '.join('---' for _ in fields)+' |']+
        ['| '+' | '.join(fmt(r.get(k,'')) for k in fields)+' |' for r in data])


def section(title,body):
    p=ROOT/'docs/audit/x18_report.md';text=p.read_text();head,old=text.split('## '+title+'\n',1)
    tail='\n## '+old.split('\n## ',1)[1] if '\n## ' in old else ''
    p.write_text(head+'## '+title+'\n\n'+body.rstrip()+'\n'+tail)
