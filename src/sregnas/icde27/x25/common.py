
from sregnas.paths import PROJECT_ROOT as _artifact_root, GPU_LOCK as _artifact_lock
from datetime import datetime, timezone
from pathlib import Path
import os
from sregnas.icde27.x23.common import read, write, now, environment as previous_environment
from sregnas.icde27.x23.common import BASE_FLAGS
from sregnas.icde27.x23.decide import NEW_FLAGS

ROOT = Path(str(_artifact_root))
OUT = ROOT/'results/icde27/x25'
CODE = ROOT/'.runtime/x25-code/release'
PYTHON = __import__("sys").executable
LOCK = Path(str(_artifact_lock))
T0 = datetime.fromisoformat('2026-11-05T10:33:49+00:00').timestamp()
QUAL_END = T0+1.5*3600
ATTENDED_END = T0+6*3600
GPU_END = T0+11*3600
REVIEW_START = T0+11.5*3600
END = T0+14*3600
FLAGS = dict(BASE_FLAGS, **NEW_FLAGS)
DATASETS = ('Reddit','Products','arXiv','Flickr')
KEYS = dict(Reddit='reddit',Products='ogbn-products',arXiv='ogbn-arxiv',Flickr='flickr')
PI_STAR = dict(Reddit='da_random',Products='da_random',arXiv='la_metis',Flickr='la_metis')

def environment(code,variant=None,dataset=None,flags=None):
    env=previous_environment(code,variant,dataset,FLAGS)
    for k in list(env):
        if k.startswith('SREGNAS_X25_') or k in ('SREGNAS_PARTITION_MODE','NUMBA_CACHE_DIR','NUMBA_DEBUG_CACHE'):
            env.pop(k,None)
    env.update(NUMBA_CACHE_DIR=str(OUT/'numba_cache'),NUMBA_NUM_THREADS='1')
    env.update(flags or {})
    return env
