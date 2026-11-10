
from sregnas.paths import PROJECT_ROOT as _artifact_root, GPU_LOCK as _artifact_lock
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from sregnas.icde27.x18.common import environment as _x18_environment

ROOT = Path(str(_artifact_root))
OUT = ROOT / 'results/icde27/x22'
PYTHON = __import__("sys").executable
LOCK = Path(str(_artifact_lock))
T0 = datetime.fromisoformat('2026-11-04T16:23:22+00:00').timestamp()
GPU_END = T0 + 8.5 * 3600
END = T0 + 9 * 3600
X20_FLAGS = {k: '1' for k in ('SREGNAS_X19_GPU_CSR', 'SREGNAS_X19_GPU_MATERIALIZE',
    'SREGNAS_X19_PREBUILD_BATCHES', 'SREGNAS_X20_ORIGIN_PREFIX_CACHE')}

def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); tmp.replace(path)

def environment(code, variant=None, dataset=None, flags=None):


    env = _x18_environment(Path(code), variant, dataset)
    for k in list(env):
        if k.startswith(('SREGNAS_X19_', 'SREGNAS_X20_', 'SREGNAS_X22_')): env.pop(k)
    for k in ('CUBLAS_WORKSPACE_CONFIG', 'CUDA_LAUNCH_BLOCKING', 'PYTORCH_DETERMINISTIC',
              'NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'): env.pop(k, None)
    env.update(flags or {})
    env['PYTHONPATH'] = str(Path(code) / 'src')
    return env
