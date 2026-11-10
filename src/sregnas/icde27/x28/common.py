
from sregnas.paths import PROJECT_ROOT as _artifact_root
from datetime import datetime
from pathlib import Path
from sregnas.icde27.x27.common import read, write, rows, table, sha, now, append

ROOT = Path(str(_artifact_root))
OUT = ROOT / 'results/icde27/x28'
PYTHON = __import__("sys").executable
T0 = datetime.fromisoformat('2026-11-06T08:01:33+00:00').timestamp()
END = T0 + 3.5 * 3600
A_END = T0 + 2 * 3600
DATASETS = ('Flickr', 'arXiv', 'Reddit', 'Products')
PI_STAR = dict(Flickr='la_metis', arXiv='la_metis', Reddit='da_random', Products='da_random')

def environment(gpu=None, variant=None, dataset=None):
    from sregnas.icde27.x25.common import environment as base, FLAGS
    env = base(ROOT, variant, dataset, FLAGS)
    env.update(NUMBA_CACHE_DIR=str(OUT/'numba_cache'), NUMBA_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
               SREGNAS_ALLOW_GPU='0' if gpu is None else '1', SREGNAS_CPU_THREADS='1')


    env['SREGNAS_X25_PARTITION_RECORDS']='1'
    if gpu is not None: env['SREGNAS_APPROVED_GPUS'] = str(gpu)
    return env

def resolve_graph(path):
    pass
    import hashlib
    path = Path(path)
    if path.suffix != '.json':
        assert path.exists(), path
        return str(path)
    changes = []
    def walk(value):
        if isinstance(value, dict): return {k:walk(v) for k,v in value.items()}
        if isinstance(value, list): return [walk(v) for v in value]
        if isinstance(value, str) and value.startswith('/dev/shm/') and value.endswith('.npy') and not Path(value).exists():
            old = Path(value)
            for name in ('x28', 'x25', 'x23', 'x27'):
                dest = ROOT/'results/icde27'/name/'shared_memory_archive'/old.parent.name/old.name
                if dest.exists():
                    changes.append(dict(original=value, resolved=str(dest)))
                    return str(dest)
            raise FileNotFoundError(value)
        return value
    value = walk(read(path))
    if not changes: return str(path)
    dest = OUT/'resolved_graphs'/hashlib.sha256(str(path).encode()).hexdigest()[:20]/path.name
    write(dest, value)
    write(dest.with_suffix('.relocation.json'), dict(source=str(path), source_sha256=sha(path), changes=changes))
    return str(dest)

def retraining_environment(gpu,spec):
    pass
    env=environment(gpu)
    receipt=spec.get('partition_receipt')
    if receipt:
        choice=read(receipt['path']).get('partition_selection',{})
        if choice.get('partition_mode')=='proxy_aware':env['SREGNAS_PARTITION_MODE']='proxy_aware'
        arm=choice.get('forced_quality_arm')
        if arm:
            assert arm in ('da_random','la_metis')
            env.update(SREGNAS_PARTITION_MODE='proxy_aware',SREGNAS_X25_PARTITION_ARM=arm)
    return env
