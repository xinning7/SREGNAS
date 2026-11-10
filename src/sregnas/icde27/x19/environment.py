
import os
from pathlib import Path

FLAGS=dict(gpu_csr='SREGNAS_X19_GPU_CSR',gpu_materialize='SREGNAS_X19_GPU_MATERIALIZE',
           prebuild_batches='SREGNAS_X19_PREBUILD_BATCHES')


def apply(env,code,flags=None,profile=False):
    flags=flags or {}
    for name,key in FLAGS.items():
        env[key]='1' if flags.get(name,False) else '0'
    env['SREGNAS_X19_FINE_PROFILE']='1' if profile else '0'
    env['SREGNAS_X19_BOOTSTRAP']='1'
    bootstrap=str(Path(code)/'src/sregnas/icde27/x19/bootstrap')
    env['PYTHONPATH']=bootstrap+os.pathsep+str(Path(code)/'src')
    return env
