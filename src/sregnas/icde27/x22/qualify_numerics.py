
import os
from pathlib import Path

ENV='SREGNAS_X22_QUALIFY_NUMERICS'
_recorded=False

def enable():
    global _recorded
    if os.environ.get(ENV)!='1':return
    import torch
    import torch_geometric.typing as typing
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    typing.WITH_TORCH_SCATTER=False
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') in (':4096:8',':16:8')
    assert torch.are_deterministic_algorithms_enabled() and not typing.WITH_TORCH_SCATTER
    if not _recorded:
        from .common import write,now
        root=os.environ.get('SREGNAS_ICDE27_RECORD_DIR')
        if root:
            write(Path(root)/'x22_qualification'/f'numerics_{os.getpid()}.json',dict(at=now(),pid=os.getpid(),
                torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                pyg_WITH_TORCH_SCATTER=typing.WITH_TORCH_SCATTER,torch_num_threads=torch.get_num_threads(),
                CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                scope='untimed equality qualification only; paired reference and changed paths'))
        _recorded=True
