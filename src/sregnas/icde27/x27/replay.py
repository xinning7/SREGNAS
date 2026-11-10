
from sregnas.paths import PROJECT_ROOT as _artifact_root
import os
from pathlib import Path
from sregnas.icde27.x25.common import environment as base_environment
import sregnas.icde27.x18.common as common
FLAGS={k:v for k,v in os.environ.items() if k.startswith(('SREGNAS_X19_','SREGNAS_X20_','SREGNAS_X22_','SREGNAS_X23_','SREGNAS_X25_')) or k=='SREGNAS_PARTITION_MODE'}
def environment(code,variant=None,dataset=None):
    env=base_environment(code,variant,dataset,FLAGS)
    env['NUMBA_CACHE_DIR']=str(_artifact_root / 'results/icde27/x27/numba_cache')
    return env
common.environment=environment
from sregnas.icde27 import recorder
_metadata=recorder.metadata
def metadata(cfg,method,out,experiment_id,record_type,argv):
    return _metadata(cfg,method,out,'X27',record_type,argv)
recorder.metadata=metadata
from sregnas.icde27.x18.replay import main
if __name__=='__main__':raise SystemExit(main())
