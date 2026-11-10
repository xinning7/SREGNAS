
import os
from .common import environment as base_environment
import sregnas.icde27.x18.common as common

FLAGS = {k:v for k,v in os.environ.items() if k.startswith(('SREGNAS_X19_','SREGNAS_X20_','SREGNAS_X22_','SREGNAS_X23_'))}
WORKSPACE = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
def environment(code,variant=None,dataset=None):
    env=base_environment(code,variant,dataset,FLAGS)
    if WORKSPACE:env['CUBLAS_WORKSPACE_CONFIG']=WORKSPACE
    return env
common.environment=environment

from sregnas.icde27 import recorder
_metadata=recorder.metadata
def metadata(cfg,method,out,experiment_id,record_type,argv):
    return _metadata(cfg,method,out,'X23',record_type,argv)
recorder.metadata=metadata

from sregnas.icde27.p2p6 import entry
_final_record=entry.final_record
def final_record(out,info,cell):
    if os.environ.get('SREGNAS_X20_DETERMINISTIC')=='1':
        if info.get('status')=='OK' and not info.get('timing_valid'):info['status']='contaminated'
        info.update(timing_valid=False,timing_scope='quality_only',qualification_scope='deterministic_fixed16')
    return _final_record(out,info,cell)
entry.final_record=final_record

from sregnas.icde27.x18.replay import main
if __name__=='__main__':raise SystemExit(main())
