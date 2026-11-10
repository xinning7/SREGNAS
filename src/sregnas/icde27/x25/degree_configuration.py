
import os

def apply(job,cfg):
    assert job.get('experiment_id')=='X25' and job.get('method')=='SREGNAS' and job.get('degree')
    assert os.environ.get('SREGNAS_PARTITION_MODE')=='proxy_aware'


    cfg.graph.partition_strategy='fi'
    cfg.runtime.require_fi_gpu=True
