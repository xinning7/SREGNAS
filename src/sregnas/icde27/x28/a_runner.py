
import argparse
import os
import subprocess
import time
from .common import *

def configuration(job):
    from sregnas.icde27.x10 import configuration as original


    return original(dict(job,P=8) if job.get('smoke') else job)

def x27c_active():
    matches=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:args=path.read_bytes().decode().split('\0')
        except (OSError,UnicodeDecodeError):continue
        if args and 'python' in Path(args[0]).name and any(a.endswith('/x27/x27c/campaign.py') for a in args[1:]) and 'run' in args:
            matches.append(int(path.parent.name))
    return matches

def make_job(dataset,global_=False,rebuild=False,smoke=False,round_=1):
    row=next(r for r in rows(ROOT/'results/icde27/x25/table_v_fixed_pool.csv') if r['dataset']==dataset and r['method']=='SREGNAS')
    source=Path(row['source']).parent;job=read(source/'job.json')
    name=f'expX28_{"A" if smoke else "C"}_{"Global" if global_ else "Local"}_{"rebuilt" if rebuild else "cached"}_{dataset}_E1_P{1 if smoke else 8}S8_Q{8 if smoke else 32}K0_seed42_fixed_pool_r{round_}'
    job.update(experiment_id='X28',version='x28',cell_id=name,P=1 if smoke else 8,count=8 if smoke else 32,
        domain='local',cache_mode='cached',x28_fixed_pool=True,x28_global=global_,x28_rebuild=rebuild,
        physical_gpus=[5] if smoke else list(range(8)),smoke=smoke,round=round_,x25_source=str(source),x25_source_run_id=row['run_id'])
    return job

def run(job):
    from sregnas.icde27 import recorder
    from sregnas.icde27.p2p6.entry import final_record
    from sregnas.icde27.x16.resources import inventory
    from sregnas.icde27.x15.observations import augment_snapshot
    p=OUT/'runs'/job['cell_id'];p.mkdir(parents=True,exist_ok=False);write(p/'job.json',job)
    gpus=job['physical_gpus'];smoke=job['smoke']
    cards=inventory();selected=[c for c in cards if c['gpu'] in gpus]
    assert len(selected)==len(gpus) and all(not c['processes'] for c in selected),'nonempty card admission'
    assert not x27c_active(),'external X27C campaign still active'
    if not smoke:
        assert os.environ.get('SREGNAS_X28_LOCKED')=='1','timing requires caller-held global lock'
    env=environment(gpus[0],'V2',job['dataset']);env.update(SREGNAS_ICDE27_RECORD_DIR=str(p),
        SREGNAS_X28_FIXED_POOL='1',SREGNAS_X28_GLOBAL=str(int(job['x28_global'])),
        SREGNAS_X28_REBUILD=str(int(job['x28_rebuild'])),SREGNAS_PARTITION_MODE='proxy_aware',
        SREGNAS_X25_PARTITION_RECORDS='1',SREGNAS_S0_PARALLEL='1')
    env['SREGNAS_APPROVED_GPUS']=','.join(map(str,gpus))
    commit=__import__('sregnas.execution.provenance', fromlist=['commit_hash']).commit_hash()
    if commit is not None:env['SREGNAS_RUN_COMMIT']=commit
    cmd=[PYTHON,'-m','sregnas.icde27.x28.a_runner','--worker',str(p)]
    from unittest.mock import patch
    with patch.dict(os.environ,env,clear=True):
        info=recorder.metadata(configuration(job),'SREGNAS',p,'X28','fixed_pool',cmd)
    known={};samples=[]
    def observe():
        sample=recorder.snapshot(gpus,os.getpid(),known);augment_snapshot(sample,os.getpid(),gpus)
        return sample
    before=observe();write(p/'resources_before.json',before)
    write(p/'run.json',dict(info,status='incomplete',job=job,started_at=now()))
    with (p/'raw.log').open('w') as log:
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            sample=observe();samples.append(sample);append(p/'resources_during.jsonl',sample);time.sleep(.2)
    after=observe();write(p/'resources_after.json',after)
    observations=[before,*samples,after]
    foreign=any(s['foreign_processes'] for s in observations)
    cpu_contention=any(s.get('cpu_contention_observed',False) for s in observations)
    telemetry_ok=all(s['gpu']['returncode']==0 and s['compute']['returncode']==0 and s.get('gpu_memory_query_returncode')==0 for s in observations)
    contention=foreign or cpu_contention
    status='OK' if child.returncode==0 else 'incomplete'
    if not smoke and contention:status='contaminated'
    elif not smoke and not telemetry_ok:status='unsupported'
    info.update(status=status,returncode=child.returncode,
        timing_valid=not smoke and not contention and telemetry_ok,timing_scope='quality_only' if smoke else 'exclusive_primary',
        resource_class='shared_initially_empty_gpu' if smoke else 'exclusive_A6000_8_reserved',
        global_lock_held=not smoke,physical_gpus=gpus,job=job,commit=commit,started_before=before.get('epoch_s'),
        finished_at=now(),contaminated=contention,external_processes_observed=foreign,
        cpu_contention_observed=cpu_contention,telemetry_valid=telemetry_ok,
        seed_rule=dict(info.get('seed_rule',{}),rebuild='training.minibatch._hash_seed(42,"x28_rebuild",candidate_index)+layer'),
        x28_global=job['x28_global'],x28_rebuild=job['x28_rebuild'],Q_requested=job['count'],P=job['P'],S=8)

    with patch.dict(os.environ,env,clear=True):
        final_record(p,info,job)
    return read(p/'run.json')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=Path);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--global',dest='global_',action='store_true');parser.add_argument('--rebuild',action='store_true')
    parser.add_argument('--dataset',default='arXiv',choices=DATASETS);parser.add_argument('--round',type=int,default=1)
    a=parser.parse_args()
    if a.worker:
        from .ablation import execute
        job=read(a.worker/'job.json');result=execute(job,configuration(job),a.worker);write(a.worker/'measurement.json',result)
        return 0 if all(result['assertions'].values()) and all(r['status']=='OK' for r in result['rows']) else 2
    if not a.smoke:raise ValueError('Use scripts/run.py for a recorded locality workload with the shared lock')
    return 0 if run(make_job(a.dataset,a.global_,a.rebuild,True,a.round))['status']=='OK' else 2

if __name__=='__main__':raise SystemExit(main())
