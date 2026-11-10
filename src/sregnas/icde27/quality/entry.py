
from sregnas.paths import PROJECT_ROOT as _artifact_root
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from sregnas.paths import PROJECT_ROOT as ROOT
from sregnas.icde27 import recorder as shared
from sregnas.icde27.p2p6.entry import terminate
from sregnas.icde27.p2p6.resources import is_oom


def configuration(job):
    from sregnas.config import load_config
    cfg=load_config(job['config'])
    cfg.runtime.system_gpus=cfg.runtime.baseline_gpus=job['gpus']
    if job['kind'] in ('post','reference','winner_replica','x3','x_capacity','x5_diagnostic','x8_check'):
        cfg.runtime.require_fi_gpu=False
    cfg.validate()
    return cfg


def record(job,out,timeout,gpu_lock):
    if out.exists(): raise FileExistsError(out)
    with Path(gpu_lock or shared.LOCK).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        out.mkdir(parents=True)
        shared.dump(out/'job.json',job)
        cfg=configuration(job)
        worker_module='sregnas.icde27.x9_worker' if job.get('x9_queue') else 'sregnas.icde27.quality.entry'
        command=[sys.executable,'-m',worker_module,'--job',str(out/'job.json'),'--out',str(out),'--worker']
        info=shared.metadata(cfg,job['method'],out,str(job['experiment_id']),job.get('record_type','reference'),command)
        if job['kind']=='x3':info['backend']='canonical'
        if 'canonical_nf' in job:
            from sregnas.models.normalization_fix import SCHEMA
            info['normalization_fix'] = bool(job['canonical_nf'])
            info['normalization_fix_schema'] = SCHEMA if job['canonical_nf'] else 'legacy'
            info['normalization_fix_sha256'] = shared.sha(ROOT/'src/sregnas/models/normalization_fix.py')
        if job.get('deterministic_diagnostic'):
            info['deterministic_diagnostic'] = True
        info.update(timing_valid=False,timing_scope='quality_only',resource_class='quality_gpu_group',
                    gpu_lock=str(gpu_lock or shared.LOCK),job=job,Q_requested=0,K=job.get('K',1),physical_gpus=job['gpus'])
        identities={};before=shared.snapshot(job['gpus'],os.getpid(),identities)
        shared.dump(out/'resources_before.json',before);shared.dump(out/'run.json',info)
        if before['foreign_processes']:
            info.update(status='contaminated',blocking_reason='GPU group occupied')
            shared.dump(out/'run.json',info);return 2
        from sregnas.execution.telemetry import gpu_idle_preflight
        release_baseline=gpu_idle_preflight(job['gpus'],owner_pid=os.getpid(),verified_owned=identities)
        shared.dump(out/'release_baseline.json',release_baseline)
        env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),SREGNAS_ICDE27_RECORD_DIR=str(out),
            SREGNAS_ALLOW_GPU='1',SREGNAS_APPROVED_GPUS=','.join(map(str,job['gpus'])),
            SREGNAS_DATA_DIR=os.environ.get('SREGNAS_DATA_DIR', str(_artifact_root / 'data')),OMP_NUM_THREADS='4',
            MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        env.pop('CUDA_VISIBLE_DEVICES',None)
        if 'canonical_nf' in job:
            env['SREGNAS_CANONICAL_NORMALIZATION_FIX'] = '1' if job['canonical_nf'] else '0'
        if job.get('deterministic_diagnostic'):
            env.update(SREGNAS_X8_DETERMINISTIC_DIAGNOSTIC='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        started=time.time();hit=False
        info.update(process_started_epoch_s=started,max_wall_seconds=timeout)
        with (out/'raw.log').open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            from sregnas.execution.telemetry import _start_ticks
            info['pipeline_process']=dict(pid=child.pid,pgid=child.pid,start_ticks=_start_ticks(child.pid),recorder_pid=os.getpid())
            shared.dump(out/'run.json',info)
            try:
                while child.poll() is None:
                    if time.time()-started>=timeout: hit=True;terminate(child);break
                    observation=shared.snapshot(job['gpus'],os.getpid(),identities)
                    with (out/'resources_during.jsonl').open('a') as stream:stream.write(json.dumps(observation)+'\n')
                    time.sleep(2)
            finally:
                if child.poll() is None:terminate(child)
        info.update(status='incomplete' if hit or child.returncode else 'OK',returncode=child.returncode,
                    process_ended_epoch_s=time.time(),process_wall_s=time.time()-started)
        if hit:info['stop_reason']=job.get('timeout_reason','over_twice_estimate')
        measurement=out/'measurement.json'
        if measurement.exists():info.update(json.loads(measurement.read_text()))
        failure=out/'failure.json'
        if failure.exists():info.update(failure=json.loads(failure.read_text()));info['status']=info['failure']['status']
        from sregnas.icde27.p2p6.cleanup import wait_for_gpu_release
        settled=wait_for_gpu_release(job['gpus'],owner_pid=os.getpid(),verified_owned=identities,
                                    baseline=release_baseline,out=out,
                                    max_idle_drift_wait_s=120 if job['kind'] in ('x3','x_capacity','x5_diagnostic') or job.get('x567_queue') or job.get('x8_queue') or job.get('x9_queue') else None)
        shared.dump(out/'release_settled.json',settled)
        info['assertions']['gpu_released_before_unlock']=settled['idle']
        if settled.get('settling_decision')=='FAIL_IDLE_MEMORY_BASELINE_DRIFT':
            info.update(status='incomplete',stop_reason='tool_cleanup_idle_memory_baseline_drift')
        if info.get('reference_result',{}).get('status')=='OOM':info['status']='OOM'
        if not all(info['assertions'].values()) and info['status']=='OK':info['status']='incomplete'
        shared.dump(out/'resources_after.json',shared.snapshot(job['gpus'],os.getpid(),identities))
        shared.dump(out/'run.json',info)
        passed=info['status']=='OK' and bool(info['assertions']) and all(info['assertions'].values())
        (out/'summary.md').write_text(f"run_id: {out.name}\n\nstatus: {info['status']}\n\ntiming_valid: false\n\ntiming_scope: quality_only\n\nASSERTION_PASS: {str(passed).lower()}\n\n"+'\n'.join(f'- {k}: {v}' for k,v in info['assertions'].items())+'\n')
        for name in ('candidates','shortlist','retrain'):
            if not (out/(name+'.tsv')).exists():shared.write_table(out/(name+'.tsv'),[],['run_id','fingerprint','status'])
        return 0 if passed else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--max-wall-seconds',type=float,default=3600);parser.add_argument('--gpu-lock',type=Path)
    parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    job=json.loads(args.job.read_text());out=args.out.resolve()
    if not args.worker:return record(job,out,args.max_wall_seconds,args.gpu_lock)
    if not os.environ.get('SREGNAS_ICDE27_RECORD_DIR'):raise RuntimeError('recorder required')
    info=json.loads((out/'run.json').read_text())
    if job.get('deterministic_diagnostic'):
        import torch
        torch.use_deterministic_algorithms(True)
    try:
        from .post import execute, reference
        if job['kind']=='x8_check':
            from sregnas.icde27.x8_checks import execute as run_x8_check
            result=run_x8_check(job,out,info)
        elif job['kind']=='x5_diagnostic':
            from sregnas.icde27.x567_gpu import execute as run_diagnostic
            result=run_diagnostic(job,out,info)
        elif job['kind']=='x_capacity':
            from sregnas.icde27.x_capacity import execute as run_capacity
            result=run_capacity(job,out,info)
        elif job['kind']=='x3':
            from sregnas.icde27.x3 import execute as run_x3
            result=run_x3(job,out,info)
        elif job['kind']=='winner_replica':
            from .replicas import execute as repeat_winner
            result=repeat_winner(job,out,info)
        else:
            result=(reference if job['kind']=='reference' else execute)(job,out,info)
        shared.dump(out/'measurement.json',result)
        return 0 if all(result['assertions'].values()) else 2
    except BaseException as exc:
        traceback.print_exc()
        partial=json.loads((out/'partial_retrains.json').read_text()) if (out/'partial_retrains.json').exists() else []
        failed=[r['status'] for r in partial if r['status']!='OK']
        shared.dump(out/'failure.json',dict(status='OOM' if is_oom(str(exc)) or (failed and all(s=='OOM' for s in failed)) else 'incomplete',
            exception=type(exc).__name__,error=str(exc)))
        return 2

if __name__=='__main__':raise SystemExit(main())
