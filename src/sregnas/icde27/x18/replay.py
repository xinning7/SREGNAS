
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time
import traceback
from .common import ROOT,OUT,LOCK,read,write,VERSION


def worker(job,out):
    from sregnas.baselines.graphnap import common
    original=common.dataset_config
    def configured(*args,**kwargs):
        result=original(*args,**kwargs)
        if job.get('cached_sampler'):result['cached_sampler']=True
        return result
    common.dataset_config=configured

    if job['method']=='PAS-PL' and job.get('collect_search_garbage'):
        from sregnas.icde27.p2p6 import port_workers
        from sregnas.icde27.x15.observations import ObservedPreloadWorker
        class CollectedPASWorker(port_workers.Observation,ObservedPreloadWorker):
            def run_candidate(self,spec):
                self.start_profile(spec['fingerprint'])
                try:return super().run_candidate(spec)
                finally:self.finish_profile()
        port_workers.PASWorker=CollectedPASWorker
    if job.get('degree'):
        if job['method']=='SREGNAS' and os.environ.get('SREGNAS_X18_MEMORY_PREP')=='1':
            from .degree import install
            install(job)
        else:
            from sregnas.icde27.x10_degree import install
            install(job,out)
    from sregnas.icde27.x10 import configuration
    if job['method']=='SREGNAS':
        from sregnas.icde27.p2p6.canonical import execute
    elif job['method']=='FastGNAS':
        from sregnas.icde27.p2p6.native import execute
    else:
        from sregnas.icde27.p2p6.ports import execute
    result=execute(job,configuration(job),out)
    write(out/'measurement.json',result)
    return 0 if all(result.get('assertions',{}).values()) and all(r['status']=='OK' for r in result['rows']) else 2


def record(job,out,timeout):
    from sregnas.icde27 import recorder as rec
    from sregnas.icde27.x10 import configuration
    from sregnas.icde27.p2p6.entry import final_record,terminate
    from sregnas.icde27.p2p6.cleanup import wait_for_gpu_release
    from sregnas.icde27.x15.observations import augment_snapshot
    from sregnas.execution.telemetry import gpu_idle_preflight,_start_ticks
    from sregnas.comparison.launch import wait_gpu_idle_preflight
    from .common import environment
    code=Path(os.environ['SREGNAS_ROOT'])
    with LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        out.mkdir(parents=True,exist_ok=False);write(out/'job.json',job)
        cfg=configuration(job)
        argv=[str(Path(os.sys.executable)),'-m','sregnas.icde27.x18.replay','--worker','--job',str(out/'job.json'),'--out',str(out)]
        info=rec.metadata(cfg,job['method'],out,'X18',job.get('record_type','fixed_pool'),argv)
        info.update(job=job,cell=job,version=job.get('version',VERSION),cached_sampler=job.get('cached_sampler',False),timing_scope='exclusive_primary',
            physical_gpus=list(range(job.get('P',8))),monitored_gpus=list(range(8)),
            server_exclusive=True,global_lock_owner_pid=os.getpid(),gpu_lock=str(LOCK),
            Q_requested=job.get('count',32),K=0,native_rng_reset=True,controller_updates=0,process_started_epoch_s=time.time())
        if job.get('degree'):info.update(graph_hash=job['graph_sha256'],degree=job['degree'])
        known={};observations=[]
        def observe():
            r=rec.snapshot(list(range(8)),os.getpid(),known);augment_snapshot(r,os.getpid(),list(range(8)));return r
        baseline=wait_gpu_idle_preflight(list(range(8)),timeout_s=120,poll_s=2)
        before=observe();write(out/'resources_before.json',before);write(out/'release_baseline.json',baseline)
        if not baseline['idle'] or before['foreign_processes'] or before['cpu_contention_observed']:
            info.update(status='contaminated',timing_valid=False,reason='exclusive preflight failed');write(out/'run.json',info);return 2
        env=environment(code,job.get('variant'),job['dataset']);env['SREGNAS_ICDE27_RECORD_DIR']=str(out)
        env['SREGNAS_S0_PARALLEL']='1'
        if job.get('cost_pass'):
            env.update(SREGNAS_X18_COST_PROFILE=job['cost_pass'],
                       SREGNAS_X18_COST_METHOD=job['method'],SREGNAS_P2P6_PROFILE='0')
            info.update(cost_profile=job['cost_pass'],timing_scope='C1_profile_exclusive')
        if 'x19_flags' in job:
            from sregnas.icde27.x19.environment import apply
            apply(env,code,job['x19_flags'])
            info['x19_flags']=job['x19_flags']
        hit=False;started=time.perf_counter()
        with (out/'raw.log').open('w') as log:
            child=subprocess.Popen(argv,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            info['pipeline_process']=dict(pid=child.pid,pgid=child.pid,start_ticks=_start_ticks(child.pid),recorder_pid=os.getpid())
            write(out/'run.json',info)
            try:
                while child.poll() is None:
                    r=observe();observations.append(r)
                    with (out/'resources_during.jsonl').open('a') as f:
                        import json
                        f.write(json.dumps(r)+'\n')
                    if time.perf_counter()-started>=timeout:hit=True;terminate(child);break
                    time.sleep(.5)
            finally:
                if child.poll() is None:terminate(child)
        settled=wait_for_gpu_release(list(range(8)),owner_pid=os.getpid(),verified_owned=known,
            baseline=baseline,out=out,max_idle_drift_wait_s=120)
        after=observe();write(out/'resources_after.json',after);write(out/'release_settled.json',settled)
        info.update(status='timeout' if hit else 'OK' if child.returncode==0 else 'incomplete',returncode=child.returncode,
            timing_valid=not settled['foreign_process_observed'] and not any(
                r['foreign_processes'] or r['cpu_contention_observed'] or r['gpu']['returncode'] or r['compute']['returncode']
                for r in [before,*observations,after]),
            process_wall_s=time.perf_counter()-started,process_ended_epoch_s=time.time(),gpu_released_before_unlock=settled['idle'],
            host_peak_rss_GiB=max((r.get('host_rss_bytes',0) or 0 for r in observations),default=0)/2**30,
            per_gpu_peak_memory_GiB={str(g):max((r.get('gpu_memory_mib',{}).get(str(g),0) for r in observations),default=0)/1024 for g in range(8)})
        return final_record(out,info,job)


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--max-wall-seconds',type=float,default=3600)
    a=p.parse_args();job=read(a.job)
    try:return worker(job,a.out) if a.worker else record(job,a.out,a.max_wall_seconds)
    except Exception as exc:
        traceback.print_exc();a.out.mkdir(parents=True,exist_ok=True)
        write(a.out/'failure.json',dict(status='OOM' if 'memory' in str(exc).lower() else 'incomplete',error=str(exc)))
        return 2


if __name__=='__main__':raise SystemExit(main())
