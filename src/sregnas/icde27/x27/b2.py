
import os,sys,time,subprocess,json,contextlib
from pathlib import Path
from .common import ROOT,OUT,CODE,read,write,append,now,END

def worker(job,out):
    import torch
    from sregnas.icde27 import s0
    original=s0.span
    @contextlib.contextmanager
    def span(name,**kw):
        observe=name=='fixed_edge_materialization_gpu'
        if observe:
            torch.cuda.init();torch.cuda.reset_peak_memory_stats()
        def record(point):
            if torch.cuda.is_initialized():
                append(out/'allocator_preparation.jsonl',dict(at=now(),perf_s=time.perf_counter(),pid=os.getpid(),gpu=0,step=name,point=point,
                    allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),**kw))
        record('before')
        with original(name,**kw):yield
        record('after')
    s0.span=span
    from sregnas.icde27.x18.replay import worker as execute
    return execute(job,out)

def install_worker_observer():
    import torch
    from sregnas.training.fullbatch import PerLayerCanonicalModel
    original=PerLayerCanonicalModel.forward
    def forward(self,x,edges):
        if self.training:return original(self,x,edges)
        path=Path(os.environ['SREGNAS_ICDE27_RECORD_DIR'])/'allocator_validation.jsonl'
        before=dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        torch.cuda.reset_peak_memory_stats()
        result=original(self,x,edges)
        append(path,dict(at=now(),perf_s=time.perf_counter(),pid=os.getpid(),gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),phase='pool_validation',feature_shape=list(x.shape),layer_edges=[int(e.shape[1]) for e in edges],before_validation=before,
            after_validation=dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())))
        return result
    PerLayerCanonicalModel.forward=forward

def main():
    if '--worker' in sys.argv:
        p=Path(sys.argv[-1]);return worker(read(p/'job.json'),p)
    from sregnas.icde27.x25.common import environment
    from sregnas.icde27.x23.driver import gpu_processes
    from sregnas.icde27 import recorder
    from sregnas.icde27.x15.observations import augment_snapshot
    from sregnas.icde27.p2p6.entry import final_record
    from sregnas.icde27.x10 import configuration
    from .a_common import sources
    allresults=[]
    for ds in ('Reddit','Products'):
        while gpu_processes():
            if time.time()>END-1800:
                write(OUT/'b2_finished.json',dict(at=now(),runs=allresults,status='skipped',reason='no empty eight-card B2 mapping before reporting margin'));return 0
            time.sleep(5)
        src=next(r for r in sources() if r['scope']=='pool' and r['dataset']==ds and r['method']=='SREGNAS')
        job=read(Path(src['source']).parent/'job.json');name=f'expX27_B2_SREGNAS_{ds}_Q32_untimed'
        job.update(experiment_id='X27',version='x27',cell_id=name,category='B2')
        p=OUT/'runs'/name;p.mkdir(parents=True,exist_ok=False);write(p/'job.json',job)
        env=environment(CODE,'V2',ds,json.loads(src['flags']));env.update(NUMBA_CACHE_DIR=str(OUT/'numba_cache'),SREGNAS_ICDE27_RECORD_DIR=str(p),SREGNAS_S0_PARALLEL='1',SREGNAS_X27_MEMORY_AUDIT='1')
        cmd=[sys.executable,'-m','sregnas.icde27.x27.b2','--worker',str(p)]
        info=recorder.metadata(configuration(job),'SREGNAS',p,'X27','profile',cmd)
        known={};snap=[]
        with (p/'raw.log').open('x') as log:
            child=subprocess.Popen(cmd,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                sample=recorder.snapshot(list(range(8)),os.getpid(),known);augment_snapshot(sample,os.getpid(),list(range(8)));append(p/'resources_during.jsonl',sample);snap.append(sample);time.sleep(.25)
        info.update(status='OK' if child.returncode==0 else 'incomplete',returncode=child.returncode,timing_valid=False,resource_class='shared_empty_gpu_untimed',timing_scope='quality_only',global_lock_held=False,physical_gpus=list(range(8)),reason='B2 memory attribution only',job=job)
        final_record(p,info,job)
        allresults.append(dict(run_id=name,out=str(p),status=info['status']))
    write(OUT/'b2_finished.json',dict(at=now(),runs=allresults))
    return 0
if __name__=='__main__':raise SystemExit(main())
