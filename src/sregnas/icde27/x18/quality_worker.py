
import argparse
from dataclasses import asdict
import gc
import os
from pathlib import Path
import time
import traceback
from .common import read,write,VERSION,OUT


def artifacts(path,info,spec):
    import csv
    fields=['run_id','fingerprint','replica','status','val_acc','test_acc','checkpoint_path','version']
    row=dict(info,checkpoint_path=spec.get('checkpoint_path'))
    for name in ('candidates','shortlist','retrain'):
        with (path/(name+'.tsv')).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fields,delimiter='\t',extrasaction='ignore',lineterminator='\n')
            writer.writeheader();writer.writerow(row)
    (path/'summary.md').write_text(f"run_id: {info['run_id']}\n\nstatus: {info['status']}\n\n"
        f"checkpoint: {spec.get('checkpoint_path')}\n\nfinished: {info['finished']}\n\n"
        f"ASSERTION_PASS: {str(info['finished'] and info['status']=='OK').lower()}\n")
    raw=path/'raw.log'
    if not raw.exists():raw.symlink_to(OUT/'logs'/f"{info['part']}_gpu{info['gpu']}.log")


def run(job,out):
    import torch
    from sregnas.execution.startup import assert_cell_start
    from sregnas.execution.provenance import freeze_run_commit
    from sregnas.training.fullbatch import run_cell,PersistentCellRuntime
    from sregnas.training.minibatch import run_products_cell,ProductsRuntimeCache
    from sregnas.icde27.ledger import call
    from sregnas.icde27.x16.resources import inventory,eligible
    torch.set_num_threads(1);device=torch.device('cuda:0');torch.cuda.set_device(device)
    assert os.environ.get('SREGNAS_PI_SOFTMAX_MAX_FIX')=='1'
    commit=freeze_run_commit();out.mkdir(parents=True,exist_ok=True)
    full,sampled=PersistentCellRuntime(),ProductsRuntimeCache();last_graph=None;completed=[]
    for task in job['tasks']:
        if time.time()>=job['deadline_epoch_s'] or Path(job['stop_file']).exists():break
        card=next(c for c in inventory() if c['gpu']==job['gpu'])
        foreign=[p for p in card['processes'] if p['pid']!=os.getpid()]
        if foreign:
            own=sum(float(p['used_mib'])/1024 for p in card['processes'] if p['pid']==os.getpid() and str(p['used_mib']).replace('.','',1).isdigit())
            if not eligible(dict(card,processes=foreign,free_gib=card['free_gib']+own),job['admission_peak_gib'],empty=job.get('require_empty',False)):
                write(out/'admission_boundary_stop.json',dict(epoch_s=time.time(),card=card));break
        spec=dict(task['spec'],gpu=job['gpu']);path=out/(task['task_id']+'_'+VERSION)
        if (path/'run.json').exists() and read(path/'run.json').get('finished'):
            completed.append(read(path/'run.json'));continue
        path.mkdir(parents=True,exist_ok=True)
        if not task.get('test_only'):
            spec['checkpoint_path']=str(path/'checkpoints'/f'{spec["fingerprint"]}_r{spec["replica"]}_{VERSION}.pt')
        if spec['graph_path']!=last_graph:

            sampled=ProductsRuntimeCache();full=PersistentCellRuntime();gc.collect();torch.cuda.empty_cache()
            last_graph=spec['graph_path']
        write(path/'spec.json',spec);os.environ['SREGNAS_ICDE27_RECORD_DIR']=str(path)
        info=dict(task.get('metadata',{}),run_id=f'X18_{job["part"]}_{task["task_id"]}_{VERSION}',experiment_id='X18',record_type='reference',
            dataset=job['dataset'],part=job['part'],fingerprint=spec['fingerprint'],replica=spec['replica'],
            gpu=job['gpu'],status='incomplete',finished=False,timing_valid=False,commit=commit,
            started_epoch_s=time.time(),architecture=spec['architecture'],method='canonical_Pi',P=1,S=1,K=1,
            source_run_id=task['source_run_id'],version=VERSION,task_id=task['task_id'],
            pi_softmax_max_fix=True,normalization_fix=True,sampled_grad_clip=1.0,
            global_lock_held=job.get('global_lock_held',False),resource_class='quality_gpu_group',
            key_hash=task.get('key_hash'))
        write(path/'run.json',info)
        try:
            startup=assert_cell_start(spec)
            def execute(one):
                if one['dataset'] in ('reddit','ogbn-products'):
                    return asdict(call(one,run_products_cell,device=device,runtime=sampled))
                return asdict(call(one,run_cell,runtime=full))
            started=time.perf_counter()
            trained=(dict(status='OK',value=task['old_val'],reused=True) if task.get('test_only') else execute(spec))
            info.update(status=trained['status'],training=trained,train_validation_wall_s=time.perf_counter()-started,
                val_acc=trained.get('value'),cell_start=startup)
            write(path/'run.json',info)
            if trained['status']=='OK' and not task.get('val_only',False):
                tested=execute(dict(spec,mode='post_selection_test',phase='X18:test'))
                info.update(status=tested['status'],test=tested,test_acc=tested.get('metrics',{}).get('test_acc'))
        except Exception as exc:
            info.update(status='OOM' if isinstance(exc,torch.OutOfMemoryError) or 'out of memory' in str(exc).lower() else 'incomplete',
                error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        info.update(finished=True,ended_epoch_s=time.time());write(path/'run.json',info);artifacts(path,info,spec)
        completed.append(info);write(out/'results.json',completed);gc.collect();torch.cuda.empty_cache()
    write(out/'worker_status.json',dict(status='OK',completed=len(completed),planned=len(job['tasks']),ended_epoch_s=time.time()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(read(a.job),a.out)


if __name__=='__main__':main()
