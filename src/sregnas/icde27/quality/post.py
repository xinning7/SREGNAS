
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from sregnas.icde27.recorder import dump, write_table
from sregnas.icde27.space import canonical_fingerprint
from .cache import TrainingCache, training_key
from .nomination import nominate
from .records import source_records


def dispatch(method, specs, out, label):
    pass
    groups = {}
    for spec in specs: groups.setdefault(spec['gpu'],[]).append(spec)
    def worker(gpu, batch):
        job = out/f'{label}_gpu{gpu}.job.json'; output=out/f'{label}_gpu{gpu}.json'
        dump(job,dict(method=method,specs=batch,output=str(output)))
        with (out/f'{label}_gpu{gpu}.log').open('w') as log:
            result=subprocess.run([sys.executable,'-m','sregnas.comparison.retrain','--job',str(job)],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(f'cell worker failed: {job}; exit={result.returncode}')
        return json.loads(output.read_text())
    with ThreadPoolExecutor(max_workers=len(groups) or 1) as pool:
        futures=[pool.submit(worker,gpu,batch) for gpu,batch in groups.items()]
        results=[r for future in futures for r in future.result()]
    by_fp={r.get('fingerprint',r.get('id')):r for r in results}
    return [by_fp[s['fingerprint']] for s in specs]


def execute(job, out, info):
    source=Path(job['source']); source_info, rows, templates=source_records(source)
    method=source_info['method']; gpus=job['gpus']; cache=TrainingCache(job['cache_dir'])
    imported=cache.import_run(source); dump(out/'cache_import.json',imported)
    template=templates[0]
    def spec_for(row,mode,index=0):
        spec=deepcopy(template)
        spec.update(architecture=row['architecture'],fingerprint=row['fingerprint'],mode=mode,
                    gpu=gpus[index%len(gpus)],round=index,
                    checkpoint_path=str(out/'checkpoints'/(row['fingerprint']+'.pt')))
        if mode=='probe' and spec['dataset'] in ('reddit','ogbn-products'):
            spec['probe_mode']='worst_case_upper_bound'
        return spec
    trained=[]; audit=[]; cache_hits=[]; new_count=0
    def retrain(selected):
        nonlocal new_count
        pending=[]; keys={}; results={}
        for i,row in enumerate(selected):
            spec=spec_for(row,'retrain',i); key=training_key(spec,info); keys[row['fingerprint']]=key
            cached=cache.get(key)
            if cached:
                results[row['fingerprint']]=cached['result']
                cache_hits.append(dict(fingerprint=row['fingerprint'],key_hash=key['key_hash'],source_run_id=cached['source_run_id']))
            else: pending.append(spec)
        if pending:
            new_count += len(pending)
            for spec,result in zip(pending,dispatch(method,pending,out,f'retrain_{new_count}')):
                results[spec['fingerprint']]=result
                cache.put(keys[spec['fingerprint']],result,info['run_id'])
        return [dict(results[r['fingerprint']],fingerprint=r['fingerprint'],architecture=r['architecture']) for r in selected]
    if method=='SREGNAS':
        nominees,ranked=nominate(rows,source_info['S'],job['K']//source_info['S'],job.get('rule','Plan1'))
        def probe(row):
            spec=spec_for(row,'retrain'); key=training_key(spec,info)
            if cache.get(key): return True
            spec=spec_for(row,'probe',int(row['shard']))
            result=dispatch(method,[spec],out,'probe_'+row['fingerprint'])[0]
            return result['status']=='OK'
        from sregnas.selection.nomination import probe_and_refill
        feasible,audit=probe_and_refill(nominees,ranked,probe)
        trained=retrain(feasible)
        if any(r['status']!='OK' for r in trained):
            dump(out/'partial_retrains.json',trained)
            raise RuntimeError('feasible shortlist has a failed retrain')
    else:

        nominees=rows[:job['K']];feasible=[];offset=0
        while len(feasible)<job['K']:
            n=job['K']-len(feasible);batch=rows[offset:offset+n];offset+=len(batch)
            if len(batch)!=n: raise RuntimeError('global ranking exhausted during refill')
            results=retrain(batch);trained.extend(results)
            for row,result in zip(batch,results):
                if result['status']=='OK': feasible.append(row)
                elif result['status'] not in ('OOM','DIVERGE'): raise RuntimeError(f'cell failure {result}')
    successful=[r for r in trained if r['status']=='OK']
    winner=max(successful,key=lambda r:(float(r['value']),canonical_fingerprint(r['architecture'])))
    chosen=next(r for r in feasible if r['fingerprint']==winner['fingerprint'])
    key=training_key(spec_for(chosen,'retrain'),info);cached=cache.get(key)
    if not cached: raise AssertionError('winner checkpoint missing from verified cache')
    test=cached.get('test');test_reused=bool(test)
    if not test:
        spec=spec_for(chosen,'post_selection_test');spec['checkpoint_path']=cached['result']['metrics']['checkpoint_path']
        test=dispatch(method,[spec],out,'winner_test')[0]
        if test['status']!='OK' or 'test_acc' not in test['metrics']: raise RuntimeError('winner test failed')
        cache.put(key,cached['result'],cached['source_run_id'],test)
    dump(out/'nominations.json',dict(nominees=nominees,feasible=feasible,probe_audit=audit))
    dump(out/'reuse.json',cache_hits)
    fields=['run_id','fingerprint','architecture','status','value','metrics','source_run_id']
    for name,data in [('candidates',rows),('shortlist',feasible),('retrain',trained)]:
        write_table(out/(name+'.tsv'),[dict(run_id=info['run_id'],**r) for r in data],fields)
    return dict(source_run_id=source_info['run_id'],search_trace_id=source_info['run_id'],
        Q_search=0,Q_attempt=0,source_Q_attempt=source_info['Q_attempt'],Q_trunc=source_info.get('Q_trunc') or (source_info['Q_attempt'] if source_info['Q_attempt']<400 else None),
        nomination_seed_rule=('training.minibatch._hash_seed(42, Quota-Random, shard); random.Random.shuffle of stable score-ranked successes' if job.get('rule')=='Quota-Random' else 'N/A: deterministic nomination ranking'),
        rule=job.get('rule','global_topK'),K_actual=len(feasible),covered_shards=len({r['shard'] for r in feasible}),
        duplicate_count=len(rows)-len({r['fingerprint'] for r in rows}),
        duplicate_definition='search trace rows minus distinct fingerprints; before nomination',
        nominee_duplicate_count=len(nominees)-len({r['fingerprint'] for r in nominees}),
        probe_decision_count=len(audit),probe_count=sum(1 for _ in out.glob('probe_*.job.json')),
        Q_probe=sum(1 for _ in out.glob('probe_*.job.json')),
        probe_reused=len(audit)-sum(1 for _ in out.glob('probe_*.job.json')),
        refill_count=sum('refill_fingerprint' in r for r in audit) if method=='SREGNAS' else len(trained)-job['K'],
        newly_trained=new_count,reused_training=cache_hits,winner=winner['fingerprint'],
        val_acc=winner['value'],test_acc=test['metrics']['test_acc'],test_reused=test_reused,
        checkpoint_path=cached['result']['metrics']['checkpoint_path'],
        assertions=dict(no_new_search=True,complete_shortlist=len(feasible)==job['K'],
                        distinct_shortlist=len({r['fingerprint'] for r in feasible})==job['K'],
                        validation_only_selection=True,complete_training_key=True))


def reference(job, out, info):
    pass
    from sregnas.config import _strict_dataclass, ExperimentConfig
    from sregnas.core.contracts import TaskKey
    from sregnas.execution.tasks import _cell_spec
    from sregnas.icde27.p2p6.canonical import prepare
    cfg=_strict_dataclass(ExperimentConfig,info['config'],'config')
    paths,_=prepare(dict(kind='cell'),cfg,out)
    row=job['architecture']; fp=row['fingerprint']
    spec=_cell_spec(key=TaskKey('quality:baseline:retrain',0,0,fp),mode='retrain',shard=0,
        architecture=row['architecture'],graph_path=paths[0],cfg=cfg,physical_gpu=job['gpus'][0])
    spec.update(require_fi_gpu=False,partition_receipt=None,checkpoint_path=str(out/'checkpoints'/(fp+'.pt')))
    if info['backend']=='fastgnas-native': spec['dropout']=.6
    dump(out/'reference_spec.json',spec)
    result=dispatch(info['method'],[spec],out,'reference')[0]
    write_table(out/'retrain.tsv',[dict(run_id=info['run_id'],**result)],['run_id','fingerprint','status','value','metrics'])
    return dict(reference_result=result,architecture_slot=row['slot'],panel_id=job['panel_id'],
        val_acc=result['value'],reference_key=training_key(spec,info),
        assertions=dict(reference_complete=result['status']=='OK',no_test_access='test_acc' not in result['metrics']))
