
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from .common import ROOT,OUT,read,write,rows,table,affected,VERSION,cpu_only


def index_runs():
    root=ROOT/'results/icde27'
    paths=list(root.glob('*/*/run.json'))+list(root.glob('*/runs/*/run.json'))
    return {p.parent.name:p.parent for p in paths if not p.is_relative_to(OUT)}


def results(source):
    found={}
    p=source/'events.jsonl'
    if p.exists():
        for line in p.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if r.get('event')=='cell_result' and r.get('mode')=='retrain':
                found[r['fingerprint']]=r['result']
    p=source/'partial_retrains.json'
    if p.exists():
        for r in read(p):found[r.get('fingerprint',r.get('id'))]=r
    p=source/'pipeline/queue.sqlite'
    if p.exists():
        with sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True) as c:
            for fp,value,status,raw in c.execute("SELECT fingerprint,value,status,result_json FROM tasks WHERE phase LIKE '%:retrain'"):
                found[fp]=json.loads(raw) if raw else dict(fingerprint=fp,value=value,status=status)
    info=read(source/'run.json')
    for key in ('retrain_result','reference_result'):
        if info.get(key):
            r=info[key];found[r.get('fingerprint',info.get('fingerprint',info.get('winner')))]=r
    p=source/'retrain.tsv'
    if p.exists():
        for r in rows(p,'\t'):
            if not r.get('fingerprint'):continue
            if r.get('result'):
                value=json.loads(r['result'])
            else:
                value=dict(r)
                for k in ('value','val_acc'):
                    if r.get(k):value['value']=float(r[k])
                if r.get('metrics'):value['metrics']=json.loads(r['metrics'])
            found.setdefault(r['fingerprint'],value)
    return found


def specs(source,info,trained):
    for name in ('resolved_specs.json','retrain_spec.json','reference_spec.json'):
        if (source/name).exists():
            value=read(source/name)
            return value if isinstance(value,list) else [value]
    found=[]
    for p in source.glob('pipeline/**/specs/*retrain*.json'):found.append(read(p))
    for p in source.glob('pipeline/*retrain*.job.json'):found.extend(read(p).get('specs',[]))
    if found:return list({(s['fingerprint'],s.get('replica',0)):s for s in found if s.get('mode')=='retrain'}.values())
    if info.get('job',{}).get('frozen_specs'):return info['job']['frozen_specs']
    queue=source/'pipeline/queue.sqlite'
    if not queue.exists():raise FileNotFoundError('no frozen retrain specification: '+str(source))


    from sregnas.config import ExperimentConfig,_strict_dataclass
    from sregnas.execution.tasks import _cell_spec
    from sregnas.execution.startup import partition_receipt
    from sregnas.core.contracts import TaskKey
    cfg=_strict_dataclass(ExperimentConfig,info['config'],'config')
    manifests=[p for p in source.glob('pipeline/graph_cache/**/manifest.json') if 'full_graph_path' in read(p)]
    if len(manifests)!=1:raise ValueError(f'expected one prepared graph, got {manifests}')
    graph=read(manifests[0]);cfg._partition_receipt=partition_receipt(manifests[0])
    with sqlite3.connect('file:'+str(queue)+'?mode=ro',uri=True) as c:
        for phase,round_,gpu,shard,fp,arch in c.execute("SELECT phase,round,gpu,shard,fingerprint,architecture_json FROM tasks WHERE phase LIKE '%:retrain'"):
            spec=_cell_spec(key=TaskKey(phase,round_,gpu,fp),mode='retrain',shard=shard,
                architecture=json.loads(arch),graph_path=graph['full_graph_path'],cfg=cfg,physical_gpu=gpu)
            spec['checkpoint_path']=trained[fp].get('metrics',{}).get('checkpoint_path')
            found.append(spec)
    return found


def input_sources(index):
    requests=defaultdict(list);skipped=[]
    def add(row,table_name):
        if row.get('dataset') not in ('Reddit','Products'):return
        rid=row.get('run_id')
        if not rid:return
        if row.get('status') not in ('OK','reused',''):
            skipped.append(dict(source_table=table_name,source_run_id=rid,status=row['status'],reason='source not OK'));return
        requests[rid].append(dict(source_table=table_name,row=row))
    main=rows(ROOT/'results/icde27/tables/exp1_seeds.csv')
    selected={}
    for r in main:
        ds=r['dataset'];method=r['method'];rid=r['run_id']
        if ds not in ('Reddit','Products') or r['status']!='OK':continue
        if method=='SREGNAS' and '-s0_' not in rid:continue
        if method=='AP' and ('-gcfix_' not in rid or 'Q400' not in rid):continue
        if method!='PAS-PL' and not rid.endswith('_nfc'):continue
        selected[(ds,method,r['seed'])]=r
    for r in selected.values():add(r,'Table III / Exp1 current seeds')
    for path,delimiter in [('tables/x9d_exp1r.csv',','),('x14/reference_val_test.csv',','),
            ('x14/fixed_partition_cells.csv',','),('x14/sensitivity_cells.csv',','),
            ('x11r/a_time_accuracy.tsv','\t'),('x11r/b_random_search.tsv','\t'),
            ('x11r/c_single_shard.tsv','\t'),('x11r/d_partition.tsv','\t')]:
        for row in rows(ROOT/'results/icde27'/path,delimiter):add(row,path)
    for rid,source in index.items():
        if ('replica' in rid and rid.startswith('expX15_')) or (source.parent.name=='EXP9' and rid.endswith('_nfc')):
            info=read(source/'run.json');add(dict(dataset=info['dataset'],run_id=rid,status=info['status']),'X15 Exp1R / Table VI nomination')
    return requests,skipped


def semantic_key(spec,info):
    from sregnas.icde27.quality.replicas import derived_seed
    graph=read(spec['graph_path'])
    workload={k:spec[k] for k in ('architecture','fingerprint','dataset','epochs','lr','weight_decay','dropout',
        'sampler_impl','feature_residency','inference_residency','replica','retrain_seed_policy')}
    key=dict(backend=info['backend'],workload=workload,derived_seed=derived_seed(info['method'],spec),
        graph_sha256=graph['graph_sha256'],graph_arrays=graph['arrays'],
        split_hash=info['split_hash'],final_budget='25 complete passes; fanout5,5; batch4096; G3',
        NF=True,clip=1.0,checkpoint='final pass, exact validation',version=VERSION)

    key['graph_arrays']={k:{a:b for a,b in v.items() if a!='path'} for k,v in key['graph_arrays'].items()}
    return key


def prepare():
    started=time.time();cpu_only();index=index_runs();requests,skipped=input_sources(index)
    records=[];lists=[];tasks={};unresolved=[];dedup={}
    for rid,uses in sorted(requests.items()):
        if rid not in index:
            unresolved.append(dict(source_run_id=rid,reason='run.json missing'));continue
        source=index[rid];info=read(source/'run.json')
        if info['status']!='OK':
            skipped.append(dict(source_run_id=rid,status=info['status'],reason='run.json not OK'));continue
        try:
            trained=results(source);frozen=specs(source,info,trained)
            if not frozen:raise ValueError('empty retrain list')
            oldwinner=info.get('winner',info.get('fingerprint'))
            if isinstance(oldwinner,dict):oldwinner=oldwinner.get('winner',oldwinner.get('fingerprint',oldwinner.get('id')))
            entries=[]
            for spec in frozen:
                fp=spec['fingerprint'];replica=spec.get('replica',0)
                old=trained.get(fp,{})
                val=old.get('value',old.get('val_acc'))
                if val is None and len(frozen)==1:val=info.get('val_acc')
                oldtest=info.get('test_acc') if fp==oldwinner else None
                for use in uses:
                    r=use['row']
                    if r.get('fingerprint',r.get('winner_fp',r.get('winner_fingerprint',r.get('winner'))))==fp:
                        if r.get('test_pct'):oldtest=float(r['test_pct'])/100
                        if val is None and r.get('val_pct'):val=float(r['val_pct'])/100
                is_affected=affected(spec['architecture'],info['backend'])
                record=dict(source_run_id=rid,dataset=info['dataset'],method=info['method'],backend=info['backend'],
                    fingerprint=fp,replica=replica,affected=is_affected,old_status=old.get('status','OK'),
                    old_val=val,old_test=oldtest,old_checkpoint=spec.get('checkpoint_path'),
                    source_tables=sorted({u['source_table'] for u in uses}),version=VERSION)
                if is_affected:
                    key=semantic_key(spec,info);seed=key['derived_seed'];triplet=(fp,info['dataset'],seed)
                    digest=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()
                    if triplet in dedup and dedup[triplet]!=digest:
                        raise ValueError('same (fingerprint,dataset,Pi seed) has conflicting complete training keys: '+str(triplet))
                    dedup[triplet]=digest;tid=f'{info["dataset"]}_{fp}_s{seed}'
                    task=tasks.setdefault(tid,dict(task_id=tid,dataset=info['dataset'],spec=deepcopy(spec),key=key,key_hash=digest,
                        source_run_id=rid,source_path=str(source),metadata={k:info.get(k) for k in
                        ('environment','backend','config_hash','graph_hash','split_hash','space_hash','seed_rule')},sources=[]))
                    task['sources'].append(rid);record.update(task_id=tid,pi_seed=seed,key_hash=digest)
                else:record.update(task_id='',reuse_reason='unaffected operator path' if info['backend']=='canonical' else 'native does not call canonical NF layer')
                records.append(record);entries.append(record)
            lists.append(dict(source_run_id=rid,source_path=str(source),dataset=info['dataset'],method=info['method'],
                backend=info['backend'],old_winner=oldwinner,source_uses=uses,entries=entries,
                seed=info['config']['search']['base_seed'],kind='selection' if len(entries)>1 else 'reference_or_replica'))
        except Exception as exc:
            unresolved.append(dict(source_run_id=rid,source_path=str(source),reason=f'{type(exc).__name__}: {exc}'))
    plan=dict(version=VERSION,started_epoch_s=started,ended_epoch_s=time.time(),lists=lists,tasks=list(tasks.values()),
        input_records=records,skipped=skipped,unresolved=unresolved,requested_runs=len(requests),
        affected_records=sum(r['affected'] for r in records),unique_affected=len(tasks),
        gate='a0_status.passed OR explicit user acceptance saved in gate_decision.json')
    write(OUT/'pi_fix_plan.json',plan);table(OUT/'pi_fix_inventory.csv',records)
    table(OUT/'pi_fix_unresolved.csv',unresolved,fields=['source_run_id','source_path','reason'])
    table(OUT/'pi_fix_skipped_sources.csv',skipped,fields=['source_table','source_run_id','status','reason'])
    print({k:plan[k] for k in ('requested_runs','affected_records','unique_affected','unresolved')},flush=True)
    return plan


if __name__=='__main__':prepare()
