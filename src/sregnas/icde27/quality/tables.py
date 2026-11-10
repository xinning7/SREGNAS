
import csv
import json
from pathlib import Path
from sregnas.icde27.recorder import read_rows,sha,dump,write_table
from sregnas.icde27.p2p6.plan import pool,DATASETS,METHODS,digest


def read(path,delimiter='\t'):
    if not path.exists():return [],[]
    with path.open() as f:
        r=csv.DictReader(f,delimiter=delimiter);return list(r),r.fieldnames


def append_table(path,rows,key='run_id'):
    old,fields=read(path);fields=list(fields or [])
    known={r[key] for r in old if key in r}
    fresh=[r for r in rows if r[key] not in known]
    for r in fresh:
        for field in r:
            if field not in fields:fields.append(field)
    if fresh:
        path.parent.mkdir(parents=True,exist_ok=True);write_table(path,old+fresh,fields)


def final_accuracy(path,info=None):
    info=info or json.loads((path/'run.json').read_text())
    if 'test_acc' in info:return info.get('val_acc'),info['test_acc'],info.get('winner')
    if (path/'pipeline/result.json').exists():
        r=json.loads((path/'pipeline/result.json').read_text());return r.get('val_acc'),r.get('test_acc'),r.get('winner')
    events=read_rows(path/'events.jsonl')
    tests=[r['result'] for r in events if r['event']=='cell_result' and r['mode']=='post_selection_test' and r['result']['status']=='OK']
    winners=[r for r in events if r['event']=='winner_selected']
    return (winners[-1].get('val_acc') if winners else None,
            tests[-1]['metrics']['test_acc'] if tests else None,winners[-1]['winner'] if winners else None)


def attempts(state):
    return {r['cell_id']:r for r in state['attempts'] if r['status']!='requeue'}


def record(job,latest):
    a=latest.get(job['cell_id'],dict(status='pending'))
    path=Path(a.get('result_path') or '/missing');info=json.loads((path/'run.json').read_text()) if (path/'run.json').exists() else {}
    return a,path,info


def selection(path):
    manifests=list(path.glob('pipeline/graph_cache/*/graph/manifest.json'))
    return json.loads(manifests[0].read_text()) if manifests else {}


def primary(root,dataset,method,state):
    rows=[]
    for p in (root/'results/icde27/EXP1').glob('*/run.json'):
        d=json.loads(p.read_text())
        if d['method']==method and d['dataset']==dataset and d['status']=='OK' and all(d['assertions'].values()) and d['config']['search']['base_seed']==42:
            if method=='SREGNAS' and bool(d.get('s0_parallel'))!=not_abandoned(state):continue
            rows.append((p.parent,d))
    return rows[0] if len(rows)==1 else (None,{})


def not_abandoned(state):return not state.get('s0_abandoned',False)


def s0_tables(root,state):
    additions=[];coarse=[];speed=[];evidence=[];latest=attempts(state)
    phases={'prep':'prep','search':'search','nomination+probe+refill':'nomination_probe_refill',
            'retrain+validation':'retrain_validation','winner-test':'winner_test','other':'other'}
    report=[]
    for job in state['jobs']:
        if job['experiment_id']!='1':continue
        a,path,info=record(job,latest)
        if not info:continue
        val,test,winner=final_accuracy(path,info);p={r['phase']:r for r in info.get('phases',[])}
        row={k:info.get(k,'') for k in ('run_id','method','dataset','environment','commit','status','record_type','timing_valid','Q_attempt','Q_search','Q_probe','Q_retry','Q_resume')}
        row.update(flag='--s0-parallel',s0_parallel=True,source=str(path/'run.json'),val_acc=val,test_acc=test,
                   E2E_s=info.get('direct_e2e_wall_s',''),T_search_s=p.get('search',{}).get('wall_s',''),
                   K_actual=info['K'] if info['status']=='OK' else '',ASSERTION_PASS=all(info['assertions'].values()))
        for name,prefix in phases.items():row[prefix+'_s']=p.get(name,{}).get('wall_s','')
        ledger=read_rows(path/'launch_ledger.jsonl')
        candidates=[r for r in ledger if r.get('stage')=='search' and r['kind']=='candidate']
        good=[r for r in candidates if r.get('status')=='OK']
        row.update(Q_candidate=len(candidates),Q_success=len(good),
            U_unique=len({r['fingerprint'] for r in candidates}),U_unique_success=len({r['fingerprint'] for r in good}),
            Q_requested=info.get('Q_requested',''),Q_trunc=info.get('Q_trunc',''),Q_warmup=info.get('Q_warmup',''),
            FI_selected=(selection(path).get('partition_selection') or {}).get('selected',''),
            recorder_status=info['status'],process_wall_s=info.get('process_wall_s',''),
            reserved_GPU_hours=info['P']*info['direct_e2e_wall_s']/3600 if info.get('direct_e2e_wall_s') is not None else '',
            process_reserved_GPU_hours=info.get('reserved_gpu_hours',''),
            phase_wall_discrepancy_fraction=info.get('phase_wall_discrepancy_fraction',''),
            failure_reason=a.get('reason',''))
        from sregnas.icde27.x_accounting import completed_accounting
        row.update(completed_accounting(path, info))
        additions.append(row)
        c={k:row[k] for k in ('run_id','method','dataset','environment','status','record_type','timing_valid','flag')}
        c.update(direct_E2E_s=info.get('direct_e2e_wall_s',''),phase_sum_s=info.get('phase_sum_s',''),physical_gpus=info['config']['runtime']['system_gpus'])
        for name,prefix in phases.items():
            for suffix in ('start_perf_s','end_perf_s','wall_s'):c[prefix+'_'+suffix]=p.get(name,{}).get(suffix,'')
        coarse.append(c)
        reference,ref=primary(root,job['dataset'],'FastGNAS',state)
        if reference and ref['timing_valid'] and info['status']=='OK' and info['timing_valid'] and info['record_type']==ref['record_type']=='full_nas':
            rp={r['phase']:r for r in ref['phases']}
            speed.append(dict(comparison_id=info['run_id']+'__'+ref['run_id'],dataset=job['dataset'],method='SREGNAS',reference_method='FastGNAS',
                record_type='full_nas',status='OK',search_speedup=rp['search']['wall_s']/p['search']['wall_s'],
                E2E_speedup=ref['direct_e2e_wall_s']/info['direct_e2e_wall_s'],fastgnas_source_run_id=ref['run_id'],
                sregnas_source_run_id=info['run_id'],method_source_run_id=info['run_id'],flag='--s0-parallel',comparison_owner='S0'))
        events=read_rows(path/'events.jsonl')
        for event in events:
            if event['event']=='prep_step_end':evidence.append(dict(run_id=info['run_id'],dataset=job['dataset'],**event))
        old=[q.parent for q in (root/'results/icde27/EXP1').glob('*/run.json') if (d:=json.loads(q.read_text()))['method']=='SREGNAS' and d['dataset']==job['dataset'] and d['status']=='OK' and not d.get('s0_parallel')]
        trace={}
        if len(old)==1 and info['status']=='OK':
            from .records import source_records
            _,before,_=source_records(old[0]);_,after,_=source_records(path)
            trace=dict(legacy_source=old[0].name,same_proposals=[(r['shard'],r['round'],r['fingerprint']) for r in before]==[(r['shard'],r['round'],r['fingerprint']) for r in after],
                       same_proxy_scores=[r['value'] for r in before]==[r['value'] for r in after],same_winner=final_accuracy(old[0])[2]==winner)
        dump(path/'s0_trace_comparison.json',trace)
        report.append(dict(dataset=job['dataset'],run_id=info['run_id'],status=info['status'],E2E_s=row['E2E_s'],**trace))
    exp1=root/'results/icde27/EXP1'
    append_table(exp1/'EXP1_summary.tsv',additions)
    append_table(root/'results/icde27/EXP3/EXP3_coarse.tsv',coarse)
    append_table(exp1/'speedup.tsv',speed,key='comparison_id')
    audit=root/'docs/audit';dump(audit/'s0_new_timing.json',evidence);dump(audit/'s0_trace_comparisons.json',report)
    with (audit/'s0_report.md').open('a') as f:
        f.write('\n## 正式运行\n\n'+('S0 已放弃，原因：语义等价单元证据未通过；沿用旧 Exp1。\n' if state.get('s0_abandoned') else '新旧记录均保留；搜索轨迹差异如实记录，不作为语义等价失败。\n'))
        for row in report:f.write('\n- '+json.dumps(row,ensure_ascii=False)+'\n')


def write_schema(root,name,rows):
    source=root/f'docs/design/figures/data/{name}.csv'
    with source.open() as f:fields=next(csv.reader(f))
    fields=list(dict.fromkeys(fields+['source_run_id','reason']))
    target=root/f'results/icde27/tables/{name}.csv';target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def quality_tables(root,state):
    latest=attempts(state);fi=[];nom=[];sens=[];shards=[];refs=[];fixed=[];details=[];random_rows=[]
    def common(job,a):return dict(dataset=job['dataset'],method=job['method'],Q=job['Q'],K=job['K'],seed=42,repeats=1,
        run_id=a.get('run_id',''),status=a['status'],reason=a.get('reason',''),source_run_id=a.get('source_run_id',''))
    for job in state['jobs']:
        a,path,info=record(job,latest);base=common(job,a);val,test,winner=final_accuracy(path,info) if info else (None,None,None)
        if info.get('Q_trunc') or (job['method'] in ('GraphNAP','AP') and job['dataset']=='Products' and job['kind']=='nas'):
            base['reason']=(base.get('reason','')+'; nominal Q400 truncated; source/actual attempts='+str(info.get('Q_trunc') or info.get('Q_attempt',''))).strip('; ')
        base['test_accuracy_pct']=100*test if test is not None and a['status']=='OK' else ''
        exp=job['experiment_id']
        if exp=='9':nom.append(dict(base,rule=job['label'],search_trace_id=info.get('search_trace_id','')))
        if exp=='10':sens.append(dict(base,sweep='E_proxy' if job['label']=='E25' else 'K',value=25 if job['label']=='E25' else 16,fixed_config_id=info.get('config_hash','')))
        if exp=='11':
            manifest=selection(path);part=manifest.get('partition_selection') or {}
            shards.append(dict(base,physical_gpus=2,logical_shards=job['S'] if job['method']=='SREGNAS' else 'N/A' if job['method']=='AP' else 2))
            ledger=read_rows(path/'launch_ledger.jsonl')
            workers={r.get('gpu') for r in ledger if r.get('stage')=='search' and r.get('gpu') is not None}
            segments=read_rows(path/'native_segments.jsonl')
            if segments:workers={r.get('worker') for r in segments if r.get('phase')=='search' and r.get('worker') is not None}
            portrows=json.loads((path/'pipeline/port_candidates.json').read_text()) if (path/'pipeline/port_candidates.json').exists() else []
            if job['method']=='GraphNAP':workers={w['worker_id'] for r in portrows for w in r.get('workers',[])}
            elif job['method']=='AP':workers={r['worker_id'] for r in portrows if r.get('worker_id') is not None}
            pp,pi=primary(root,job['dataset'],'SREGNAS',state)
            main_winner=final_accuracy(pp,pi)[2] if pp else None
            details.append(dict(base,experiment_id=exp,selected_partition=part.get('selected'),actual_active_workers=len(workers) if workers else None,
                active_worker_ids=sorted(workers),primary_winner=main_winner if job['method']=='SREGNAS' and job['S']==8 else None,
                same_winner_as_P8=winner==main_winner if winner and main_winner and job['method']=='SREGNAS' and job['S']==8 else None,per_shard_Q=[job['Q']//job['S']]*job['S'] if job['method']=='SREGNAS' else None,
                graph_shards=manifest.get('shards'),message_count=sum(r['E_msg_sub'] for r in manifest.get('shards',[])) if manifest else None,
                cache_bytes=sum(Path(r['path']).stat().st_size for r in manifest.get('shards',[]) if Path(r['path']).exists()) if manifest else None,winner=winner))
        if exp=='13':
            ledger=read_rows(path/'launch_ledger.jsonl');launched=[r for r in ledger if r['kind']=='candidate'];unique=len({r['fingerprint'] for r in launched})
            random_rows.append(dict(base,unique_candidates=unique,duplicate_rate=1-unique/len(launched) if launched else '',winner=winner))
        if job['kind']=='reference':
            arch=job['architecture'];r=info.get('reference_result',{})
            refs.append(dict(dataset=job['dataset'],panel_id=job['panel_id'],pool_id=DATASETS[job['dataset']]+'_pool32',architecture_slot=arch['slot'],
                architecture_id=arch['fingerprint'],architecture_semantics_hash=digest(arch['architecture']),backend=info.get('backend',''),
                reference_profile_id='G3_25pass' if job['dataset']=='Products' else 'full_batch_200epoch',reference_key=info.get('reference_key',{}).get('key_hash',''),
                seed=42,repeats=1,reference_run_id=a.get('run_id',''),status=a['status'],validation_accuracy_pct=100*r['value'] if r.get('value') is not None and a['status']=='OK' else '',reason=a.get('reason','')))
        if exp in ('9','10'):details.append(dict(base,experiment_id=exp,K_actual=info.get('K_actual'),covered_shards=info.get('covered_shards'),duplicate_count=info.get('duplicate_count'),
            probe_count=info.get('probe_count'),probe_decision_count=info.get('probe_decision_count'),probe_reused=info.get('probe_reused'),refill_count=info.get('refill_count'),newly_trained=info.get('newly_trained'),reuse=info.get('reused_training'),source_Q_attempt=info.get('source_Q_attempt')))
    for dataset in DATASETS:
        path,info=primary(root,dataset,'SREGNAS',state)
        if not path:continue
        val,test,winner=final_accuracy(path,info);manifest=selection(path);part=manifest.get('partition_selection') or {};scores=part.get('scores',{})
        fixedjob=next(j for j in state['jobs'] if j['experiment_id']=='8' and j['dataset']==dataset)
        a,fp,fin=record(fixedjob,latest);ft=final_accuracy(fp,fin)[1] if fin and a['status']=='OK' else None
        selected=part.get('selected');other=fixedjob['partition']
        values={selected:100*test if test is not None else '',other:100*ft if ft is not None else ''}
        fi.append(dict(dataset=dataset,selected_partition=selected,Q=400,K=8,seed=42,repeats=1,selector_config_id=info['config_hash'],run_id=a.get('run_id',''),status=a['status'],
            fixed_metis_test_pct=values.get('metis',''),fixed_random_test_pct=values.get('random',''),fi_test_pct=100*test if test is not None else '',
            selection_prep_s=next((p['wall_s'] for p in info['phases'] if p['phase']=='prep'),''),timing_source='exclusive_primary',source_run_id=info['run_id']))
        details.append(dict(dataset=dataset,experiment='FI',source_run_id=info['run_id'],scores=scores,development_domain=dataset=='Flickr',
            SNR_only=max(('metis','random'),key=lambda k:scores[k]['SNR']) if scores else None,LA_only=max(('metis','random'),key=lambda k:scores[k]['LA']) if scores else None))
        nom.append(dict(dataset=dataset,rule='Plan1',Q=400,K=8,seed=42,repeats=1,search_trace_id=info['run_id'],run_id=info['run_id'],status='OK',test_accuracy_pct=100*test,source_run_id=info['run_id']))
        from .records import plan1_statistics
        details.append(dict(dataset=dataset,method='SREGNAS',experiment_id='9',rule='Plan1',run_id=info['run_id'],**plan1_statistics(path)))
    for dataset in ('arXiv','Products'):
        for method in METHODS:
            p,inf=primary(root,dataset,method,state)
            if not p:continue
            _,test,_=final_accuracy(p,inf)
            for sweep,value in [('K',8),('E_proxy',100 if method=='SREGNAS' else 50)]:
                sens.append(dict(dataset=dataset,method=method,sweep=sweep,value=value,Q=inf.get('Q_attempt'),K=8,seed=42,repeats=1,run_id=inf['run_id'],source_run_id=inf['run_id'],status=inf['status'],test_accuracy_pct=100*test if test is not None else '',reason='baseline E unit=optimizer steps' if method in ('GraphNAP','AP','FastGNAS') and sweep=='E_proxy' else ''))
    (root/'results/icde27/tables').mkdir(parents=True,exist_ok=True)
    fixed=fixed32_rows(root,state,refs)
    for name,rows in [('fi_selection',fi),('nomination',nom),('sensitivity',sens),('shard_quality',shards),('proxy_fixed32',fixed),('proxy_reference_cells',refs)]:write_schema(root,name,rows)
    dump(root/'results/icde27/tables/quality_details.json',details);dump(root/'results/icde27/tables/random_search.json',random_rows)
    lines=['# 质量实验记录','', '所有新质量格 timing_valid=false、timing_scope=quality_only，不进入任何加速比；S0 的新正式 Exp1 单独记时。',
        '', '本任务明确授权的预算/划分/分组锁覆盖旧协议限制；训练、采样、优化器、初始化与复训预算按原入口执行。Flickr 为 development domain。',
        '', '| 实验 | 数据集 | 方法/格 | 状态 | run_id / 原因 |','|---|---|---|---|---|']
    for j in state['jobs']:
        a=latest.get(j['cell_id'],dict(status='pending'));lines.append(f"| {j['experiment_id']} | {j['dataset']} | {j['method']} {j['label']} | {a['status']} | {a.get('run_id','')} {a.get('reason','')} |")
    lines.extend(['','结果表位于 `results/icde27/tables/`：fi_selection、nomination、sensitivity、shard_quality、proxy_fixed32、proxy_reference_cells。',
        '', '复用源见各表 source_run_id、各格 reuse.json 与 quality_details.json。固定池只按代理分数提名；按本轮收尾授权，最终复训协议下 OOM 的架构视为不可行，不计入池内最优，入围中的不可行架构按原代理排名补位（SREGNAS 保留片内配额和 refill 顺序）。未测或其它失败仍为未知，不当作不可行。参考验证分数仅用于提名结束后的指标计算。FastGNAS 在 panel_1 为 N/A (unaligned)，仅 panel_2 使用 native 参考。不报告 selection regret。',
        '', '各格起止与实际墙钟见 quality_status.tsv；失败和未测项保留空值，不外推。'])
    for name, rows, columns in [
        ('Exp8 FI',fi,['dataset','selected_partition','fixed_metis_test_pct','fixed_random_test_pct','fi_test_pct','selection_prep_s','status']),
        ('Exp9 提名',nom,['dataset','rule','test_accuracy_pct','status','source_run_id']),
        ('Exp10 敏感性',sens,['dataset','method','sweep','value','test_accuracy_pct','status']),
        ('Exp11 逻辑分片',shards,['dataset','method','logical_shards','test_accuracy_pct','status']),
        ('Exp12 固定32池',fixed,['dataset','method','panel_id','pool_best_val_pct','shortlist_best_val_pct','observed_shortlist_regret_pp','status','reason']),
        ('Exp13 随机搜索',random_rows,['dataset','test_accuracy_pct','unique_candidates','duplicate_rate','status'])]:
        lines.extend(['', '## '+name, '', '| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |'])
        for row in rows:lines.append('| '+' | '.join(str(row.get(c,'')) for c in columns)+' |')
    lines.extend(['', '## FI 原始统计与单指标选择', '', '| 数据集 | 划分 | SNR | LA | FI | SNR-only | LA-only | source_run_id |', '|---|---|---|---|---|---|---|---|'])
    for d in details:
        if d.get('experiment')!='FI':continue
        for partition,score in d.get('scores',{}).items():
            lines.append('| '+' | '.join(str(v) for v in [d['dataset'],partition,score['SNR'],score['LA'],score['FI'],d['SNR_only'],d['LA_only'],d['source_run_id']])+' |')
    columns=['dataset','method','run_id','K_actual','covered_shards','duplicate_count','probe_count','probe_reused','refill_count','newly_trained']
    lines.extend(['', '## 提名与重训计数', '', '| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |'])
    for d in details:
        if d.get('experiment_id') in ('9','10'):
            lines.append('| '+' | '.join('' if d.get(c) is None else str(d[c]) for c in columns)+' |')
    lines.extend(['', '重复数口径为搜索记录行数减唯一指纹数；probe_count 为真实 probe 启动数，probe_reused 为完整训练 key 已成功的可行性判定复用数。逻辑分片覆盖、消息量、缓存字节、worker 和 P8 winner 对照见 quality_details.json。'])
    (root/'docs/audit/quality_report.md').write_text('\n'.join(lines)+'\n')


def proxy_evidence(root, job, attempt, info):
    pass
    paths=[]
    sourcefile=root/'results/icde27/quality_jobs'/(job['cell_id']+'_source.json')
    if sourcefile.exists():paths.append(Path(json.loads(sourcefile.read_text())['source']))
    paths.extend(Path(p) for p in attempt.get('additional_proxy_sources',[]))
    raw=[];sources=[]
    for path in paths:
        d=json.loads((path/'run.json').read_text())
        if d['run_id'] not in sources:raw.extend(d.get('rows',[]));sources.append(d['run_id'])
    if info and info['run_id'] not in sources:raw.extend(info.get('rows',[]));sources.append(info['run_id'])
    return raw,sources


def feasible_fixed32_shortlist(method, rows, infeasible):
    pass
    from sregnas.selection.nomination import probe_and_refill
    if method=='SREGNAS':
        from .nomination import nominate
        nominees,ranked=nominate(rows,8,1)
    else:
        ordered=sorted(rows,key=lambda r:(-float(r['value']),r['fingerprint']))
        nominees,ranked=ordered[:8],{0:ordered}
    selected,audit=probe_and_refill(nominees,ranked,lambda r:r['fingerprint'] not in infeasible)
    return selected,audit,[r['fingerprint'] for r in nominees]


def fixed32_rows(root,state,refs):
    latest=attempts(state);output=[]
    for job in [j for j in state['jobs'] if j['kind']=='fixed32']:
        a,path,info=record(job,latest);dataset=job['dataset'];method=job['method'];panel='panel_2' if method=='FastGNAS' else 'panel_1'
        raw,sources=proxy_evidence(root,job,a,info)


        poolrows=pool(dataset,root)
        pool_ids={r['fingerprint'] for r in poolrows}
        mapping={r['fingerprint']:int(r['shard']) for r in raw if 'shard' in r} if method=='SREGNAS' else {r['fingerprint']:0 for r in raw}
        if method=='SREGNAS' and raw:
            frozen={r['fingerprint']:i%8 for i,r in enumerate(poolrows)}
            assert all(mapping.get(r['fingerprint'])==frozen[r['fingerprint']] for r in raw), 'recorded frozen mapping changed'
        scores={r['fingerprint']:r for r in raw if r['status']=='OK' and r.get('value') is not None}
        reference_rows=[r for r in refs if r['dataset']==dataset and r['panel_id']==panel]
        infeasible={r['architecture_id'] for r in reference_rows if r['status']=='OOM'}
        feasible={r['architecture_id'] for r in reference_rows if r['status']=='OK'}
        assert not infeasible & feasible, 'conflicting final-reference feasibility'
        classified=feasible|infeasible
        complete_reference=classified==pool_ids
        shortlist=[];audit=[];initial=[];refill_error=''
        if set(scores)==pool_ids and complete_reference:
            rows=[dict(r,shard=mapping[fp]) for fp,r in scores.items()]
            try:shortlist,audit,initial=feasible_fixed32_shortlist(method,rows,infeasible)
            except AssertionError as exc:refill_error=str(exc)
        observed=[r for r in reference_rows if r['status']=='OK']
        q={r['architecture_id']:r['validation_accuracy_pct'] for r in observed}
        ids=[r['fingerprint'] for r in shortlist];valid=complete_reference and len(ids)==8 and all(fp in q for fp in ids)
        refill_count=sum('refill_fingerprint' in r for r in audit)
        policy=f'final-retrain OOM excluded; feasible_pool={len(feasible)}/32; infeasible_OOM={len(infeasible)}; proxy-score refill={refill_count}'
        row=dict(dataset=dataset,method=method,rule='FI+Plan1' if method=='SREGNAS' else 'Native top-8' if method=='FastGNAS' else 'proxy top-8',
            Qbench=32,K=8,seed=42,repeats=1,panel_id=panel,pool_id=DATASETS[dataset]+'_pool32',architecture_semantics_pool_hash=digest([r['native_architecture_json'] if method=='FastGNAS' else r['architecture_json'] for r in poolrows]),
            backend='fastgnas-native' if method=='FastGNAS' else 'canonical',reference_profile_id='G3_25pass' if dataset=='Products' else 'full_batch_200epoch',alignment_status='aligned',
            verified_reference_set_key=digest([(r['reference_key'],r['status']) for r in reference_rows]) if complete_reference else '',reference_source_run_ids=json.dumps([r['reference_run_id'] for r in reference_rows]),
            run_id=a.get('run_id',''),source_run_id=json.dumps(sources),status='OK' if valid else 'incomplete',pool_best_val_pct=max(q.values()) if complete_reference and q else '',
            shortlist_best_val_pct=max(q[fp] for fp in ids) if valid else '',observed_shortlist_regret_pp=max(q.values())-max(q[fp] for fp in ids) if valid else '',
            reason=policy if valid else policy+f'; incomplete: proxy_scores={len(scores)}/32; known_reference_status={len(classified)}/32; '+refill_error)
        output.append(row)
        dump(root/'results/icde27/tables'/f'fixed32_{dataset}_{method}.json',dict(shortlist=ids,proxy_sources=sources,pool_best_in_shortlist=any(q[fp]==max(q.values()) for fp in ids) if valid else None,
            mapping='frozen Exp7 index_mod_8_v1' if method=='SREGNAS' else 'global',reference_independent=True,
            feasibility_policy='final_retrain_OOM_excluded_proxy_score_refill',
            infeasible_fingerprints=sorted(infeasible),feasible_pool_size=len(feasible),
            unknown_reference_fingerprints=sorted(pool_ids-classified),initial_shortlist=initial,
            refill_audit=audit,refill_count=refill_count,reference_values_used_for_nomination=False))
        if method=='FastGNAS':output.append(dict(row,panel_id='panel_1',alignment_status='unaligned',status='N/A',reason='fastgnas-native and canonical model semantics are unaligned',pool_best_val_pct='',shortlist_best_val_pct='',observed_shortlist_regret_pp='',reference_source_run_ids='',verified_reference_set_key=''))
    return output
