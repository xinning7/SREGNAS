
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import statistics
import time
from .common import ROOT,OUT,PYTHON,T0,GPU_END,read,write,now,X20_FLAGS
from .plan import cell,main_cells

DATASETS=('Reddit','Products','arXiv','Flickr')
FRAMEWORK_CODE=ROOT/'.runtime/x22-code/framework_release'
CANONICAL_CODE=ROOT/'.runtime/x22-code/canonical'
if (OUT/'writer_recovery_decision.json').exists():
    recovered=read(OUT/'writer_recovery_decision.json')
    if recovered.get('accepted'):
        FRAMEWORK_CODE=CANONICAL_CODE=Path(recovered['code'])
if (OUT/'existing_large_input_plan.json').exists():
    CANONICAL_CODE=Path(read(OUT/'existing_large_input_plan.json')['code'])

def info(c):
    p=Path(c['out'])/'run.json'
    return read(p) if p.exists() else {}

def table(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

def cpu(part,phase,filename):
    c=dict(cell_name='x22_decision_'+part,phase=phase,code=str(ROOT),expected_s=30,
        mandatory=True,internal_lock=False,flags={'CUDA_VISIBLE_DEVICES':'','SREGNAS_ALLOW_GPU':'0'},
        out=str(OUT/'checks'/('decision_'+part)),run_id='expX22_decision_'+part,
        argv=[PYTHON,'-m','sregnas.icde27.x22.decide','--part',part])
    write(OUT/'queue'/filename,[c])

def fixed16(c):
    job=read(c['job']);job['count']=16;write(c['job'],job)
    c.update(Q=16,workspace=':4096:8')
    c['run_id']=c['run_id'].replace('Q32K0','Q16K0');c['out']=c['out'].replace('Q32K0','Q16K0')
    c['argv']=[v.replace('Q32K0','Q16K0') for v in c['argv']]
    return c

def equality(left,right):
    a,b=info(left),info(right);ar,br=a.get('rows',[]),b.get('rows',[])
    complete=all(x.get('status')=='OK' for x in (a,b)) and len(ar)==len(br)==16
    checks=dict(complete=complete,fingerprints=complete and [r['fingerprint'] for r in ar]==[r['fingerprint'] for r in br],
        exact_proxy_scores=complete and all(x.get('value') is not None and x.get('value')==y.get('value') for x,y in zip(ar,br)),
        successful_candidates=complete and all(r['status']=='OK' for r in ar+br))
    if left['method']=='SREGNAS':
        checks['deterministic_enabled']=complete and all(r.get('metrics',{}).get('x20_deterministic_algorithms') is True for r in ar+br)
    if left.get('flags',{}).get('SREGNAS_X22_QUALIFY_NUMERICS')=='1':
        receipts=[[read(p) for p in (Path(c['out'])/'x22_qualification').glob('numerics_*.json')] for c in (left,right)]
        checks['worker_deterministic_receipts']=all(rs and all(r['torch_deterministic_algorithms'] and r['pyg_WITH_TORCH_SCATTER'] is False for r in rs) for rs in receipts)
    return dict(dataset=left['dataset'],method=left['method'],reference=left['out'],changed=right['out'],
        checks=checks,accepted=all(checks.values()),scores=[[r.get('value') for r in rows] for rows in (ar,br)])

def safe_flags(flags,method):
    return {k:v for k,v in flags.items() if method=='SREGNAS' or not k.startswith('SREGNAS_X19_')}

def framework():
    plan=read(OUT/'framework_qualification_effective.json') if (OUT/'framework_qualification_effective.json').exists() else read(OUT/'queue/030_framework_qualification.json');pairs=[]
    for c in plan:
        if c['category']!='qualify_reference':continue
        changed=next(x for x in plan if x['method']==c['method'] and x['dataset']==c['dataset'] and x['category']=='qualify_framework')
        pairs.append(equality(c,changed))
    cpu_receipt=read(OUT/'checks/framework_release/run.json')
    passed=cpu_receipt['status']=='OK' and all(r['accepted'] for r in pairs) and time.time()<=T0+3*3600
    flags=read(OUT/'framework_candidate_flags.json') if passed else {}
    decision=dict(at=now(),accepted=passed,flags=flags,comparisons=pairs,cpu_receipt=str(OUT/'checks/framework_release/run.json'),
        code=str(FRAMEWORK_CODE),ledger_check='CPU all SQL tables and launch/event records equal; no SQL or training semantics changed',
        reason='all required gates passed before development deadline' if passed else 'at least one equality/CPU/deadline gate failed; no unverified framework option enabled')
    write(OUT/'framework_decision.json',decision)

    early=[]
    for ds in DATASETS:
        for method in ('SREGNAS','PAS-PL'):
            for variant in ('x20','fixed'):
                c=cell(ds,method,category='early_'+variant,code=FRAMEWORK_CODE,pool=True,
                    flags=safe_flags(flags,method) if variant=='fixed' else {},expected_s=120 if ds=='Products' else 90)
                c.update(phase='1.4',reduction_rank=1 if ds in ('arXiv','Flickr') else 0);early.append(c)
    for method in ('SREGNAS','PAS-PL'):
        c=cell('Products',method,category='early_full',code=FRAMEWORK_CODE,flags=safe_flags(flags,method),
            expected_s=300 if method=='SREGNAS' else 420);c['phase']='1.4';early.append(c)
    write(OUT/'queue/040_early_readout.json',early);cpu('early','1.4','049_early_summary.json')
    ablation=[]
    toggles={'all_on':{},'no_prebuild':{'SREGNAS_X19_PREBUILD_BATCHES':'0'},
        'no_gpu_materialize':{'SREGNAS_X19_GPU_MATERIALIZE':'0'},'no_gpu_csr':{'SREGNAS_X19_GPU_CSR':'0'}}
    for rep in (1,2):
        for ds in ('Reddit','Products'):
            for variant,toggle in toggles.items():
                c=cell(ds,category=f'ablation_{variant}_r{rep}',pool=True,code=FRAMEWORK_CODE,
                    flags=dict(flags,**toggle),expected_s=120)
                c.update(phase='2.1',reduction_rank=1 if rep==2 else 0,ablation_variant=variant,replicate=rep);ablation.append(c)
    write(OUT/'queue/050_ablation.json',ablation);cpu('ablation','2.1','059_ablation_decision.json')
    return decision

def row(c):
    r=info(c);phases={p['phase']:p['wall_s'] for p in r.get('phases',[])}
    return dict(cell_name=c['cell_name'],run_id=c['run_id'],method=c['method'],dataset=c['dataset'],
        category=c['category'],status=r.get('status','not_run'),timing_valid=r.get('timing_valid',False),
        prep_s=r.get('prep_s',phases.get('prep')),pool_s=r.get('pool_s'),search_s=phases.get('search'),
        E2E_s=r.get('direct_e2e_wall_s',r.get('e2e_s')),completed=r.get('completed_candidates',r.get('Q_success')),
        source=str(Path(c['out'])/'run.json'))

def early():
    plan=read(OUT/'queue/040_early_readout.json')
    if (OUT/'queue/041_early_writer_repaired.json').exists():plan+=read(OUT/'queue/041_early_writer_repaired.json')
    if (OUT/'writer_recovery_plan.json').exists():plan.append(read(OUT/'writer_recovery_plan.json')['full'])
    rows=[row(c) for c in plan]
    table(OUT/'early_readout.csv',rows)
    fields=['dataset','method','category','status','prep_s','pool_s','search_s','E2E_s']
    text='## 1.4 早期读数（诊断，不进主表）\n\n| '+' | '.join(fields)+' |\n| '+' | '.join('---' for _ in fields)+' |\n'
    for r in rows:
        text+='| '+' | '.join(f'{r[k]:.2f}' if isinstance(r.get(k),float) else str(r.get(k) or '') for k in fields)+' |\n'
    (OUT/'early_readout.md').write_text(text)
    return dict(at=now(),rows=rows,status='OK')

def ablation():
    plan=read(OUT/'queue/050_ablation.json');rows=[]
    for c in plan:rows.append(dict(row(c),variant=c['ablation_variant'],replicate=c['replicate']))
    grouped=defaultdict(list)
    for r in rows:
        if r['status']=='OK' and r['timing_valid'] and r['completed']==32:grouped[r['variant']].append(r)
    summaries=[]
    for variant,items in grouped.items():
        by_dataset={ds:[r for r in items if r['dataset']==ds] for ds in ('Reddit','Products')}
        if any(not rs for rs in by_dataset.values()):continue
        summaries.append(dict(variant=variant,n=len(items),
            pool_mean_s=statistics.mean(statistics.mean(r['pool_s'] for r in rs) for rs in by_dataset.values()),
            prep_mean_s=statistics.mean(statistics.mean(r['prep_s'] for r in rs) for rs in by_dataset.values())))
    summary={r['variant']:r for r in summaries}
    winner=min(summaries,key=lambda r:(r['pool_mean_s'],r['variant']))['variant'] if summaries else 'all_on'
    flags=dict(X20_FLAGS)
    flags['SREGNAS_X19_PREBUILD_BATCHES']='0' if winner=='no_prebuild' else '1'
    for variant,key in [('no_gpu_materialize','SREGNAS_X19_GPU_MATERIALIZE'),('no_gpu_csr','SREGNAS_X19_GPU_CSR')]:
        if 'all_on' in summary and variant in summary:
            flags[key]='1' if summary['all_on']['prep_mean_s']<summary[variant]['prep_mean_s'] else '0'
    shared=read(OUT/'framework_decision.json')['flags'];flags.update(shared)
    value=dict(at=now(),status='OK' if summaries else 'incomplete',winner_by_pool=winner,flags=flags,rows=rows,summary=summaries,
        rule='equal-weight mean of the two dataset pool means; prep flags retained iff equal-weight prep mean is shorter; no accuracy consulted',
        canonical_gpu_csr_note='after common CSR input, gpu_csr is inactive regardless of the selected flag')
    write(OUT/'ablation_decision.json',value);table(OUT/'ablation_runs.csv',rows);table(OUT/'ablation_summary.csv',summaries)
    submit_canonical_qualification(flags)
    submit_figures(flags)
    return value

def submit_canonical_qualification(flags):
    ready=(OUT/'checks/canonical_input/run.json').exists() and read(OUT/'checks/canonical_input/run.json')['status']=='OK'
    ready=ready and all((OUT/'geometry'/ds/'run.json').exists() and read(OUT/'geometry'/ds/'run.json')['status']=='OK' for ds in DATASETS)
    if not ready or time.time()>=T0+4*3600:
        write(OUT/'canonical_decision.json',dict(at=now(),accepted=False,reason='input/structure checks incomplete or development deadline'))
        reservations=read(OUT/'reservations.json');reservations.pop('2.3.score_qualification',None);write(OUT/'reservations.json',reservations)
        submit_main(flags,False);return
    previous=read(OUT/'framework_qualification_effective.json') if (OUT/'framework_qualification_effective.json').exists() else read(OUT/'queue/030_framework_qualification.json');plan=[];pairs=[]
    for ds in DATASETS:
        for method in ('SREGNAS','PAS-PL','AP','GraphNAP'):
            ref=next((c for c in previous if c['dataset']==ds and c['method']==method and c['category']=='qualify_framework'),None)
            base=dict(safe_flags(flags,method),SREGNAS_X20_BOOTSTRAP='1',SREGNAS_X20_DETERMINISTIC='1')
            if method!='SREGNAS':base['SREGNAS_X22_QUALIFY_NUMERICS']='1'
            expected=500 if method=='GraphNAP' and ds in ('Reddit','Products') else 180 if method=='GraphNAP' else 75
            if ref is None:
                ref=fixed16(cell(ds,method,category='input_reference',pool=True,code=CANONICAL_CODE,flags=base,expected_s=expected))
                ref.update(phase='2.3.score_qualification',launch_deadline=T0+4*3600);plan.append(ref)
            changed=fixed16(cell(ds,method,category='input_common',pool=True,code=CANONICAL_CODE,
                flags=dict(base,SREGNAS_X22_CANONICAL_INPUT='1'),expected_s=expected))
            changed.update(phase='2.3.score_qualification',launch_deadline=T0+4*3600);plan.append(changed);pairs.append(dict(reference=ref,changed=changed))
    write(OUT/'input_qualification_pairs.json',pairs)
    write(OUT/'queue/060_input_qualification.json',plan);cpu('canonical','2.3','069_input_decision.json')

def pending_seconds():
    state=read(OUT/'driver_state.json');done=state['cells'];total=0
    for path in (OUT/'queue').glob('*.json'):
        for c in read(path):
            if c['cell_name'] not in done and c.get('mandatory',True):total+=c['expected_s']
    return total

def submit_figures(flags):
    cells=[]
    for ds in DATASETS:
        for method in ('SREGNAS','PAS-PL','AP'):
            c=cell(ds,method,category='table5',pool=True,code=FRAMEWORK_CODE,flags=safe_flags(flags,method),expected_s=90)
            c['phase']='2.2';cells.append(c)
    graphs=read(ROOT/'results/icde27/x18/v2/degree_inputs/manifest.json')
    for i,graph in enumerate(graphs):
        for method in ('SREGNAS','AP'):
            c=cell('Reddit',method,category='degree',pool=True,code=FRAMEWORK_CODE,
                flags=safe_flags(flags,method),expected_s=90,**graph)
            c.update(phase='2.2',reduction_rank=2 if i not in (0,len(graphs)//2,len(graphs)-1) else 0);cells.append(c)
    for ds in ('arXiv','Products'):
        for p in (2,4):
            for method in ('SREGNAS','AP'):
                c=cell(ds,method,category='scaling',pool=True,P=p,scaling_mode='fixed_workload',
                    code=FRAMEWORK_CODE,flags=safe_flags(flags,method),expected_s=150)
                c.update(phase='2.2',reduction_rank=3);cells.append(c)
    for ds in ('Flickr','arXiv'):
        c=cell(ds,'FastGNAS',category='published_fastgnas',code=FRAMEWORK_CODE,
            flags=safe_flags(flags,'FastGNAS'),expected_s=720 if ds=='Flickr' else 1020)
        c.update(phase='2.2',reduction_rank=1);cells.append(c)

    dropped=[];remaining=pending_seconds()+9360
    for rank in (3,2,1):
        if time.time()+sum(c['expected_s'] for c in cells)+remaining+1800<=GPU_END:break
        removed=[c for c in cells if c.get('reduction_rank')==rank];dropped+=removed
        cells=[c for c in cells if c not in removed]
    write(OUT/'figure_reductions.json',dict(at=now(),dropped=dropped,rule='Fig8, then intermediate degree points, then FastGNAS X20 reuse'))
    write(OUT/'queue/070_tables_figures_fastgnas.json',cells)

def canonical():
    pairs=[equality(r['reference'],r['changed']) for r in read(OUT/'input_qualification_pairs.json')]
    accepted=all(r['accepted'] for r in pairs) and time.time()<=T0+4*3600
    unchanged=[]
    if (OUT/'existing_large_input_plan.json').exists():
        check=OUT/'checks/existing_large_input/run.json'
        accepted=accepted and check.exists() and read(check).get('status')=='OK'
        unchanged=read(OUT/'existing_large_input_plan.json')['unchanged_large_baselines']
    value=dict(at=now(),accepted=accepted,comparisons=pairs,reason='all bitwise and deterministic score checks passed before deadline' if accepted else 'failed or late equality gate; canonical input remains disabled')
    value['unchanged_large_baselines']=unchanged
    write(OUT/'canonical_decision.json',value)
    flags=read(OUT/'ablation_decision.json')['flags'];submit_main(flags,accepted)
    return value

def submit_main(flags,canonical):
    import fcntl
    with (OUT/'formal_submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _submit_main(flags,canonical)

def _submit_main(flags,canonical):
    if (OUT/'formal_submitted.json').exists():return
    selected=dict(flags)
    if canonical:
        selected['SREGNAS_X22_CANONICAL_INPUT']='1'
        if (OUT/'existing_large_input_plan.json').exists():selected['SREGNAS_X22_CANONICAL_EXISTING_LARGE']='1'
    code=CANONICAL_CODE if canonical else FRAMEWORK_CODE
    cells=main_cells(code,selected)
    for c in cells:c['flags']=safe_flags(c['flags'],c['method'])

    pending=pending_seconds();dropped=[]
    for seed in (44,43):
        if time.time()+pending+sum(c['expected_s'] for c in cells)+1800<=GPU_END:break
        drop=[c for c in cells if c['seed']==seed];dropped+=drop;cells=[c for c in cells if c['seed']!=seed]
    write(OUT/'main_reductions.json',dict(at=now(),dropped=dropped,pending_mandatory_s=pending))
    write(OUT/'queue/100_formal_main.json',cells)
    write(OUT/'formal_submitted.json',dict(at=now(),epoch_s=time.time(),before_deadline=time.time()<=T0+4.25*3600,
        flags=selected,code=str(code),cells=len(cells),seeds=sorted({c['seed'] for c in cells}),queue='100_formal_main.json'))
    if (OUT/'held_optional_submission.json').exists():
        write(OUT/'queue/110_optional_decision.json',read(OUT/'held_optional_submission.json'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['framework','early','ablation','canonical'],required=True);a=p.parse_args()
    result=globals()[a.part]();write(OUT/'checks'/('decision_'+a.part)/'run.json',dict(status='OK',at=now(),decision=result))
    print(a.part,'decision saved',flush=True)

if __name__=='__main__':main()
