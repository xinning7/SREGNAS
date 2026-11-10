
import fcntl
import time
from pathlib import Path
from .common import ROOT,OUT,PYTHON,T0,DEV_END,FORMAL_END,BASE_FLAGS,read,write,now
from .plan import cell,main_cells

NEW_FLAGS={k:'1' for k in ('SREGNAS_X23_INCREMENTAL_RECORDS','SREGNAS_X23_LEDGER_INDEX',
    'SREGNAS_X23_POOL_PROGRESS','SREGNAS_X23_POOL_READY','SREGNAS_X23_OUTER_OBSERVER')}


def cpu_cell(name,module,args=(),expected=60):
    return dict(cell_name='x23_'+name,phase='1.qualification',code=str(ROOT),expected_s=expected,
        mandatory=True,internal_lock=False,flags={'CUDA_VISIBLE_DEVICES':'','SREGNAS_ALLOW_GPU':'0'},
        out=str(OUT/'checks'/name),argv=[PYTHON,'-m',module,*args])


def resize(c,count):
    old=c['out'];job=read(c['job']);job['count']=count;write(c['job'],job)
    c.update(Q=count,run_id=c['run_id'].replace('Q32K0',f'Q{count}K0'),out=old.replace('Q32K0',f'Q{count}K0'))
    c['argv']=[c['out'] if v==old else v for v in c['argv']]
    return c


def qualification_plan():
    reference=read(OUT/'reference_plan_effective.json');changed=[];pairs=[]
    code=ROOT/'.runtime/x23-code/release'
    for r in reference:
        c=resize(cell(r['dataset'],r['method'],category='qualify_changed',pool=True,code=code,
            flags=dict(r['flags'],**NEW_FLAGS),expected_s=r['expected_s']),16)
        c.update(workspace=':4096:8',phase='1.qualification',launch_deadline=DEV_END)
        c['argv']=[PYTHON,'-m','sregnas.icde27.x22.gated','--receipt',str(OUT/'checks/framework/run.json'),'--',*c['argv']]
        changed.append(c);pairs.append(dict(reference=r,changed=c))
    write(OUT/'qualification_pairs.json',pairs);write(OUT/'queue/020_qualify_changed.json',changed)
    checks=[];observers=[]
    for ds in ('arXiv','Products'):
        pair=[]
        for option in ('0','1'):
            flags=dict(BASE_FLAGS,SREGNAS_X20_BOOTSTRAP='1',SREGNAS_X20_DETERMINISTIC='1',SREGNAS_X22_QUALIFY_NUMERICS='1',SREGNAS_X23_OUTER_OBSERVER=option)
            c=resize(cell(ds,'AP',category='observer_'+option,pool=True,code=code,flags=flags,expected_s=100,cost_pass='control'),8)
            c.update(workspace=':4096:8',phase='2.observer',launch_deadline=DEV_END)
            c['argv']=[PYTHON,'-m','sregnas.icde27.x22.gated','--receipt',str(OUT/'checks/framework/run.json'),'--',*c['argv']]
            checks.append(c);pair.append(c)
        observers.append(pair)
    write(OUT/'observer_pairs.json',observers);write(OUT/'queue/021_observer_check.json',checks)
    write(OUT/'queue/022_decide.json',[cpu_cell('decision','sregnas.icde27.x23.decide')])


def equality(reference, changed):
    from sregnas.icde27.x22.decide import equality as x22_equality
    left=dict(reference,flags=dict(reference.get('flags',{})))
    if left['method']=='SREGNAS':



        left['flags'].pop('SREGNAS_X22_QUALIFY_NUMERICS',None)
    return x22_equality(left,changed)


def decide():
    pairs=[equality(p['reference'],p['changed']) for p in read(OUT/'qualification_pairs.json')]
    for pair in pairs:
        audits=[]
        for path in (pair['reference'],pair['changed']):
            r=read(Path(path)/'run.json') if (Path(path)/'run.json').exists() else {}
            a=r.get('assertions',{})
            audits.append(bool(a) and all(a.values()))
        pair['source_audit_and_runtime_assertions']=audits
        pair['accepted']=pair['accepted'] and all(audits)
    observer=[]
    keys=('optimizer_steps','training_target_exposures','target_node_ids','touched_training_node_ids',
        'distinct_target_nodes','distinct_touched_training_nodes','forward_input_edges','forward_layer_input_edges')
    for pair in read(OUT/'observer_pairs.json'):
        values=[];statuses=[]
        for c in pair:
            root=Path(c['out']);r=read(root/'run.json') if (root/'run.json').exists() else {};statuses.append(r.get('status'))
            values.append({(v['fingerprint'],v['worker']):{k:v[k] for k in keys}
                for p in (root/'cost_observations').glob('*.json') for v in [read(p)]})
        observer.append(dict(dataset=pair[0]['dataset'],old=pair[0]['out'],new=pair[1]['out'],statuses=statuses,
            candidates=len(values[0]),fields=list(keys),accepted=statuses==['OK','OK'] and len(values[0])==8 and values[0]==values[1]))
    cpu=read(OUT/'checks/framework/run.json') if (OUT/'checks/framework/run.json').exists() else {}
    gate_state=read(OUT/'driver_state.json').get('cells',{}).get('x23_qualification_applicability',{})
    accepted=gate_state.get('status')=='OK' and cpu.get('status')=='OK' and all(p['accepted'] for p in pairs) and all(p['accepted'] for p in observer) and time.time()<=DEV_END
    result=dict(at=now(),accepted=accepted,comparisons=pairs,observer_comparisons=observer,cpu=cpu,flags=NEW_FLAGS if accepted else {},
        reason='all five systems, records/ledger, source audit and AP observer gates passed before development cutoff' if accepted else 'a required gate failed or deadline expired; new options not enabled')
    write(OUT/'qualification_decision.json',result)
    if accepted:
        compare=[]

        for method,ds,option in [('PAS-PL','Reddit',False),('FastGNAS','Flickr',True),('PAS-PL','Reddit',True),('FastGNAS','Flickr',False)]:
            flags=dict(BASE_FLAGS)
            if option:flags.update({k:v for k,v in NEW_FLAGS.items() if k not in ('SREGNAS_X23_POOL_READY','SREGNAS_X23_OUTER_OBSERVER')})
            c=cell(ds,method,category='compare_after' if option else 'compare_before',code=ROOT/'.runtime/x23-code/release',flags=flags,expected_s=750 if method=='FastGNAS' else 360)
            c['phase']='1.compare';compare.append(c)
        write(OUT/'queue/030_before_after.json',compare)
        submit(NEW_FLAGS)
    return result


def submit(new_flags=None):
    with (OUT/'formal_submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (OUT/'formal_submitted.json').exists():return
        flags=dict(BASE_FLAGS,**(new_flags or {}));code=ROOT/'.runtime/x23-code/release' if (ROOT/'.runtime/x23-code/release/.git').exists() else ROOT/'.runtime/x23-code/reference'
        main=main_cells(code,flags=flags)
        write(OUT/'queue/100_main.json',main)
        figures=[]
        for ds in ('Flickr','arXiv'):
            c=cell(ds,'FastGNAS',category='published_fastgnas',code=code,flags=flags,expected_s=750 if ds=='Flickr' else 1020);c['phase']='3.figures';figures.append(c)
        for ds in ('Reddit','Products','arXiv','Flickr'):
            for method in ('SREGNAS','PAS-PL','AP'):
                c=cell(ds,method,category='table5',pool=True,code=code,flags=flags,expected_s=100);c['phase']='3.figures';figures.append(c)
        for graph in read(ROOT/'results/icde27/x18/v2/degree_inputs/manifest.json'):
            for method in ('SREGNAS','AP'):
                c=cell('Reddit',method,category='degree',pool=True,code=code,flags=flags,expected_s=100,**graph);c['phase']='3.figures';figures.append(c)
        for ds in ('arXiv','Products'):
            for p in (2,4):
                for method in ('SREGNAS','AP'):
                    c=cell(ds,method,category='scaling',pool=True,P=p,scaling_mode='fixed_workload',code=code,flags=flags,expected_s=150);c['phase']='3.figures';figures.append(c)
        for ds in ('arXiv','Products'):
            for mode in ('sync','control','trace'):
                c=resize(cell(ds,'PAS-PL',category='table4_'+mode,pool=True,code=code,flags=flags,expected_s=120,cost_pass=mode),8)
                job=read(c['job']);job['record_type']='profile';write(c['job'],job);c['record_type']='profile'
                old=c['out'];c['out']=old.replace('_fixed_pool_','_profile_');c['run_id']=c['run_id'].replace('_fixed_pool_','_profile_');c['argv']=[c['out'] if v==old else v for v in c['argv']]
                c['phase']='3.figures';figures.append(c)
        write(OUT/'queue/110_figures.json',figures)
        optional=[]
        for ds,expected in [('Reddit',3400),('Products',4700)]:
            c=cell(ds,'FastGNAS',category='optional_fastgnas',code=code,flags=flags,expected_s=expected);c.update(phase='4',mandatory=False);optional.append(c)
        for seed in (43,44):
            for ds in ('Reddit','Products','arXiv','Flickr'):
                c=cell(ds,'AP',category='supplemental_AP',seed=seed,code=code,flags=flags,expected_s=650 if ds in ('Reddit','Products') else 450);c.update(phase='4',mandatory=False);optional.append(c)
        write(OUT/'queue/200_optional.json',optional)
        write(OUT/'formal_submitted.json',dict(at=now(),epoch_s=time.time(),before_deadline=time.time()<=FORMAL_END,flags=flags,code=str(code),main_cells=len(main),figure_cells=len(figures)))
        write(OUT/'reservations.json',{})

        (OUT/'FINISH').write_text(now()+'\n')


def main():
    r=decide();write(OUT/'checks/decision/run.json',dict(status='OK',decision=r['accepted'],at=now()))
if __name__=='__main__':main()
