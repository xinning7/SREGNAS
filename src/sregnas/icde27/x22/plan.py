
import argparse
import subprocess
from pathlib import Path
from .common import ROOT,OUT,PYTHON,T0,read,write,now,X20_FLAGS

def cell(dataset,method='SREGNAS',category='main',seed=42,pool=False,code=None,flags=None,expected_s=None,**extra):
    from sregnas.icde27.x18 import plan
    plan.OUT=OUT
    c=plan.pool_cell(dataset,method,'V2' if method=='SREGNAS' else None,category=category,**extra) if pool else plan.nas_cell(dataset,method,'V2' if method=='SREGNAS' else None,seed=seed,category=category,**extra)
    name=c['cell_id'].replace('x18v2_','x22_')
    rid=f'expX22_{method}-{category}_{dataset}_E1_P{c["P"]}S{c["S"]}_Q{c["Q"]}K{c["K"]}_seed{seed}_{c["record_type"]}_20261105_{name}'
    out=OUT/'runs'/rid
    argv=plan.command(c,out)
    if pool:argv[2]='sregnas.icde27.x22.replay'
    else:argv[argv.index('--experiment-id')+1]='X22'
    selected=dict(X20_FLAGS if method=='SREGNAS' else {'SREGNAS_X20_ORIGIN_PREFIX_CACHE':'1'})
    selected.update(flags or {})
    if method=='FastGNAS':
        selected.pop('SREGNAS_X22_CANONICAL_INPUT',None)
        selected.pop('SREGNAS_X22_CANONICAL_EXISTING_LARGE',None)
    if method!='SREGNAS' and dataset in ('Reddit','Products') and selected.get('SREGNAS_X22_CANONICAL_EXISTING_LARGE')=='1':
        selected.pop('SREGNAS_X22_CANONICAL_INPUT',None)
    c.update(cell_name=name,run_id=rid,out=str(out),argv=argv,code=str(code or ROOT),flags=selected,
        expected_s=expected_s or (120 if pool else 240 if method=='SREGNAS' else 360),phase=category,mandatory=True)
    return c

def main_cells(code,flags=None,fallback=False):
    cells=[]
    for seed in (42,43,44):
        for ds in ('Reddit','Products','arXiv','Flickr'):
            for method in ('SREGNAS','PAS-PL','AP') if seed==42 else ('SREGNAS','PAS-PL'):
                c=cell(ds,method,category='fallback_main' if fallback else 'main',seed=seed,code=code,flags=flags,
                    expected_s=(300 if method=='SREGNAS' else 420 if method=='PAS-PL' else 660) if ds in ('Reddit','Products') else (180 if method=='SREGNAS' else 300 if method=='PAS-PL' else 480))
                c.update(phase='2.4',fallback=fallback,reduction_rank=2 if seed==44 else 1 if seed==43 else 0)
                cells.append(c)
    return cells

def prepare(code,x18_code):
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'queue').mkdir(exist_ok=True)
    write(OUT/'fallback_queue.json',main_cells(code,fallback=True))
    diag=[]
    for method in ('SREGNAS','PAS-PL'):
        flags=dict(SREGNAS_X20_BOOTSTRAP='1',SREGNAS_X20_TRACE='1',SREGNAS_X19_BOOTSTRAP='1',SREGNAS_X19_FINE_PROFILE='1')
        c=cell('Reddit',method,category='diagnostic',code=code,flags=flags,expected_s=360)
        profile=OUT/'profiles'/(c['cell_name']+'.json');profile.parent.mkdir(exist_ok=True)
        c['argv']=["py-spy",'record','--subprocesses','--rate','50','--format','speedscope','-o',str(profile),'--',*c['argv']]
        c['phase']='1.1';diag.append(c)
    write(OUT/'queue/010_diagnostic.json',diag)
    drift=[]
    for rep in (1,2):
        for method,workspace,legacy in [('PAS-PL',':4096:8',False),('PAS-PL',None,False),('SREGNAS',':4096:8',False),('SREGNAS',None,False),('PAS-PL',None,True)]:
            label='x18' if legacy else 'workspace' if workspace else 'default'
            c=cell('Reddit',method,category=f'drift_{label}_r{rep}',code=x18_code if legacy else code,expected_s=300)
            c.update(phase='1.2',workspace=workspace,reduction_rank=1 if rep==2 else 0)
            if legacy:
                c.update(flags={},legacy_x18=True,legacy_env=str(OUT/'legacy_x18_environment.json'))
                c['argv'][c['argv'].index('--experiment-id')+1]='X18'
            drift.append(c)
    write(OUT/'queue/020_drift.json',drift)
    write(OUT/'reservations.json',{'1.3':900,'1.4':1800,'2.1':1800,'2.2':3600,'2.3':600,'2.4':9360})
    write(OUT/'preregistration.json',dict(T0=T0,T0_UTC='2026-11-04T16:23:22Z',recorded_at=now(),gpu_end=T0+8.5*3600,end=T0+9*3600,
        initial_commit='8da4440427e6fc4fa4471f58d4d5a0d93c68918e',code=str(code),legacy_x18=str(x18_code)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--code',required=True,type=Path);p.add_argument('--x18-code',required=True,type=Path)
    a=p.parse_args();prepare(a.code,a.x18_code)
