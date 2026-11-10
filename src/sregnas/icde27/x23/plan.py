
import argparse
import subprocess
from pathlib import Path
from .common import ROOT,OUT,PYTHON,T0,read,write,now,X20_FLAGS

def cell(dataset,method='SREGNAS',category='main',seed=42,pool=False,code=None,flags=None,expected_s=None,**extra):
    from sregnas.icde27.x18 import plan
    plan.OUT=OUT
    c=plan.pool_cell(dataset,method,'V2' if method=='SREGNAS' else None,category=category,**extra) if pool else plan.nas_cell(dataset,method,'V2' if method=='SREGNAS' else None,seed=seed,category=category,**extra)
    name=c['cell_id'].replace('x18v2_','x23_')
    rid=f'expX23_{method}-{category}_{dataset}_E1_P{c["P"]}S{c["S"]}_Q{c["Q"]}K{c["K"]}_seed{seed}_{c["record_type"]}_20261105_{name}'
    out=OUT/'runs'/rid
    argv=plan.command(c,out)
    if pool:argv[2]='sregnas.icde27.x23.replay'
    else:argv[argv.index('--experiment-id')+1]='X23'
    selected=dict(X20_FLAGS if method=='SREGNAS' else {'SREGNAS_X20_ORIGIN_PREFIX_CACHE':'1'})
    selected.update(flags or {})
    if method=='FastGNAS':
        selected.pop('SREGNAS_X23_CANONICAL_INPUT',None)
        selected.pop('SREGNAS_X23_CANONICAL_EXISTING_LARGE',None)
    if method!='SREGNAS' and dataset in ('Reddit','Products') and selected.get('SREGNAS_X23_CANONICAL_EXISTING_LARGE')=='1':
        selected.pop('SREGNAS_X23_CANONICAL_INPUT',None)
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
                c.update(phase='3.main',fallback=fallback,reduction_rank=2 if seed==44 else 1 if seed==43 else 0)
                cells.append(c)
    return cells
