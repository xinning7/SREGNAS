
from copy import deepcopy
from pathlib import Path
from .common import ROOT,OUT,PYTHON,VERSION,DATASETS,configuration,read,write


def pool_cell(dataset,method,variant=None,category='fixed_pool',P=8,**extra):
    cid='x18v2_'+'_'.join(map(str,filter(lambda x:x is not None,(category,method,variant,dataset,f'P{P}'))))
    if extra.get('degree'):cid+=f'_d{extra["degree"]}'
    job=dict(experiment_id='X18',method=method,dataset=dataset,kind='replay',P=P,count=32,
        executor='G3',domain='local',cache_mode='cached',record_type='fixed_pool',
        cached_sampler=method in ('AP','PAS-PL','GraphNAP'),
        collect_search_garbage=method=='AP' or method=='PAS-PL' and dataset in ('Reddit','Products'),
        cell_id=cid,variant=variant,version=VERSION,**extra)
    if method=='PAS-PL':job['capacity_probe']=True
    path=OUT/'jobs'/(cid+'.json');write(path,job)
    from sregnas.icde27.p2p6.plan import logical_shards
    logical=logical_shards(job)
    return dict(cell_id=cid,category=category,dataset=dataset,method=method,variant=variant,kind='replay',
        job=str(path),P=P,S=logical if logical is not None else 'NA',Q=32,K=0,seed=42,record_type='fixed_pool',timeout_seconds=7200,version=VERSION)


def nas_cell(dataset,method,variant=None,seed=42,category='main',partition=None,**extra):
    from sregnas.icde27.x15.plan import source_for
    source=source_for(ROOT,dataset,'AP' if method in ('PAS-PL','GraphNAP') else method)
    cfg=deepcopy(read(source/'run.json')['config']);cfg['search'].update(q=400,base_seed=seed);cfg['fixed_x']=8
    cid=f'x18v2_{category}_{method}_{dataset}_s{seed}'+(f'_{variant}' if variant else '')+(f'_{partition}' if partition else '')
    if partition:
        cfg['graph']['partition_strategy']=partition;cfg['runtime']['require_fi_gpu']=False
    if extra.get('E') is not None:cid+=f'_E{extra["E"]}'
    if extra.get('N') is not None:
        n=extra['N'];cid+=f'_N{n}';cfg['graph']['num_gpus']=n
        cfg['selection']['k']=8//n
        cfg['runtime'].update(system_gpus=[0,1],baseline_gpus=[0,1],require_native_topology=n==2,
            topology_mode='native' if n==2 else f'oversubscribed_{n//2}x')
    cfg['name']='x18_'+cid.replace('-','_')
    path=OUT/'configs'/(cid+'.json');write(path,cfg)
    return dict(cell_id=cid,category=category,dataset=dataset,method=method,variant=variant,kind='nas',
        config=str(path),P=2 if extra.get('N') is not None else 8,S=cfg['graph']['num_gpus'],Q=20 if category=='graphnap_profile' else 400,K=8,seed=seed,
        source_run_id=source.name,record_type='profile' if category=='graphnap_profile' else 'full_nas',
        timeout_seconds=10800,version=VERSION,**extra)


def command(cell,out):
    if cell['kind']=='replay':
        return [PYTHON,'-m','sregnas.icde27.x18.replay','--job',cell['job'],'--out',str(out),
                '--max-wall-seconds',str(cell['timeout_seconds'])]
    cmd=[PYTHON,'-m','sregnas.comparison','--method',cell['method'],'--config',cell['config'],
        '--hardware',str(ROOT/'configs/hardware/a6000.yaml'),'--data-dir',str(ROOT/'data'),'--out',str(out),
        '--allow-gpu','--icde27-record','--canonical-winner-tie','--pi-softmax-max-fix',
        '--experiment-id','X18','--record-type',cell['record_type'],'--max-wall-seconds',str(cell['timeout_seconds'])]
    cmd+=['--x18-experiment']
    if cell['method']=='SREGNAS':
        cmd+=['--s0-parallel','--x18-memory-prep']
        if cell.get('variant'):
            e,b=configuration(cell['variant'],cell['dataset']);e=cell.get('E',e)
            cmd+=['--proxy-steps',str(e)]
            if b is not None:cmd+=['--proxy-batch-targets',str(b),'--proxy-ap-targets']
    elif cell['method']=='FastGNAS':cmd+=['--native-reset-architecture-rng','--native-config',str(ROOT/'configs/icde27/fastgnas_native.yaml')]
    else:cmd+=['--cached-sampler']
    if cell['method']=='PAS-PL':
        cmd+=['--paspl-full-nas']
        if cell['dataset'] in ('Reddit','Products'):cmd+=['--paspl-collect-search-garbage']
    if cell['method']=='AP' and cell['dataset'] in ('Reddit','Products'):
        cmd+=['--collect-search-garbage']
    if cell.get('fanout5'):cmd+=['--x16-paspl-fanout5']
    if cell['category']=='graphnap_profile':cmd+=['--max-search-candidates','20']
    if cell.get('quality_only'):cmd+=['--quality-only']
    return cmd


def stage1_pools():
    return [pool_cell(ds,'SREGNAS',v) for ds in DATASETS for v in ('V0','V1','V2','V3','V4')]+[
        pool_cell(ds,'PAS-PL') for ds in DATASETS]
