
import copy
import hashlib
import json
import os
from pathlib import Path
import time
from .recorder import dump, sha
from .x10 import ROOT, QUEUE, DEADLINE, run_job, save


def upper_edges(rowptr, col):
    import numpy as np
    rows=np.repeat(np.arange(len(rowptr)-1,dtype=np.int64),np.diff(rowptr))
    keep=rows<col
    return rows[keep],np.asarray(col[keep],dtype=np.int64)


def sampled_csr(u,v,keys,n,target,original_degree):
    import numpy as np
    from sregnas.data.products import edge_index_to_undirected_csr
    selected=keys < target/original_degree
    return edge_index_to_undirected_csr(np.stack([u[selected],v[selected]]),n)


def prepare_graphs():
    import numpy as np
    from sregnas.data.products import ProductsMmapGraph
    from sregnas.data.partition_selection import balanced_random_membership
    from sregnas.data.sampling import shuffle_csr_segments
    source=ROOT/'data/Reddit/sregnas_cache_v1'
    graph=ProductsMmapGraph.open(source/'full/manifest.json')
    directory=QUEUE/'graphs';directory.mkdir(exist_ok=False)
    membership=balanced_random_membership(graph.num_nodes,8,42)
    np.save(directory/'membership.npy',membership)
    member_hash=sha(directory/'membership.npy')
    u,v=upper_edges(graph.rowptr,graph.col)
    assert 2*len(u)==len(graph.col), 'source graph must be loop-free and symmetric'

    keys=np.random.default_rng(42).random(len(u),dtype=np.float32)
    original_degree=len(graph.col)/graph.num_nodes
    records=[]
    for target in (12,25,50,100,200,'original'):
        started=time.perf_counter()
        root=directory/str(target);(root/'full').mkdir(parents=True)
        manifest=copy.deepcopy(graph.manifest)
        for name,record in manifest['arrays'].items():
            record['path']=str((source/'full'/record['path']).resolve())
        if target=='original':
            rowptr,col=graph.rowptr,graph.col
        else:
            rowptr,col=sampled_csr(u,v,keys,graph.num_nodes,target,original_degree)
            for name,arr in (('rowptr',rowptr),('col',col)):
                np.save(root/'full'/(name+'.npy'),arr)
                manifest['arrays'][name]=dict(path=name+'.npy',shape=list(arr.shape),dtype=str(arr.dtype))
        h=hashlib.sha256();h.update(memoryview(rowptr).cast('B'));h.update(memoryview(col).cast('B'))
        manifest.update(num_directed_messages=len(col),graph_sha256=h.hexdigest())
        dump(root/'full/manifest.json',manifest)
        permutations={}
        for layer in (1,2):
            path=root/f'seed42_layer{layer}.npy'
            if target=='original':
                path.symlink_to(source/path.name)
            else:
                np.save(path,shuffle_csr_segments(rowptr,col,42+layer-1))
            permutations[path.name]=sha(path)
        ready=dict(graph_manifest=str(root/'full/manifest.json'),permutation_sha256=permutations,
            num_nodes=graph.num_nodes,num_features=graph.num_features,num_classes=graph.num_classes)
        dump(root/'ready.json',ready)
        record=dict(degree=str(target),target_degree=original_degree if target=='original' else target,
            actual_mean_degree=len(col)/graph.num_nodes,directed_edges=len(col),undirected_edges=len(col)//2,
            graph_manifest=str(root/'full/manifest.json'),graph_root=str(root),graph_sha256=h.hexdigest(),
            membership_path=str(directory/'membership.npy'),membership_sha256=member_hash,
            generation_s=time.perf_counter()-started,num_nodes=graph.num_nodes,
            input_features_labels_masks_shared=True,seed=42,
            sampling='numpy.default_rng(42), one float32 uniform key per undirected edge; nested Bernoulli keep p=target/original_degree')
        records.append(record);dump(directory/'manifest.json',records)
        print('GRAPH '+json.dumps(record),flush=True)
    return records


def load_data(manifest):
    import numpy as np
    import torch
    from torch_geometric.data import Data
    from sregnas.data.products import ProductsMmapGraph
    graph=ProductsMmapGraph.open(manifest)
    edge=torch.empty((2,graph.col.size),dtype=torch.int64)
    edge[0]=torch.from_numpy(np.array(graph.col,dtype=np.int64))
    edge[1]=torch.from_numpy(np.repeat(np.arange(graph.num_nodes),np.diff(graph.rowptr)))
    arrays={name:torch.from_numpy(np.array(getattr(graph,name),copy=True))
        for name in ('x','labels','train_mask','val_mask','test_mask')}
    data=Data(x=arrays.pop('x'),y=arrays.pop('labels'),edge_index=edge,num_nodes=graph.num_nodes,**arrays)
    return data.contiguous(),graph


def install(job,out):
    pass
    import numpy as np
    from sregnas.data import reddit, partition_selection
    from sregnas.baselines.graphnap import runtime
    from sregnas.baselines.fastgnas.data import partition, data_utils
    root=Path(job['graph_root'])
    os.environ['REDDIT_GRAPH_CACHE']=str(root)
    reddit.prepare_reddit_graph=lambda root=None:json.loads((Path(job['graph_root'])/'ready.json').read_text())
    original_select=partition_selection.select_partition
    stages={}
    def select(*args,**kwargs):
        started=time.perf_counter()
        result=original_select(*args,**kwargs)
        stages.update(partition_s=time.perf_counter()-started,partition_end_perf_s=time.perf_counter())
        assert np.array_equal(result[0],np.load(job['membership_path'])), 'random membership changed'
        return result
    partition_selection.select_partition=select
    original_build=reddit.build_reddit_graph_stage
    def build(cfg,out_dir):
        started=time.perf_counter()
        result=original_build(cfg,out_dir)
        ended=time.perf_counter()
        stages.update(graph_stage_s=ended-started,
            graph_read_and_permutation_check_s=stages['partition_end_perf_s']-stages['partition_s']-started,
            shard_materialization_s=ended-stages['partition_end_perf_s'],
            materialization_definition='after partition through degree accounting, eight shard payloads and manifest write')
        dump(out/'shard_preparation.json',stages)
        assert result['membership_sha256']==job['membership_sha256']
        return result
    reddit.build_reddit_graph_stage=build
    def port_graph(cfg,unused_out):
        data,graph=load_data(job['graph_manifest'])
        dump(out/'data_read.json',dict(graph_manifest=job['graph_manifest'],graph_sha256=graph.manifest['graph_sha256']))
        return data,graph.num_classes,job['graph_manifest']
    runtime.prepare_graph=port_graph
    def native_data(name,root='dataset',force_reload=False,train_ratio=0.):
        from types import SimpleNamespace
        assert name.lower()=='reddit' and not force_reload and not train_ratio
        data,graph=load_data(job['graph_manifest'])
        dump(out/'data_read.json',dict(graph_manifest=job['graph_manifest'],graph_sha256=graph.manifest['graph_sha256']))
        return SimpleNamespace(num_classes=graph.num_classes,num_features=graph.num_features),data
    partition.get_dataset=data_utils.get_dataset=native_data


def run_degree(state):

    for attempt in list(state['attempts']):
        path=Path(attempt['result_path'])/'run.json'
        info=json.loads(path.read_text()) if path.exists() else {}
        prelaunch='X10 exclusive preflight failed' in attempt.get('error','') and not (path.parent/'events.jsonl').exists()
        if info.get('failure',{}).get('error')=="'normalization_fix_schema'" or prelaunch:
            source=next(j for j in state['jobs'] if j['cell_id']==attempt['cell_id'])
            repair=dict(source,cell_id=source['cell_id']+'_repair1',run_id=source['run_id']+'_repair1',
                repair_of=attempt['run_id'],repair_reason='strict idle preflight before any training; add existing wait' if prelaunch else 'missing NF provenance schema after training; tools-only bug')
            state['jobs'].append(repair);save(state);run_job(repair,state,DEADLINE-180)
    state['stage']='degree_preparation';state['degree_started_epoch_s']=time.time();save(state)
    deadline=min(time.time()+7200,DEADLINE-180)
    graphs=prepare_graphs();state['graphs']=graphs;save(state)
    settings=json.loads((QUEUE/'degree_ready.json').read_text())
    state['stage']='degree';save(state)
    for graph in graphs:
        for method in ('SREGNAS','FastGNAS','AP'):
            label=f"degree-{graph['degree']}"
            job=dict(experiment_id='X10',dataset='Reddit',method=method,kind='replay',P=8,count=32,
                executor='G3',domain='local',cache_mode='cached',record_type='fixed_pool',stage='degree',
                cell_id=f'X10_{method}_{label}',collect_search_garbage=method=='AP',
                native_fixed_random=settings.get('native_fixed_random',False),
                run_id=f'expX10_{method}-{label}_Reddit_E1_P8S{8 if method!="AP" else "NA"}_Q32K0_seed42_fixed_pool_20261026',**graph)
            state['jobs'].append(job);save(state)
            run_job(job,state,deadline)
    state['degree_ended_epoch_s']=time.time();save(state)
    costs=set()
    degree_success=set()
    for attempt in state['attempts']:
        if attempt['status']!='OK' or not attempt.get('timing_valid'):
            continue
        info=json.loads((Path(attempt['result_path'])/'run.json').read_text())
        j=info['job']
        if j['stage']=='cost':
            costs.add((j['cost_category'],j['repeat'],j['canonical_nf']))
        elif j['stage']=='degree':
            degree_success.add((j['degree'],j['method']))
    mandatory_complete=len(costs)==24 and len(degree_success)==18
    state['optional_gate']=dict(mandatory_complete=mandatory_complete,remaining_s=DEADLINE-time.time(),
        cost_success=len(costs),degree_success=len(degree_success))
    if mandatory_complete and DEADLINE-time.time()>=3600:
        state['stage']='optional_factorial';save(state)
        for domain in ('local','global'):
            for cache in ('cached','rebuild'):
                job=dict(experiment_id='X10',dataset='Products',method='SREGNAS',kind='replay',P=8,count=32,
                    executor='G3',domain=domain,cache_mode=cache,record_type='fixed_pool',stage='factorial',factorial=True,
                    cell_id=f'X10_factorial_{domain}_{cache}',
                    run_id=f'expX10_SREGNAS-{domain}-{cache}_Products_E1_P8S8_Q32K0_seed42_fixed_pool_20261026')
                state['jobs'].append(job);save(state);run_job(job,state,DEADLINE-180)
    else:
        state['optional_skipped']='mandatory stages incomplete' if not mandatory_complete else 'remaining budget below 1 h after mandatory stages'
        save(state)
