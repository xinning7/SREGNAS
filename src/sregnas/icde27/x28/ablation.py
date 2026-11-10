
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import time
from .common import read,write

GLOBAL='SREGNAS_X28_GLOBAL'
REBUILD='SREGNAS_X28_REBUILD'

def options(cell):
    if not cell.get('x28_fixed_pool'):raise ValueError('X28 is fixed-pool only')
    if cell.get('kind')!='replay' or cell.get('method')!='SREGNAS':raise ValueError('X28 requires canonical fixed pool')
    return bool(cell.get('x28_global',False)),bool(cell.get('x28_rebuild',False))

def rebuild_seed(base_seed,index):
    from sregnas.training.minibatch import _hash_seed
    return _hash_seed(int(base_seed),'x28_rebuild',int(index))

def materialize_one(rowptr,permutations,membership,part,caps,device='cuda:0'):
    pass
    import numpy as np
    import torch
    nodes=np.flatnonzero(membership==part).astype(np.int64)
    degree=np.diff(rowptr)[nodes]
    local_ptr=np.r_[0,np.cumsum(degree)]
    from sregnas.data.sampling import _ragged_positions
    positions=_ragged_positions(rowptr[nodes],degree)
    ptr=torch.as_tensor(local_ptr,dtype=torch.long,device=device)
    target=torch.repeat_interleave(torch.arange(len(nodes),device=device),torch.as_tensor(degree,device=device))
    member=torch.as_tensor(membership,dtype=torch.long,device=device)
    local=torch.full((len(membership),),-1,dtype=torch.long,device=device)
    local[torch.as_tensor(nodes,device=device)]=torch.arange(len(nodes),device=device)
    result=[]
    for permutation,cap in zip(permutations,caps):
        neighbor=torch.as_tensor(permutation[positions],dtype=torch.long,device=device)
        inside=member[neighbor]==part
        prefix=torch.cat((ptr.new_zeros(1),inside.long().cumsum(0)))
        rank=prefix[1:]-1-prefix[ptr[:-1]][target]
        keep=inside & (rank<cap)
        result.append(torch.stack((local[neighbor[keep]],target[keep])).cpu().contiguous())
    return result

def execute(cell,cfg,out):
    import numpy as np
    import torch
    from sregnas.icde27 import ledger
    from sregnas.icde27.p2p6.resources import mark_stage
    from sregnas.icde27.p2p6.plan import selected_rows
    from sregnas.icde27.x18.cold_prep import build
    from sregnas.icde27.x18.shared_graph import Store
    from sregnas.execution.startup import partition_receipt
    from sregnas.execution.tasks import _cell_spec
    from sregnas.core.contracts import TaskKey
    from sregnas.icde27.x23.pool import execute as pool_execute
    global_,rebuild=options(cell)
    ledger.start_clock('X28 before reading local graph; X25 boundary')
    mark_stage(out,'prep/graph_and_neighborhood')
    manifest,_,arrays=build(cfg,out/'prep',return_arrays=True,x28_global=global_)
    cfg._partition_receipt=partition_receipt(out/'prep/graph/manifest.json')
    paths=[s['path'] for s in manifest['shards']]
    inputs=None
    if rebuild:


        source=Path(manifest['x22_canonical_manifest']);meta=read(source)
        csr={k:str(Path(meta['arrays'][k]['path']) if Path(meta['arrays'][k]['path']).is_absolute()
                   else source.parent/meta['arrays'][k]['path']) for k in ('rowptr','col')}
        member=arrays['memberships'][manifest['partition_selection']['selected']]
        if global_:member=np.zeros_like(member)
        store=Store(out/'rebuild_inputs');m=store.array(member)
        inputs=dict(csr,membership=m['_x18_array'],base_seed=cfg.graph.permutation_seed,caps=[cfg.graph.fanout_a,cfg.graph.fanout_b],global_domain=global_)
        write(out/'rebuild_inputs.json',inputs)
    del arrays
    groups={g:[] for g in cell['physical_gpus']}
    for index,row in enumerate(selected_rows(cell)):
        shard=0 if cell.get('smoke') else index%8
        gpu=cell['physical_gpus'][index%cell['P']]
        spec=_cell_spec(key=TaskKey('P2P6:system:proxy',index//8,shard,row['fingerprint']),mode='proxy',shard=shard,
            architecture=json.loads(row['architecture_json']),graph_path=paths[shard],cfg=cfg,physical_gpu=gpu)
        spec.update(p2p6_index=index,p2p6_neighborhood_hash=None,p2p6_rebuild=None,
                    checkpoint_path=str(out/'checkpoints'/(row['fingerprint']+'.pt')),
                    x28_rebuild_inputs=inputs,x28_global=global_,x28_rebuild=rebuild)
        groups[gpu].append(spec)
    write(out/'specs.json',groups)
    output,started,ended=pool_execute(out,cell,groups)
    records=sorted([r for w in output for r in w['rows']],key=lambda r:r['index'])
    return dict(rows=records,spans=[s for w in output for s in w['spans']],controller_updates=0,
        neighborhood_policy='X25 degree-group permutations; global drops partition filtering; rebuilt uses H(42,x28_rebuild,candidate_index)+layer',
        assertions=dict(worker_processes_complete=not any(w.get('process_failed') for w in output),
            rebuild_structure_identical=all(w['structure_assertions'] for w in output)),
        x28_global=global_,x28_rebuild=rebuild)

def edge_stats(edges,n):
    import torch
    from sregnas.icde27.p2p6.canonical import tensor_hash
    indegrees=[torch.bincount(e[1].cpu(),minlength=n) for e in edges]
    return dict(hash=tensor_hash(edges),messages=[int(e.shape[1]) for e in edges],
        indegree_histograms=[torch.bincount(d).tolist() for d in indegrees],
        indegree_hash=tensor_hash(indegrees))

def run_worker(job):
    import numpy as np
    import torch
    from sregnas.training.fullbatch import PersistentCellRuntime,run_cell
    from sregnas.execution.startup import assert_cell_start
    from sregnas.icde27 import ledger
    from sregnas.icde27.x23.pool import worker_ready,worker_done
    from sregnas.icde27.p2p6.profiling import capture
    from sregnas.icde27.x18.cold_prep import segmented_permutation
    from sregnas.training.fixed_proxy import index_state,payload_index
    cell,gpu=job['cell'],job['gpu'];global_,rebuild=options(cell)
    torch.set_num_threads(1);assert torch.cuda.is_available();device=torch.device('cuda:0')
    full=PersistentCellRuntime();result=dict(rows=[],spans=[],structure_assertions=True)
    audit=cell.get('smoke',False);observed=[]
    if audit:
        from sregnas.icde27.x19 import proxy
        original=proxy.prebuild
        def inspected(*args,**kwargs):
            plan=original(*args,**kwargs)
            observed.append(dict(steps=len(plan),train_pool_size=int(args[1].numel()),targets=[int(t.numel()) for t,b in plan],
                messages_by_step=[[int(e.shape[1]) for e in b.edges] for t,b in plan]))
            return plan
        proxy.prebuild=inspected
    rebuild_arrays=None
    worker_ready(job)
    torch.cuda.reset_peak_memory_stats()
    pool_allocated_peak=0.;pool_reserved_peak=0.
    for spec in job['specs']:
        started=time.perf_counter();assert_cell_start(spec);materialization_s=None;stats=None;before=None
        observed.clear();seed=None;pre_train_allocated_peak=0.;pre_train_reserved_peak=0.
        try:
            with capture(f"worker{gpu}_{spec['fingerprint']}_proxy"):
                data,_,_=full.load_payload(spec['graph_path'],'proxy')
                if audit:before=edge_stats([data['edge_index_layer1'],data['edge_index_layer2']],len(data['x']))
                def call(candidate,runtime):
                    nonlocal materialization_s,rebuild_arrays,stats,seed,pre_train_allocated_peak,pre_train_reserved_peak
                    if rebuild:
                        begin=time.perf_counter();inputs=spec['x28_rebuild_inputs']
                        if rebuild_arrays is None:
                            rebuild_arrays={k:np.load(inputs[k],mmap_mode='r') for k in ('rowptr','col','membership')}
                        seed=rebuild_seed(inputs['base_seed'],spec['p2p6_index'])
                        ptr,col,member=(rebuild_arrays[k] for k in ('rowptr','col','membership'))
                        perms=[segmented_permutation(ptr,col,seed+l) for l in range(2)]
                        edges=materialize_one(ptr,perms,member,0 if global_ else spec['shard'],inputs['caps'])
                        prepared=index_state(edges,len(data['x']))
                        data.pop('_proxy_fixed_runtime_index',None);data.pop('_proxy_fixed_index_load_s',None)
                        data['edge_index_layer1'],data['edge_index_layer2']=[e.to(device) for e in edges]
                        data['proxy_fixed_index']=prepared
                        data.update(msg1=edges[0].shape[1],msg2=edges[1].shape[1],expected_e_msg_sub=sum(e.shape[1] for e in edges))
                        payload_index(data,[data['edge_index_layer1'],data['edge_index_layer2']],device)
                        torch.cuda.synchronize();materialization_s=time.perf_counter()-begin
                        del perms,edges,prepared



                    pre_train_allocated_peak=torch.cuda.max_memory_allocated()/2**30
                    pre_train_reserved_peak=torch.cuda.max_memory_reserved()/2**30
                    return run_cell(candidate,runtime=runtime)
                row=asdict(ledger.call(spec,call,runtime=full))
                torch.cuda.synchronize()
                if audit:
                    stats=edge_stats([data['edge_index_layer1'],data['edge_index_layer2']],len(data['x']))
                    if rebuild:assert stats['indegree_hash']==before['indegree_hash'] and stats['hash']!=before['hash']
        except Exception as exc:
            import traceback;traceback.print_exc()
            from sregnas.icde27.p2p6.resources import is_oom
            row=dict(status='OOM' if is_oom(repr(exc)) else 'incomplete',error=repr(exc),failure_stage='pool/proxy')
            if isinstance(exc,AssertionError):result['structure_assertions']=False
        ended=time.perf_counter()
        pool_allocated_peak=max(pool_allocated_peak,pre_train_allocated_peak,torch.cuda.max_memory_allocated()/2**30)
        pool_reserved_peak=max(pool_reserved_peak,pre_train_reserved_peak,torch.cuda.max_memory_reserved()/2**30)
        row.update(fingerprint=spec['fingerprint'],index=spec['p2p6_index'],worker=gpu,gpu=gpu,shard=spec['shard'],
            wall_s=ended-started,epochs=spec['epochs'],materialization_s=materialization_s,rebuild_seed=seed,
            peak_allocated_GiB=pool_allocated_peak,peak_reserved_GiB=pool_reserved_peak,
            pre_train_peak_allocated_GiB=pre_train_allocated_peak,
            memory_peak_scope='cumulative pool peak, preserving pre-run-cell H2D/materialization before its internal reset',
            neighborhood_hash=stats['hash'] if stats else None,audit=dict(before=before,after=stats,batches=list(observed)) if audit else None)
        result['rows'].append(row);result['spans'].append(dict(worker=gpu,logical_worker=spec['shard'],start_perf_s=started,end_perf_s=ended))
        if row['status'] not in ('OK','OOM','DIVERGE'):return worker_done(job,result,2)
    return worker_done(job,result,0)
