
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .shared_graph import Store
from .prep import undirected_csr

ENV='SREGNAS_X18_MEMORY_PREP'


def enabled():return os.environ.get(ENV)=='1'


def segmented_permutation(rowptr,col,seed,chunk_entries=2_000_000):
    pass
    rng=np.random.default_rng(seed)
    degree=np.diff(rowptr)
    order=np.argsort(degree,kind='stable')
    sorted_degree=degree[order]
    bounds=np.flatnonzero(np.r_[True,sorted_degree[1:]!=sorted_degree[:-1],True])
    result=np.empty_like(col)
    for left,right in zip(bounds[:-1],bounds[1:]):
        d=int(sorted_degree[left])
        if not d:continue
        step=max(1,chunk_entries//d);offset=np.arange(d,dtype=np.int64)
        for start in range(left,right,step):
            ids=order[start:min(start+step,right)]
            positions=rowptr[ids,None]+offset[None,:]
            result[positions]=rng.permuted(col[positions],axis=1)
    return result


def raw_dataset(dataset):
    from sregnas.data.graph import _load_dataset,_load_products_raw
    return _load_products_raw() if dataset=='ogbn-products' else _load_dataset(dataset)


def partitions(config,rowptr,col,raw):
    from sregnas.data.proxy_partition_mode import partition_mode
    if partition_mode() == 'proxy_aware' and config.partition_strategy == 'fi':
        from sregnas.data.proxy_partition import masked_labels,select_from_config
        y,split=masked_labels(raw['labels'],raw['train_mask'],raw['val_mask'])
        if os.environ.get('SREGNAS_X25_PARTITION_RECORDS') == '1':
            from sregnas.icde27.x25.partition import recorded_selection
            return recorded_selection(config,rowptr,col,y,split)
        return select_from_config(config,rowptr,col,y,split)
    from sregnas.data.sampling import metis_partition
    from sregnas.data.partition_selection import balanced_random_membership,partition_index
    from sregnas.icde27.s0 import ordered_map,span
    n=len(rowptr)-1;parts=config.num_gpus
    def construct(method):
        with span('partition_build',method=method):
            return metis_partition(rowptr,col,parts) if method=='metis' else balanced_random_membership(n,parts,config.partition_seed)
    candidates=dict(zip(('metis','random'),ordered_map(construct,('metis','random'))))
    labeled=np.asarray(raw['train_mask'],dtype=bool)
    if config.partition_label_scope=='train_val':labeled=labeled|np.asarray(raw['val_mask'],dtype=bool)
    def score(method):
        with span('FI_statistics',method=method):
            return partition_index(rowptr,col,np.asarray(raw['labels']),labeled,candidates[method],parts)
    scores=dict(zip(('metis','random'),ordered_map(score,('metis','random'))))
    selected=max(('metis','random'),key=lambda key:scores[key]['FI'])
    return candidates,dict(schema='partition_fi_v1',requested_strategy=config.partition_strategy,
        selected=selected,random_seed=config.partition_seed,random_rng='numpy.RandomState',
        label_scope=config.partition_label_scope,criterion='SNR*LA; cap=5; exact ties prefer METIS',
        metis_backend='torch_sparse.partition',scores=scores,test_labels_used=False)


def build(cfg,out_dir,*,both_partitions=False,return_arrays=False,x28_global=False):
    from sregnas.icde27.s0 import span,ordered_map
    from sregnas.data.graph import _per_layer_capped_edges,_tensor_rows
    from sregnas.training.fixed_proxy import add_prepared_index
    from .common import OUT
    if x28_global and os.environ.get('SREGNAS_X28_FIXED_POOL')!='1':
        raise ValueError('X28 global materialization is restricted to its fixed-pool entry')
    cfg.validate();graph_dir=Path(out_dir)/'graph';graph_dir.mkdir(parents=True,exist_ok=True)
    store=Store(graph_dir)
    with span('read_graph'):
        if os.environ.get('SREGNAS_X22_CANONICAL_INPUT')=='1':
            from sregnas.icde27.x22.canonical_input import raw_dataset as canonical_raw
            raw=canonical_raw(cfg.graph.dataset)
        else:
            raw=raw_dataset(cfg.graph.dataset)
    n=int(raw['num_nodes'])
    with span('csr_materialization'):
        if raw.get('x22_canonical_manifest'):
            rowptr,col=raw['rowptr'],raw['col']
        elif os.environ.get('SREGNAS_X19_GPU_CSR')=='1':
            from sregnas.icde27.x19.preparation import gpu_csr
            edge=raw['edge_index'].cpu().numpy() if isinstance(raw['edge_index'],torch.Tensor) else raw['edge_index']
            rowptr,col=gpu_csr(edge,n)
        else:
            edge=raw['edge_index'].cpu().numpy() if isinstance(raw['edge_index'],torch.Tensor) else raw['edge_index']
            rowptr,col=undirected_csr(edge,n)
    with span('fixed_permutations'):
        perms=[segmented_permutation(rowptr,col,cfg.graph.permutation_seed+layer) for layer in range(2)]
    memberships,choice=partitions(cfg.graph,rowptr,col,raw)
    selected=choice['selected'] if cfg.graph.partition_strategy=='fi' else cfg.graph.partition_strategy
    if selected not in memberships:raise ValueError('X18 cold path requires FI, METIS, or random partition')
    digests_path=OUT/'csr_validation.json'
    digests=json.loads(digests_path.read_text()) if digests_path.exists() else {}
    graph_digest=digests.get(os.environ.get('SREGNAS_X18_CSR_KEY',cfg.graph.dataset),{}).get('csr_sha256')
    if not graph_digest and os.environ.get('SREGNAS_X18_PREP_VALIDATION')!='1':
        raise ValueError('cold CSR must pass the untimed CPU validation before training')
    with span('full_graph_shared_memory'):
        if cfg.graph.dataset in ('reddit','ogbn-products'):
            arrays={k:store.array(raw[k]) for k in ('x','labels','train_mask','val_mask','test_mask')}
            arrays.update(rowptr=store.array(rowptr),col=store.array(col))
            records={k:dict(path=v['_x18_array'],shape=v['shape'],dtype=v['dtype']) for k,v in arrays.items()}
            full_path=graph_dir/'full_graph_shared.json'
            full_path.write_text(json.dumps(dict(schema='products_mmap_csr_v1',dataset=cfg.graph.dataset,
                num_nodes=n,num_features=int(raw['num_features']),num_classes=int(raw['num_classes']),
                num_directed_messages=len(col),graph_sha256=graph_digest or 'CPU_validation_pending',
                arrays=records,storage='RAM_only_shared_arrays',adjacency_builder='x18_v2_cold_stable_integer_csr')))
        else:
            full_edge=torch.from_numpy(np.stack((col.astype(np.int64),np.repeat(np.arange(n),np.diff(rowptr)))))
            full_path=graph_dir/'full_graph.x18-shared.json'
            store.payload(full_path,dict({k:raw[k] for k in ('x','labels','train_mask','val_mask','test_mask','num_features','num_classes')},edge_index=full_edge))
    manifests={}
    for method in (tuple(memberships) if both_partitions else (selected,)):
        membership=memberships[method]
        sampling_membership=np.zeros_like(membership) if x28_global else membership
        sampling_parts=1 if x28_global else cfg.graph.num_gpus
        directory=graph_dir/method if both_partitions else graph_dir
        directory.mkdir(parents=True,exist_ok=True)
        prebuilt_edges=None
        if os.environ.get('SREGNAS_X19_GPU_MATERIALIZE')=='1':
            from sregnas.icde27.x19.preparation import gpu_fixed_edges
            with span('fixed_edge_materialization_gpu',method=method):
                prebuilt_edges,internal=gpu_fixed_edges(rowptr,perms,sampling_membership,
                    (cfg.graph.fanout_a,cfg.graph.fanout_b),sampling_parts)
        else:
            rows=np.repeat(np.arange(n,dtype=np.int64),np.diff(rowptr))
            inside=sampling_membership[rows]==sampling_membership[col]
            internal=np.bincount(rows[inside],minlength=n)
            del rows,inside
        def materialize(part):
            with span('offline_materialization',method=method,shard=part):
                nodes=np.flatnonzero(sampling_membership==part).astype(np.int64)
                if prebuilt_edges is None:
                    first,msg1=_per_layer_capped_edges(rowptr,perms[0],sampling_membership,part,cfg.graph.fanout_a)
                    second,msg2=_per_layer_capped_edges(rowptr,perms[1],sampling_membership,part,cfg.graph.fanout_b)
                else:
                    first,second=prebuilt_edges[part];msg1=first.shape[1];msg2=second.shape[1]
                expected=int((np.minimum(internal[nodes],cfg.graph.fanout_a)+np.minimum(internal[nodes],cfg.graph.fanout_b)).sum())
                payload={k:_tensor_rows(raw[k],nodes) for k in ('x','labels','train_mask','val_mask','test_mask')}
                payload.update(node_ids=torch.from_numpy(nodes),edge_index_layer1=first,edge_index_layer2=second,
                    msg1=msg1,msg2=msg2,expected_e_msg_sub=expected,num_features=int(raw['num_features']),num_classes=int(raw['num_classes']))
                add_prepared_index(payload,shard=part)
                path=store.payload(directory/f'shard_{part}.x18-shared.json',payload)
                return dict(shard=part,nodes=len(nodes),core_nodes=len(nodes),touched_nodes=len(nodes),rho_v=1.0,
                    msg1=msg1,msg2=msg2,E_msg_sub=expected,path=path)

        shards=[materialize(part) for part in range(sampling_parts)]
        if x28_global:
            shards=[dict(shards[0],shard=part) for part in range(cfg.graph.num_gpus)]
        counts=[r['E_msg_sub'] for r in shards]
        part_choice=dict(choice,requested_strategy=method if both_partitions else cfg.graph.partition_strategy)
        manifest=dict(dataset=cfg.graph.dataset,partition_strategy=method if both_partitions else cfg.graph.partition_strategy,
            partition_selection=part_choice,hierarchical_r=cfg.graph.hierarchical_r,num_shards=cfg.graph.num_gpus,
            permutation_seed=cfg.graph.permutation_seed,permutation_algorithm='degree_group_numpy_generator_permuted_v2',
            fanout_a=cfg.graph.fanout_a,fanout_b=cfg.graph.fanout_b,f_int=5,f_ext=0,
            computed=dict(min=min(counts),med=float(np.median(counts)),max=max(counts)),shards=shards,
            full_graph_path=str(full_path),products_graph_manifest_path=str(full_path) if cfg.graph.dataset=='ogbn-products' else None,
            reddit_graph_manifest_path=str(full_path) if cfg.graph.dataset=='reddit' else None,
            shared_root=str(store.root),preparation_version='cold_memory_v2',cold_start=True,
            cached_CSR_read=bool(raw.get('x22_canonical_manifest')),cached_permutations_read=False,cached_shards_read=False,
            x19_gpu_csr=os.environ.get('SREGNAS_X19_GPU_CSR')=='1' and not bool(raw.get('x22_canonical_manifest')),
            x19_gpu_materialize=os.environ.get('SREGNAS_X19_GPU_MATERIALIZE')=='1',
            **({'x22_canonical_manifest':raw['x22_canonical_manifest']} if raw.get('x22_canonical_manifest') else {}),
            graph_content_hash_checked=False,graph_arrays_written_to_disk=False,
            **({'x28_global':True,'graph_domain':'whole_graph_cross_shard_neighbors'} if x28_global else {}))
        path=directory/'manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
        manifests[method]=dict(manifest,path=str(path))
    if both_partitions:
        root_manifest=dict(manifests[selected],partition_strategy='fi',partition_selection=choice)
        (graph_dir/'manifest.json').write_text(json.dumps(root_manifest,indent=2)+'\n')
    else:root_manifest=manifests[selected]
    if os.environ.get('SREGNAS_X19_GPU_CSR')=='1' or os.environ.get('SREGNAS_X19_GPU_MATERIALIZE')=='1':

        torch.cuda.empty_cache()
    if return_arrays:return root_manifest,manifests,dict(rowptr=rowptr,col=col,permutations=perms,memberships=memberships)
    return root_manifest
