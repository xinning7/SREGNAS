
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

from .common import ROOT,OUT,read,write


def undirected_csr(edge_index,num_nodes):
    pass
    import numpy as np
    edge_index=np.asarray(edge_index,dtype=np.int64)
    if edge_index.ndim!=2 or edge_index.shape[0]!=2:raise ValueError('edge_index must have shape [2, E]')
    src,dst=edge_index
    valid=(src!=dst)&(src>=0)&(dst>=0)&(src<num_nodes)&(dst<num_nodes)
    src,dst=src[valid],dst[valid]
    packed=np.concatenate((src*np.int64(num_nodes)+dst,dst*np.int64(num_nodes)+src))
    packed.sort(kind='stable')
    keep=np.empty(len(packed),dtype=bool)
    if len(packed):
        keep[0]=True;keep[1:]=packed[1:]!=packed[:-1]
    packed=packed[keep]
    counts=np.bincount(packed//np.int64(num_nodes),minlength=num_nodes)
    col=(packed%np.int64(num_nodes)).astype(np.int32,copy=False)
    rowptr=np.empty(num_nodes+1,dtype=np.int64);rowptr[0]=0
    np.cumsum(counts,out=rowptr[1:])
    return rowptr,col


def layer_permutations(rowptr,col,base_seed,num_layers=2):
    from sregnas.data.sampling import shuffle_csr_segments

    with ThreadPoolExecutor(max_workers=min(2,num_layers)) as pool:
        return list(pool.map(lambda layer:shuffle_csr_segments(rowptr,col,base_seed+layer),range(num_layers)))


def verify(config,source,out):
    import numpy as np
    import torch
    from sregnas.config import load_config
    from sregnas.data.graph import build_graph_stage
    from sregnas.icde27 import ledger
    torch.set_num_threads(1)
    cfg=load_config(config)
    started=time.perf_counter();new=build_graph_stage(cfg,out)
    manifests=list(source.glob('pipeline/**/graph/manifest.json'))
    if len(manifests)!=1:raise ValueError(f'expected one source graph manifest: {manifests}')
    original=read(manifests[0]);checks={}
    checks['membership_bitwise']=np.array_equal(np.load(manifests[0].parent/'membership.npy'),np.load(out/'graph/membership.npy'))
    checks['permutation_hashes_identical']=original['permutation_sha256']==new['permutation_sha256']
    checks['FI_decision_identical']=original.get('partition_selection')==new.get('partition_selection')
    if original.get('products_graph_manifest_path'):
        old_manifest=Path(original['products_graph_manifest_path']);new_manifest=Path(new['products_graph_manifest_path'])
        for name in ('rowptr','col'):
            checks[f'full_graph_{name}_bitwise']=np.array_equal(
                np.load(old_manifest.parent/(name+'.npy'),mmap_mode='r'),
                np.load(new_manifest.parent/(name+'.npy'),mmap_mode='r'))
    for old,current in zip(original['shards'],new['shards']):
        a=torch.load(old['path'],map_location='cpu',weights_only=False)
        b=torch.load(current['path'],map_location='cpu',weights_only=False)
        for name in ('x','labels','train_mask','val_mask','node_ids','edge_index_layer1','edge_index_layer2'):
            checks[f'shard{old["shard"]}_{name}_bitwise']=torch.equal(a[name],b[name])
    value=dict(status='OK' if all(checks.values()) else 'failed',checks=checks,source=str(source),
        wall_s=time.perf_counter()-started,implementation='stable integer CSR sort; two independent frozen permutation streams, ordered results',
        version='x18_prep_v1',training_changed=False)
    write(out/'equivalence.json',value)
    if not all(checks.values()):raise AssertionError('preparation optimization changed output')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();verify(a.config,a.source,a.out)
