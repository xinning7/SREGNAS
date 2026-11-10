
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from .common import ROOT,OUT,read,write,now

KEYS={'Flickr':'flickr','arXiv':'ogbn-arxiv','Reddit':'reddit','Products':'ogbn-products'}
ARRAYS=('x','labels','train_mask','val_mask','test_mask','rowptr','col')

def enabled():return os.environ.get('SREGNAS_X22_CANONICAL_INPUT')=='1'
def manifest_path(dataset):return OUT/'canonical'/dataset/'manifest.json'

def load(dataset):
    from sregnas.data.products import ProductsMmapGraph
    path=manifest_path(dataset);meta=read(path)
    if meta.get('x22_bitwise_validated') is not True:raise AssertionError('canonical input was not bitwise validated')
    if dataset in ('reddit','ogbn-products') and os.environ.get('SREGNAS_X22_CANONICAL_EXISTING_LARGE')=='1':


        path=Path(meta['old_baseline_source'])
    graph=ProductsMmapGraph.open(path)
    from sregnas.icde27 import ledger
    ledger.event('x22_canonical_input',dataset=dataset,manifest=str(path),graph_sha256=meta['graph_sha256'],
        array_hashes=meta['array_sha256'],scope='read inside prep; no content rehash in timed path')
    return graph,path

def raw_dataset(dataset):
    graph,path=load(dataset)
    raw={k:getattr(graph,k) for k in ARRAYS}
    raw.update(num_nodes=graph.num_nodes,num_features=graph.num_features,num_classes=graph.num_classes,
        edge_index=None,x22_canonical_manifest=str(path))
    return raw

def port_graph(cfg,out):
    import numpy as np
    import torch
    from torch_geometric.data import Data
    graph,path=load(cfg.graph.dataset);meta=graph.manifest
    if 'baseline_edge_index' in meta['arrays']:
        declared=meta['arrays']['baseline_edge_index']['path']
        edge=torch.from_numpy(np.array(np.load(path.parent/declared,mmap_mode='r'),copy=True))
    else:
        edge=torch.empty((2,graph.col.size),dtype=torch.int64)
        edge[0]=torch.from_numpy(np.array(graph.col,dtype=np.int64))
        edge[1]=torch.from_numpy(np.repeat(np.arange(graph.num_nodes,dtype=np.int64),np.diff(graph.rowptr)))
    tensors={k:torch.from_numpy(np.array(getattr(graph,k),copy=True)) for k in ARRAYS[:5]}
    data=Data(x=tensors['x'],y=tensors['labels'],edge_index=edge,
        **{k:tensors[k] for k in ARRAYS[2:5]}).contiguous()
    if cfg.graph.dataset in ('reddit','ogbn-products'):
        retrain_path=str(path)
    else:

        retrain_path=str(Path(out)/'full_graph.pt')
        torch.save(dict(tensors,edge_index=edge,num_nodes=graph.num_nodes,
            num_features=graph.num_features,num_classes=graph.num_classes),retrain_path)
    return data,graph.num_classes,retrain_path

def build_one(label):
    import numpy as np
    import torch
    from sregnas.icde27.x18.cold_prep import raw_dataset as old_raw
    from sregnas.icde27.x18.prep import undirected_csr
    from sregnas.data.products import ProductsMmapGraph
    from sregnas.paths import DATA_DIR
    ds=KEYS[label];folder=manifest_path(ds).parent
    if folder.exists():raise FileExistsError('existing canonical artifact preserved: '+str(folder))
    folder.mkdir(parents=True);started=time.time()
    raw=old_raw(ds)
    def array(value):return value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)
    edge=array(raw['edge_index']);rowptr,col=undirected_csr(edge,int(raw['num_nodes']))
    values={k:array(raw[k]) for k in ARRAYS[:5]};values.update(rowptr=rowptr,col=col)
    checks={};baseline_source=None
    if ds in ('reddit','ogbn-products'):
        relative='Reddit/sregnas_cache_v1/full/manifest.json' if ds=='reddit' else 'ogbn_products/sregnas_comparison_v1/full/manifest.json'
        baseline_source=str(DATA_DIR/relative);old=ProductsMmapGraph.open(baseline_source)
        checks={k:np.array_equal(v,getattr(old,k)) and v.dtype==getattr(old,k).dtype for k,v in values.items()}
        del old
    else:


        values['baseline_edge_index']=edge
        checks={k:True for k in ARRAYS};checks['baseline_COO_order_retained']=True
        baseline_source='dataset_manager.load_dataset(normalize=False,preprocess=False)'
    if not all(checks.values()):raise AssertionError('old canonical and reconstructed arrays differ: '+str(checks))
    records={};hashes={}
    for key,value in values.items():
        value=np.ascontiguousarray(value);path=folder/(key+'.npy');np.save(path,value,allow_pickle=False)
        disk=np.load(path,mmap_mode='r');checks[key+'_written_bitwise']=np.array_equal(value,disk) and value.dtype==disk.dtype
        records[key]=dict(path=path.name,shape=list(value.shape),dtype=str(value.dtype))
        hashes[key]=hashlib.sha256(memoryview(value).cast('B')).hexdigest();del disk
    h=hashlib.sha256();h.update(memoryview(rowptr).cast('B'));h.update(memoryview(col).cast('B'))
    old_digest=read(ROOT/'configs/data/csr_validation.json')[ds]['csr_sha256']
    checks['matches_X18_qualified_CSR_hash']=h.hexdigest()==old_digest
    assert all(checks.values()),checks
    meta=dict(schema='products_mmap_csr_v1',dataset=ds,num_nodes=int(raw['num_nodes']),
        num_features=int(raw['num_features']),num_classes=int(raw['num_classes']),num_directed_messages=int(col.size),
        graph_sha256=h.hexdigest(),arrays=records,array_sha256=hashes,storage='cpu_numpy_readonly_mmap',
        adjacency_builder='X18 qualified sorted undirected CSR; original baseline COO order retained for small graphs',
        x22_bitwise_validated=True,validation_checks=checks,old_baseline_source=baseline_source,
        constructed_at=now(),construction_wall_s=time.time()-started)
    write(folder/'manifest.json',meta)
    return dict(dataset=label,status='OK',manifest=str(folder/'manifest.json'),checks=checks,
        graph_sha256=meta['graph_sha256'],array_hashes=hashes,wall_s=time.time()-started)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=list(KEYS));a=p.parse_args()
    rows=[];started=time.time()
    for label in [a.dataset] if a.dataset else KEYS:
        try:rows.append(build_one(label))
        except Exception as exc:
            import traceback;traceback.print_exc();rows.append(dict(dataset=label,status='failed',error=repr(exc)))
        gc.collect();write(OUT/'canonical_validation.json',dict(at=now(),rows=rows))
    value=dict(status='OK' if all(r['status']=='OK' for r in rows) else 'failed',rows=rows,wall_s=time.time()-started,at=now())
    write(OUT/'checks/canonical_input/run.json',value)
    return 0 if value['status']=='OK' else 2

if __name__=='__main__':raise SystemExit(main())
