
import hashlib
from pathlib import Path
import numpy as np
from .recorder import dump, sha


def array_hash(value):
    digest=hashlib.sha256()
    digest.update(str(value.dtype).encode());digest.update(str(value.shape).encode())
    digest.update(memoryview(value).cast('B'))
    return digest.hexdigest()


def equivalent_mapping(source,reference):
    import torch
    from sregnas.data.products import ProductsMmapGraph
    from sregnas.data.validate import validate_dataset_dict
    graph=ProductsMmapGraph.open(source)
    other=ProductsMmapGraph.open(reference)
    checks={}
    for name in ('x','labels','train_mask','val_mask','test_mask','rowptr','col'):
        a,b=array_hash(getattr(graph,name)),array_hash(getattr(other,name))
        checks[name]=dict(source_sha256=a,reference_sha256=b,equal=a==b)
    assert all(row['equal'] for row in checks.values()), 'Products cached input differs from raw-prepared graph'
    edge=np.empty((2,len(graph.col)),dtype=np.int64)
    edge[0]=np.repeat(np.arange(graph.num_nodes,dtype=np.int64),np.diff(graph.rowptr))
    edge[1]=graph.col
    result={name:torch.from_numpy(np.array(getattr(graph,name),copy=True))
        for name in ('x','labels','train_mask','val_mask','test_mask')}
    result.update(edge_index=torch.from_numpy(edge),num_nodes=graph.num_nodes,
        num_features=graph.num_features,num_classes=graph.num_classes)
    validate_dataset_dict(result,'ogbn-products')
    return result,checks


def install(out):
    from sregnas.data import graph
    from .x10 import ROOT
    source=ROOT/'data/ogbn_products/sregnas_comparison_v1/full/manifest.json'


    reference=ROOT/'results/icde27/EXPX10/expX10_SREGNAS-local-cached_Products_E1_P8S8_Q32K0_seed42_fixed_pool_20261026/prep/graph/products_mmap/manifest.json'
    original=graph._load_dataset
    calls=[]
    def load(name):
        if name!='ogbn-products':
            return original(name)
        import time
        start=time.perf_counter()
        result,checks=equivalent_mapping(source,reference)
        calls.append(dict(source_manifest=str(source),source_manifest_sha256=sha(source),
            reference_manifest=str(reference),reference_manifest_sha256=sha(reference),
            arrays=checks,all_equal=True,wall_s=time.perf_counter()-start))
        dump(Path(out)/'factorial_data_equivalence.json',dict(adapter='audited_mmap_to_identical_raw_mapping_v1',
            original_cell_preserved=True,scope='CPU input loading only; original CSR/permutation builders and training unchanged',calls=calls))
        return result
    graph._load_dataset=load
