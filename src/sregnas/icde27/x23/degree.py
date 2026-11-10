
import os

ENV='SREGNAS_X23_DEGREE_INPUT'

def raw(job, dataset):
    if dataset!='reddit':raise ValueError('X10 degree inputs are Reddit-only')
    from sregnas.data.products import ProductsMmapGraph
    from sregnas.icde27 import ledger
    from sregnas.icde27.x22.canonical_input import ARRAYS
    graph=ProductsMmapGraph.open(job['graph_manifest'])
    ledger.event('x23_degree_input',manifest=job['graph_manifest'],graph_sha256=job['graph_sha256'],degree=job['degree'],scope='same frozen X10 CSR consumed by AP; read inside prep')
    value={key:getattr(graph,key) for key in ARRAYS}
    value.update(num_nodes=graph.num_nodes,num_features=graph.num_features,num_classes=graph.num_classes,
        edge_index=None,x22_canonical_manifest=job['graph_manifest'])
    return value

def install(job):
    if os.environ.get(ENV)!='1':return
    if os.environ.get('SREGNAS_X22_CANONICAL_INPUT')!='1':raise AssertionError('common CSR mode required for X23 degree binding')
    from sregnas.icde27.x22 import canonical_input
    canonical_input.raw_dataset=lambda dataset:raw(job,dataset)
