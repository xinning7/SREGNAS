
import fcntl
import os
from pathlib import Path
from .common import ROOT,OUT,LOCK,VERSION,read,write


def inputs():
    import numpy as np
    from sregnas.data.products import ProductsMmapGraph
    target=OUT/'degree_inputs';target.mkdir(parents=True,exist_ok=True)
    graphs=read(ROOT/'results/icde27/x10_queue/graphs/manifest.json')
    with LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        digests=read(OUT/'csr_validation.json')
        for row in graphs:
            path=target/f"reddit_degree_{row['degree']}_edge_index.npy"
            if not path.exists():
                graph=ProductsMmapGraph.open(row['graph_manifest'])
                edge=np.lib.format.open_memmap(path,mode='w+',dtype=np.int64,shape=(2,len(graph.col)))
                edge[0]=graph.col
                edge[1]=np.repeat(np.arange(graph.num_nodes,dtype=np.int64),np.diff(graph.rowptr))
                edge.flush();del edge
            row['cold_edge_index_path']=str(path)
            row['cold_CSR_key']='reddit_degree_'+row['degree']
            digests[row['cold_CSR_key']]=dict(csr_sha256=row['graph_sha256'],
                source=row['graph_manifest'],status='OK',scope='frozen X10 input graph exported as raw COO outside timing',version=VERSION)
        write(OUT/'csr_validation.json',digests)
        write(target/'manifest.json',graphs)
    return graphs


def install(job):
    import numpy as np
    from . import cold_prep
    original=cold_prep.raw_dataset
    path=Path(job['cold_edge_index_path'])
    if not path.exists():raise FileNotFoundError(path)
    def load(dataset):
        if dataset!='reddit':raise ValueError('degree variants are scoped to Reddit')
        value=original(dataset)
        value['edge_index']=np.load(path,mmap_mode='r')
        return value
    cold_prep.raw_dataset=load
    os.environ['SREGNAS_X18_CSR_KEY']=job['cold_CSR_key']
    if os.environ.get('SREGNAS_X23_DEGREE_INPUT')=='1':
        from sregnas.icde27.x23.degree import install as shared_degree
        shared_degree(job)
