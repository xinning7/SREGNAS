
import copy
import os
import time

CACHE = 'SREGNAS_X20_FAST_VAL_CSC_CACHE'
PROFILE = 'SREGNAS_X20_FAST_VAL_CSC_PROFILE'


def enabled():
    return os.environ.get(CACHE)=='1' or os.environ.get(PROFILE)=='1'


def loader(worker, data, *, num_neighbors, input_nodes, batch_size):
    from torch_geometric.loader import NeighborLoader
    import torch_geometric.sampler.neighbor_sampler as sampler_module

    use_cache=os.environ.get(CACHE)=='1';profile=os.environ.get(PROFILE)=='1'
    edge=data.edge_index;key=(id(edge),edge._version,int(data.num_nodes))
    cache=getattr(worker,'_x20_validation_csc',None)
    if use_cache and (cache is None or cache['key']!=key or cache['edge'] is not edge):
        cache=dict(key=key,edge=edge,base=None,samplers={})
        worker._x20_validation_csc=cache
    fanout=tuple(num_neighbors);sampler=None;hit=False
    if use_cache and cache['base'] is not None:
        hit=True
        if fanout not in cache['samplers']:
            sampler=copy.copy(cache['base'])
            sampler.num_neighbors=list(num_neighbors)
            cache['samplers'][fanout]=sampler
        sampler=cache['samplers'][fanout]



    kwargs=dict(num_neighbors=num_neighbors,input_nodes=input_nodes,
                batch_size=batch_size,shuffle=False)
    if sampler is not None:kwargs['neighbor_sampler']=sampler
    original=sampler_module.to_csc;builds=[]
    def measured_csc(*args,**kw):
        start=time.perf_counter()
        try:return original(*args,**kw)
        finally:builds.append((start,time.perf_counter()))
    if profile:sampler_module.to_csc=measured_csc
    start=time.perf_counter()
    try:
        result=NeighborLoader(data,**kwargs)
    finally:
        if profile:sampler_module.to_csc=original
    ctor_s=time.perf_counter()-start
    if use_cache and cache['base'] is None:
        cache['base']=result.node_sampler
        cache['samplers'][fanout]=result.node_sampler
    if profile:
        from sregnas.icde27.ledger import event
        event('x20_native_validation_csc',worker_id=getattr(worker,'worker_id',None),
              cache_enabled=use_cache,cache_hit=hit,num_neighbors=list(num_neighbors),
              csc_builds=len(builds),csc_wall_s=sum(b-a for a,b in builds),
              csc_intervals=builds,loader_ctor_wall_s=ctor_s)
    return result
