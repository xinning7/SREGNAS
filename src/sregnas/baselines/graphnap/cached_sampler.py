
import time
import os


def initialize(worker, data, fanout):
    if not worker.config.get('cached_sampler', False):
        return
    from torch_geometric.sampler import NeighborSampler
    started = time.perf_counter()
    worker._cached_neighbor_sampler = NeighborSampler(data, num_neighbors=list(fanout))
    worker._cached_sampler_data = data
    worker._cached_sampler_fanout = tuple(fanout)
    worker._cached_sampler_setup_s = time.perf_counter() - started
    worker._cached_sampler_build_count = 1


def kwargs(worker, data, fanout):
    if not worker.config.get('cached_sampler', False):
        return {}
    if not hasattr(worker, '_cached_neighbor_sampler'):
        initialize(worker, data, fanout)
    if worker._cached_sampler_data is not data or tuple(fanout) != worker._cached_sampler_fanout:
        raise ValueError('cached sampler graph/fanout changed within a worker')
    return {'neighbor_sampler': worker._cached_neighbor_sampler}


def metrics(worker):
    if not worker.config.get('cached_sampler', False):
        return {}
    return dict(cached_sampler=True, sampler_build_count=worker._cached_sampler_build_count,
                sampler_setup_s=worker._cached_sampler_setup_s,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                sampler_device='cpu', csc_source_device=str(worker._cached_sampler_data.edge_index.device))
