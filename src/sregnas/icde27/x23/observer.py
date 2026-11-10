
from functools import wraps

class BatchIterator:
    def __init__(self, iterator, observer):
        self.iterator, self.observer = iterator, observer
    def __iter__(self): return self
    def __len__(self): return len(self.iterator)
    def __getattr__(self, name): return getattr(self.iterator, name)
    def __next__(self):
        with self.observer.span('sampling'):
            batch = next(self.iterator)
        if self.observer.method in ('AP', 'PAS-PL', 'GraphNAP') and hasattr(batch, 'n_id'):
            self.observer.batch(batch.n_id, batch.batch_size, batch.edge_index.shape[1])
        return batch

def install(observer):
    from torch_geometric.loader import NodeLoader
    def wrapper(fn):
        @wraps(fn)
        def create(*args, **kwargs):
            return BatchIterator(fn(*args, **kwargs), observer)
        return create
    observer.patch(NodeLoader, '_get_iterator', wrapper)
