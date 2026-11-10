
from contextlib import contextmanager
from functools import wraps
import os
from pathlib import Path
import time
import uuid

ENV = 'SREGNAS_X18_COST_PROFILE'


def enabled():
    return os.environ.get(ENV) in ('sync', 'control', 'trace')


def tensor_bytes(state):
    return sum(t.numel() * t.element_size() for t in (state or {}).values())


class Observer:
    def __init__(self, method, fingerprint, worker, *, scope='candidate', mapping=None,
                 train_mask=None, synchronize=None, mode=None):
        self.method, self.fingerprint, self.worker = method, fingerprint, worker
        self.scope, self.mapping, self.train_mask = scope, mapping, train_mask
        self.mode = mode or os.environ.get(ENV, 'control')
        self.synchronize = synchronize
        self.phase = 'setup'
        self.stack, self.patches = [], []
        self.totals = dict(sampling=0., transfer=0., compute=0., validation=0.)
        self.step_started = None
        self.steps = []
        self.updates = self.targets = self.input_edges = self.layer_edges = 0
        self.target_ids, self.input_ids = [], []
        self.send_bytes = self.receive_bytes = 0
        self.trace = None

    def sync(self):
        if self.mode == 'sync' and self.synchronize:
            self.synchronize()

    def patch(self, owner, name, wrapper):
        fn = getattr(owner, name)
        self.patches.append((owner, name, fn))
        setattr(owner, name, wrapper(fn))

    @contextmanager
    def span(self, category):
        if self.phase != 'train':
            yield
            return
        self.sync()
        frame = [time.perf_counter(), 0.]
        self.stack.append(frame)
        try:
            yield
        finally:
            self.sync()
            elapsed = time.perf_counter() - frame[0]
            self.stack.pop()
            self.totals[category] += elapsed - frame[1]
            if self.stack:
                self.stack[-1][1] += elapsed

    def measured(self, category):
        def wrapper(fn):
            @wraps(fn)
            def call(*args, **kwargs):
                with self.span(category):
                    return fn(*args, **kwargs)
            return call
        return wrapper

    def begin_step(self):
        self.end_step()
        self.phase = 'train'
        self.sync()
        self.step_started = time.perf_counter()

    def end_step(self):
        if self.step_started is not None:
            self.sync()
            self.steps.append(time.perf_counter() - self.step_started)
            self.step_started = None

    def training(self, fn):
        @wraps(fn)
        def call(*args, **kwargs):
            old = self.phase
            self.phase = 'setup'
            try:
                return fn(*args, **kwargs)
            finally:
                self.end_step()
                self.phase = old
        return call

    def validation(self, fn):
        @wraps(fn)
        def call(*args, **kwargs):
            if self.phase == 'validation':
                return fn(*args, **kwargs)
            self.end_step()
            old, self.phase = self.phase, 'validation'
            self.sync()
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.sync()
                self.totals['validation'] += time.perf_counter() - start
                self.phase = old
        return call

    def batch(self, ids, batch_size, edges, *, layer_edges=None):


        if self.phase != 'train':
            return
        self.target_ids.append(ids[:int(batch_size)].detach())
        self.input_ids.append(ids.detach())
        self.input_edges += int(edges)
        self.layer_edges += int(layer_edges if layer_edges is not None else edges * 2)

    def install(self):
        import torch
        import torch.nn.functional as F
        from torch_geometric.data import Data
        from torch.utils.data import DataLoader
        from torch.utils.data.dataloader import _BaseDataLoaderIter
        for owner, name, category in (
            (Data, 'to', 'transfer'),
            (torch.nn.Module, '_call_impl', 'compute'), (torch.Tensor, 'backward', 'compute'),
            (torch.optim.Adam, 'zero_grad', 'compute'), (DataLoader, '__iter__', 'sampling')):
            self.patch(owner, name, self.measured(category))

        def device_copy(name):
            def wrapper(fn):
                @wraps(fn)
                def call(tensor, *args, **kwargs):
                    destination = tensor.device
                    if name == 'cpu':
                        destination = torch.device('cpu')
                    elif name == 'cuda':
                        destination = torch.device('cuda', torch.cuda.current_device())
                    elif name == 'copy_':
                        source = args[0] if args else kwargs.get('src')
                        destination = getattr(source, 'device', destination)
                    else:
                        requested = kwargs.get('device', args[0] if args else None)
                        if isinstance(requested, torch.Tensor):
                            destination = requested.device
                        elif isinstance(requested, (str, torch.device, int)):
                            destination = torch.device(requested)


                    if destination.type != tensor.device.type or (
                        destination.index is not None and destination.index != tensor.device.index
                    ):
                        with self.span('transfer'):
                            return fn(tensor, *args, **kwargs)
                    return fn(tensor, *args, **kwargs)
                return call
            return wrapper
        for name in ('to', 'cuda', 'cpu', 'copy_'):
            self.patch(torch.Tensor, name, device_copy(name))

        def loss(fn):
            @wraps(fn)
            def call(input, target, *args, **kwargs):
                if self.phase == 'train':
                    self.targets += target.numel()
                with self.span('compute'):
                    return fn(input, target, *args, **kwargs)
            return call
        def step(fn):
            @wraps(fn)
            def call(*args, **kwargs):
                with self.span('compute'):
                    result = fn(*args, **kwargs)
                if self.phase == 'train':
                    self.updates += 1
                return result
            return call
        def loader_init(fn):
            @wraps(fn)
            def call(*args, **kwargs):
                if self.method in ('AP', 'PAS-PL', 'GraphNAP') and self.phase == 'setup':
                    self.begin_step()
                with self.span('sampling'):
                    return fn(*args, **kwargs)
            return call
        def next_batch(fn):
            @wraps(fn)
            def call(*args, **kwargs):
                with self.span('sampling'):
                    result = fn(*args, **kwargs)
                if os.environ.get('SREGNAS_X23_OUTER_OBSERVER') != '1' and self.method in ('AP', 'PAS-PL', 'GraphNAP') and hasattr(result, 'n_id'):
                    self.batch(result.n_id, result.batch_size, result.edge_index.shape[1])
                return result
            return call
        self.patch(F, 'cross_entropy', loss)
        self.patch(torch.optim.Adam, 'step', step)
        self.patch(DataLoader, '__init__', loader_init)
        self.patch(_BaseDataLoaderIter, '__next__', next_batch)
        if os.environ.get('SREGNAS_X23_OUTER_OBSERVER') == '1':
            from sregnas.icde27.x23.observer import install
            install(self)

    def install_shard(self):
        import torch
        from sregnas.training import fixed_proxy
        self.patch(fixed_proxy, 'train_proxy_batches', self.training)
        def draw(fn):
            @wraps(fn)
            def call(*args, **kwargs):
                self.begin_step()
                with self.span('sampling'):
                    return fn(*args, **kwargs)
            return call
        def sliced(fn):
            @wraps(fn)
            def call(index, targets):
                with self.span('sampling'):
                    batch = fn(index, targets)
                self.target_ids.append(targets.detach())
                self.input_ids.append(batch.nodes.detach())
                self.input_edges += sum(e.shape[1] for e in batch.edges)
                self.layer_edges += sum(e.shape[1] for e in batch.edges)
                return batch
            return call
        self.patch(fixed_proxy.APTargetDraws, 'draw', draw)
        self.patch(fixed_proxy.FixedProxyIndex, 'slice', sliced)
        self.patch(fixed_proxy, 'forward_batch', self.measured('compute'))


        original_training = fixed_proxy.train_proxy_batches
        def training_with_validation(*args, **kwargs):
            self._validation_start = None
            try:
                return original_training(*args, **kwargs)
            finally:
                if self._validation_start is not None:
                    self.sync()
                    self.totals['validation'] += time.perf_counter() - self._validation_start
                    self._validation_start = None
        self.patch(fixed_proxy, 'train_proxy_batches', lambda fn: training_with_validation)
        def mode(fn):
            @wraps(fn)
            def call(module, mode=True):
                if not mode and self.phase == 'train':
                    self.end_step()
                    self.phase = 'validation'
                    self.sync()
                    self._validation_start = time.perf_counter()
                return fn(module, mode)
            return call
        self.patch(torch.nn.Module, 'train', mode)

    def install_worker(self, worker):
        if self.method == 'GraphNAP':
            if getattr(worker, '_x18_cost_class_dispatch', False):



                return
            def train(fn):
                @wraps(fn)
                def call(*args, **kwargs):
                    return self.graph_training(fn, *args, **kwargs)
                return call
            def evaluate(fn):
                @wraps(fn)
                def call(*args, **kwargs):
                    return self.graph_validation(fn, *args, **kwargs)
                return call
            self.patch(worker, 'train_on_partition', train)
            self.patch(worker, 'evaluate_on_partition', evaluate)
        elif self.method in ('AP', 'PAS-PL'):
            self.patch(worker, 'train_architecture', self.training)
            self.patch(worker, 'evaluate_architecture', self.validation)
            def loader(fn):
                @wraps(fn)
                def call(*args, **kwargs):
                    if self.phase != 'validation':
                        self.begin_step()
                    with self.span('sampling'):
                        return fn(*args, **kwargs)
                return call
            self.patch(worker, '_create_neighbor_loader', loader)
        else:
            from sregnas.baselines.fastgnas.batching.cache_pool import MiniBatchCacheObject
            self._last_elu = None
            def elu(fn):
                @wraps(fn)
                def call(*args, **kwargs):
                    result = fn(*args, **kwargs)
                    self._last_elu = result
                    return result
                return call
            def batch(fn):
                @wraps(fn)
                def call(*args, **kwargs):
                    self.begin_step()
                    with self.span('sampling'):
                        result = fn(*args, **kwargs)
                    elu = self._last_elu
                    if elu is None or elu.edge_index.data_ptr() != result.edge_index.data_ptr():
                        raise AssertionError('observed native ELU does not match the returned batch')
                    self.batch(elu.n_id, result.batch_size, result.edge_index.shape[1],
                               layer_edges=result.edge_index.shape[1] * result.num_layers)
                    return result
                return call
            self.patch(MiniBatchCacheObject, 'get_elu', elu)
            self.patch(worker.batch_manager, 'get_batch', batch)
            self.patch(worker, '_evaluate', self.validation)

    def graph_training(self, function, *args, **kwargs):
        self.receive_bytes += tensor_bytes(args[1] if len(args) > 1 else kwargs.get('model_state'))
        result = self.training(function)(*args, **kwargs)
        self.send_bytes += tensor_bytes(result[0])
        return result

    def graph_validation(self, function, *args, **kwargs):
        self.receive_bytes += tensor_bytes(args[1] if len(args) > 1 else kwargs.get('model_state'))
        return self.validation(function)(*args, **kwargs)

    def start(self):
        self.install()
        self.started = time.perf_counter()
        from sregnas.icde27.p2p6.profiling import capture
        self.label = f'x18cost_{self.method}_{self.fingerprint}_w{self.worker}_{self.scope}_{uuid.uuid4().hex[:8]}'
        self.trace = capture(self.label, enabled=self.mode == 'trace')
        self.trace.__enter__()
        return self

    def finish(self, status='OK'):
        self.end_step()
        ended = time.perf_counter()
        try:
            if self.trace is not None:
                self.trace.__exit__(None, None, None)
        finally:
            for owner, name, original in reversed(self.patches):
                setattr(owner, name, original)
        import torch
        def unique(tensors):
            return torch.unique(torch.cat([t.cpu().long() for t in tensors])).long() if tensors else torch.empty(0, dtype=torch.long)
        targets, touched = unique(self.target_ids), unique(self.input_ids)
        if self.train_mask is not None:
            touched = touched[self.train_mask.cpu()[touched].bool()]
        if self.mapping is not None:
            mapping = self.mapping.cpu()
            targets, touched = mapping[targets], mapping[touched]
        from .common import write, VERSION
        row = dict(version=VERSION, method=self.method, fingerprint=self.fingerprint,
                   worker=self.worker, scope=self.scope, pass_name=self.mode, status=status,
                   optimizer_steps=self.updates, training_target_exposures=self.targets,
                   target_node_ids=targets.tolist(), touched_training_node_ids=touched.tolist(),
                   distinct_target_nodes=len(targets), distinct_touched_training_nodes=len(touched),
                   forward_input_edges=self.input_edges, forward_layer_input_edges=self.layer_edges,
                   phase_seconds=self.totals, training_step_seconds=self.steps,
                   training_wall_s=sum(self.steps), observed_call_wall_s=ended-self.started,
                   logical_model_bytes_sent=self.send_bytes, logical_model_bytes_received=self.receive_bytes,
                   trace_label=self.label if self.mode == 'trace' else None,
                   timing_scope='C1 per-worker step clocks; validation amortized over optimizer steps; setup excluded',
                   edge_scope='layer input COO edges, before operator self loops; SREGNAS sums distinct fixed layer graphs',
                   transfer_scope='physical H2D from separate trace; setup graph preload excluded')
        root = Path(os.environ['SREGNAS_ICDE27_RECORD_DIR']) / 'cost_observations'
        write(root / f'{self.label}_{self.mode}.json', row)
        return row


def start_worker(worker, fp):
    import torch
    method = os.environ['SREGNAS_X18_COST_METHOD']
    data = getattr(worker, 'data_partition_cpu', None) if method == 'GraphNAP' else getattr(worker, 'cpu_data', None)
    mapping = getattr(worker, 'global_node_ids_cpu', None) if method == 'GraphNAP' else None
    if os.environ.get('SREGNAS_X23_OUTER_OBSERVER') == '1' and method == 'PAS-PL':

        data = worker.full_data
    mask = getattr(data, 'train_mask', None)
    observer = Observer(method, fp, worker.worker_id, mapping=mapping, train_mask=mask,
                        synchronize=lambda: torch.cuda.synchronize(worker.device)).start()
    observer.install_worker(worker)
    return observer


def shard_call(spec, function, runtime):
    import torch
    from sregnas.training.fullbatch import load_graph_payload


    runtime.load_payload(spec['graph_path'], 'proxy')
    payload = load_graph_payload(spec['graph_path'])
    observer = Observer('SREGNAS', spec['fingerprint'], spec['shard'],
                        mapping=payload['node_ids'], train_mask=payload['train_mask'],
                        synchronize=lambda: torch.cuda.synchronize(runtime.device)).start()
    observer.install_shard()
    status = 'incomplete'
    try:
        result = function(spec, runtime=runtime)
        status = result.status
        return result
    finally:
        observer.finish(status)


def native_call(function, worker, arch, state, config):
    import torch
    from sregnas.icde27.native import fingerprint
    data = worker.cpu_subgraph_for_sampling
    observer = Observer('FastGNAS', fingerprint(arch), worker.worker_id, scope='training_segment',
                        mapping=getattr(data, 'global_node_ids', None), train_mask=data.train_mask,
                        synchronize=lambda: torch.cuda.synchronize(worker.device)).start()
    observer.install_worker(worker)
    status = 'incomplete'
    try:
        result = observer.training(function)(worker, arch, state, config)
        status = 'OK'
        return result
    finally:
        observer.finish(status)


def native_validation(function):
    @wraps(function)
    def call(worker, arch, state):
        if not enabled():
            return function(worker, arch, state)
        from sregnas.baselines.fastgnas.timing_hook import phase
        if phase() != 'search':
            return function(worker, arch, state)
        import torch
        from sregnas.icde27.native import fingerprint
        observer = Observer('FastGNAS', fingerprint(arch), worker.worker_id, scope='global_validation',
                            synchronize=lambda: torch.cuda.synchronize(worker.device)).start()
        status = 'incomplete'
        try:
            result = observer.validation(function)(worker, arch, state)
            status = 'OK'
            return result
        finally:
            observer.finish(status)
    return call
