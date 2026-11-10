
import os
import time
from sregnas.baselines.graphnap.adapter import Worker as GraphBase
from sregnas.baselines.ap.adapter import Worker as APBase
from sregnas.baselines.paspl.adapter import Worker as PASBase
from .profiling import capture


class Observation:
    def placement(self):
        import ray
        return dict(worker=self.worker_id, visible=os.environ.get("CUDA_VISIBLE_DEVICES"), gpu_ids=ray.get_gpu_ids())

    def spans(self):
        return getattr(self, "_p2_spans", [])

    def start_profile(self, fingerprint):
        self._p2_start = time.perf_counter()
        self._p2_capture = capture(f"worker{self.worker_id}_{fingerprint}")
        self._p2_capture.__enter__()
        if os.environ.get('SREGNAS_X18_COST_PROFILE') in ('sync', 'control', 'trace'):
            from sregnas.icde27.x18.cost_hooks import start_worker
            self._x18_cost = start_worker(self, fingerprint)

    def finish_profile(self):
        end = time.perf_counter()
        cost = getattr(self, '_x18_cost', None)
        if cost is not None:
            self._x18_cost = None
            cost.finish()
        context = getattr(self, "_p2_capture", None)
        if context:
            context.__exit__(None, None, None)
            self._p2_capture = None
            if not hasattr(self, "_p2_spans"): self._p2_spans = []
            self._p2_spans.append(dict(worker=self.worker_id, start_perf_s=self._p2_start, end_perf_s=end))


class GraphWorker(Observation, GraphBase):
    _x18_cost_class_dispatch = True

    def train_on_partition(self, *args, **kwargs):
        cost = getattr(self, '_x18_cost', None)
        if cost is not None:
            return cost.graph_training(super().train_on_partition, *args, **kwargs)
        return super().train_on_partition(*args, **kwargs)

    def evaluate_on_partition(self, *args, **kwargs):
        cost = getattr(self, '_x18_cost', None)
        if cost is not None:
            return cost.graph_validation(super().evaluate_on_partition, *args, **kwargs)
        return super().evaluate_on_partition(*args, **kwargs)

    def begin(self, spec):
        self._gc_fingerprint = spec["fingerprint"]
        self.start_profile(spec["fingerprint"])
        return super().begin(spec)

    def finish_profile(self):
        super().finish_profile()
        if self.config.get("collect_search_garbage", False):
            import gc
            from sregnas.icde27.ledger import event
            count = gc.collect()
            event("graphnap_candidate_gc", worker=self.worker_id,
                  fingerprint=self._gc_fingerprint, unreachable_objects_collected=count)


class APWorker(Observation, APBase):
    def run_candidate(self, spec):
        self.start_profile(spec["fingerprint"])
        try: return super().run_candidate(spec)
        finally: self.finish_profile()


class PASWorker(Observation, PASBase):
    def run_candidate(self, spec):
        self.start_profile(spec["fingerprint"])
        try: return super().run_candidate(spec)
        finally: self.finish_profile()
