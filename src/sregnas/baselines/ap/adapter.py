

from sregnas.baselines.graphnap.common import ParallelCandidateWorker
from .reference import ArchParallelWorkerV2


class Worker(ParallelCandidateWorker, ArchParallelWorkerV2):
    def run_candidate(self, spec):
        row = super().run_candidate(spec)
        if self.config.get("collect_search_garbage", False):
            import gc
            import time
            from sregnas.baselines.graphnap.truncation import host_rss_bytes
            started = time.perf_counter()
            row["host_rss_before_collect_bytes"] = host_rss_bytes()
            row["unreachable_objects_collected"] = gc.collect()
            row["host_rss_bytes"] = host_rss_bytes()
            row["garbage_collection_wall_s"] = time.perf_counter()-started
            row["wall_s"] += row["garbage_collection_wall_s"]
        return row

    def _create_neighbor_loader(self, train_mask, batch_size):
        if self.full_data.x.device.type != "cpu" or self.cpu_data.x.device.type != "cpu":
            raise AssertionError("AP search features must stay on CPU")
        return super()._create_neighbor_loader(train_mask, batch_size)

    def evaluate_architecture(self, spec, state):
        if self.cpu_data.x.device.type != "cpu":
            raise AssertionError("AP proxy evaluation features must stay on CPU")
        return super().evaluate_architecture(spec, state)
