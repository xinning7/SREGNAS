

import time
import math
import ray
import torch

from sregnas.baselines.fastgnas.data.partition import PartitionGraph
from .common import (ObservedWorker, GraphNASLogger, cpu_state, failure,
                     result_row, validate_score, STEPS, ModelBuilder, reset_search_rng, state_hash)
from .reference import RayWorker, CABackends


def partitions_from_data(data, count):

    partitioner = PartitionGraph.__new__(PartitionGraph)
    partitioner.dataset_name = "canonical"
    partitioner.num_partitions = count
    partitioner.max_hops = 2
    partitioner.full_graph = data
    partitioner.all_train_indices = data.train_mask.nonzero().view(-1)
    partitioner.dataset_graph_stats = None
    return partitioner._partition_balanced_stratified_khop()


class Worker(ObservedWorker, RayWorker):
    def begin(self, spec):
        super().begin(spec)
        self.config["_port_initialization_seed"] = self._port_seed
        initial = ModelBuilder.build_model(spec, self.data_partition.num_node_features,
                                          self.config["num_classes"], self.config)
        self._port_initial_state = cpu_state(initial.state_dict())
        self._port_initialization_hash = state_hash(self._port_initial_state)
        self._port_training_seed = reset_search_rng(spec["fingerprint"], self.device, self.worker_id)
        from sregnas.icde27.ledger import event
        event("graphnap_rng", fingerprint=spec["fingerprint"], partition_id=self.worker_id,
              initialization_seed=self._port_seed, initialization_hash=self._port_initialization_hash,
              training_seed=self._port_training_seed,
              init_seed_inputs=[spec["fingerprint"], "search", 0],
              train_seed_inputs=[spec["fingerprint"], "search", 0, self.worker_id])
        return self.metrics()

    def train_on_partition(self, *args, **kwargs):
        if args[1] is None:
            args = (args[0], self._port_initial_state, *args[2:])
        state, loss, timing = super().train_on_partition(*args, **kwargs)

        return cpu_state(state), loss, timing


class Backend(CABackends):
    def __init__(self, workers, config):

        self.workers = workers
        self.config = config
        self.num_workers = len(workers)
        self.logger = GraphNASLogger(name="graphnap")
        self.parameter_average_count = 0
        self.parameter_average_bytes = 0

    def synchronize_parameters(self, states):
        averaged = super().synchronize_parameters(states)
        self.parameter_average_count += 1


        self.parameter_average_bytes += sum(
            tensor.numel() * tensor.element_size()
            for state in states for tensor in state.values()
        )
        return averaged

    def run_candidate(self, spec):
        started = time.perf_counter()
        row = result_row(spec)
        self.parameter_average_count = self.parameter_average_bytes = 0
        stage = "seed"
        try:
            ray.get([worker.begin.remote(spec) for worker in self.workers])
            stage = "train"
            state, loss, timing = self.train_epoch(
                spec, epochs=5, steps_per_epoch=10,
                batch_size=self.config["training"]["batch_size"],
            )
            if not math.isfinite(loss):
                raise FloatingPointError("nonfinite proxy loss")
            stage = "proxy_eval"
            score = self.evaluate_model(spec, state)
            validate_score(score, "global_accuracy")
            row.update(status="OK", value=score["global_accuracy"], loss=loss,
                       evaluation=score, training=timing)
        except Exception as exc:
            row.update(status=failure(exc), failure_stage=stage,
                       error=f"{type(exc).__name__}: {exc}")
        try:
            metrics = ray.get([worker.metrics.remote() for worker in self.workers])
            row.update(workers=metrics,
                       optimizer_steps=min(m["optimizer_steps"] for m in metrics),
                       optimizer_steps_total=sum(m["optimizer_steps"] for m in metrics),
                       search_features_on_gpu=all(m["search_features_on_gpu"] for m in metrics))
            for field in ("peak_allocated_bytes", "peak_reserved_bytes"):
                values = [m[field] for m in metrics if m[field] is not None]
                row[field] = max(values) if values else None
            if row["status"] == "OK" and any(m["optimizer_steps"] != STEPS for m in metrics):
                row.update(status="incomplete", value=None, failure_stage="budget",
                           error="every replica must complete 50 optimizer steps")
        except Exception as exc:
            row["metrics_error"] = f"{type(exc).__name__}: {exc}"
            if row["status"] == "OK":
                row.update(status=failure(exc), value=None, failure_stage="worker_metrics",
                           error=row["metrics_error"])
        row.update(parameter_average_count=self.parameter_average_count,
                   parameter_average_bytes=self.parameter_average_bytes,
                   parameter_average_bytes_definition="sum input state tensor bytes, all replicas, all means",
                   wall_s=time.perf_counter() - started)
        return row
