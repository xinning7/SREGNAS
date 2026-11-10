

from copy import deepcopy
import logging
import hashlib
import math
import random
import time

import numpy as np
import torch

from sregnas.core.architecture import Architecture
from sregnas.models.canonical import load_model_module
from sregnas.training.minibatch import _hash_seed


METHODS = ("GraphNAP", "AP", "PAS-PL")
STEPS = 50


class GraphNASLogger:
    pass

    def __init__(self, name, **kwargs):
        self.logger = logging.getLogger(name)


def token_space():
    from sregnas.icde27.space import FIELDS
    return deepcopy(FIELDS)


def candidate(layers, fingerprint=None):
    space = token_space()
    if not isinstance(layers, list) or len(layers) != 2:
        raise ValueError("D1 requires exactly two canonical layers")
    for layer in layers:
        if set(layer) != set(space) or any(layer[k] not in v for k, v in space.items()):
            raise ValueError("candidate outside the P0-C token intersection")
        if load_model_module()._canonicalize_layer(layer) != layer:
            raise ValueError("candidate contains a noncanonical head token")
    fp = Architecture(layers).fingerprint()
    if fingerprint is not None and fp != fingerprint:
        raise ValueError("fingerprint does not match the canonical definition")
    return dict(fingerprint=fp, layers=deepcopy(layers), num_layers=2)


class ModelBuilder:
    @staticmethod
    def build_model(spec, input_dim, output_dim, config):
        spec = candidate(spec["layers"], spec["fingerprint"])
        if "_port_initialization_seed" in config:


            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(config["_port_initialization_seed"])
                return load_model_module().GNASModel(input_dim, output_dim, spec["layers"], dropout=0.0)
        return load_model_module().GNASModel(
            input_dim, output_dim, spec["layers"], dropout=0.0
        )


def reset_search_rng(fingerprint, device, partition_id=None):
    import os
    if os.environ.get('SREGNAS_X22_QUALIFY_NUMERICS') == '1':
        from sregnas.icde27.x22.qualify_numerics import enable
        enable()
    inputs = [fingerprint, "search", 0]
    if partition_id is not None:
        inputs.append(partition_id)
    seed = _hash_seed(*inputs)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def state_hash(state):
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def dataset_config(dataset, num_classes):
    profiles = {
        "flickr": (512, [25, 10]), "ogbn-arxiv": (512, [15, 10]),
        "ogbn-products": (256, [15, 10]), "reddit": (512, [5, 5]),
    }
    batch, fanout = profiles[dataset]




    return dict(num_classes=num_classes, learning_rate=0.01, weight_decay=5e-4,
                dropout=0.0, neighbor_sizes=fanout,
                training=dict(batch_size=batch, fan_out=fanout),
                protocol="D3-new-Reddit" if dataset == "reddit" else "D3-reference")


def failure(exc):
    text = str(exc).lower()
    if isinstance(exc, torch.OutOfMemoryError) or "out of memory" in text:
        return "OOM"
    if isinstance(exc, TimeoutError) or "gettimeouterror" in type(exc).__name__.lower():
        return "timeout"
    if isinstance(exc, (NotImplementedError, ImportError)):
        return "unsupported"
    return "incomplete"


def cpu_state(state):
    return {key: value.detach().cpu() for key, value in state.items()}


class ObservedWorker:
    pass

    def begin(self, spec):
        candidate(spec["layers"], spec["fingerprint"])
        self._port_optimizer_steps = 0
        self._port_started = time.perf_counter()
        self._port_seed = reset_search_rng(spec["fingerprint"], self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        return self.metrics()

    def metrics(self):
        if __import__('os').environ.get('SREGNAS_X22_INCREMENTAL_AUDIT')=='1':


            from sregnas.icde27.x22.origins import audit
            audit()
        from .cached_sampler import metrics as sampler_metrics
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        data = getattr(self, "full_data", None)
        if data is None:
            data = self.data_partition
        return dict(
            **sampler_metrics(self),
            worker_id=self.worker_id, seed=getattr(self, "_port_seed", None),
            initialization_hash=getattr(self, "_port_initialization_hash", None),
            training_seed=getattr(self, "_port_training_seed", None),
            optimizer_steps=getattr(self, "_port_optimizer_steps", 0),
            elapsed_s=(time.perf_counter() - self._port_started
                       if hasattr(self, "_port_started") else None),
            peak_allocated_bytes=(torch.cuda.max_memory_allocated(self.device)
                                  if self.device.type == "cuda" else None),
            peak_reserved_bytes=(torch.cuda.max_memory_reserved(self.device)
                                 if self.device.type == "cuda" else None),
            feature_device=str(data.x.device),
            search_features_on_gpu=data.x.device.type == "cuda",
        )

    def x22_finalize_framework(self):
        from sregnas.icde27.x22.ray_finish import finish_worker
        return finish_worker()


class ParallelCandidateWorker(ObservedWorker):
    pass

    def run_candidate(self, spec):
        started = time.perf_counter()
        row = result_row(spec)
        stage = "seed"
        try:
            self.begin(spec)
            stage = "train"
            state, loss, stats = self.train_architecture(
                spec, None, epochs=5,
                batch_size=self.default_batch_size,
                steps_per_epoch=10,
            )
            if self._port_optimizer_steps != STEPS:
                raise RuntimeError(f"optimizer budget incomplete: {self._port_optimizer_steps}/{STEPS}")
            if not math.isfinite(loss):
                raise FloatingPointError("nonfinite proxy loss")
            stage = "proxy_eval"
            score = self.evaluate_architecture(spec, state)
            validate_score(score, "val_accuracy")
            row.update(status="OK", value=score["val_accuracy"], loss=loss,
                       evaluation=score, training=stats)
        except Exception as exc:
            row.update(status=failure(exc), failure_stage=stage,
                       error=f"{type(exc).__name__}: {exc}")
        row.update(self.metrics(), wall_s=time.perf_counter() - started)
        return row

    def capacity_probe(self, spec):
        pass
        row = result_row(spec)
        try:
            self.begin(spec)
            self.train_architecture(spec, None, epochs=1,
                                    batch_size=self.default_batch_size, steps_per_epoch=1)
            assert self._port_optimizer_steps == 1
            row.update(status="OK", expected_optimizer_steps=1)
        except Exception as exc:
            row.update(status=failure(exc), failure_stage="capacity_train_step",
                       error=f"{type(exc).__name__}: {exc}")
        row.update(self.metrics())
        return row


def validate_score(score, key):
    value = score.get(key)
    if (not score.get("evaluation_successful", False) or value is None
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise RuntimeError("invalid proxy validation result")


def result_row(spec):
    return dict(fingerprint=spec["fingerprint"], architecture=spec["layers"],
                status="incomplete", value=None, failure_stage=None,
                seed_inputs=[spec["fingerprint"], "search", 0],
                seed_function="sregnas.training.minibatch._hash_seed",
                optimizer_steps=0, peak_allocated_bytes=None,
                peak_reserved_bytes=None, parameter_average_count=None,
                parameter_average_bytes=None, proxy_split="val",
                proxy_inference="sampled", expected_optimizer_steps=STEPS)
