

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from sregnas.core import Architecture

from sregnas.models.canonical import assert_model, load_model_module
from sregnas.core.contracts import CellResult, Status
from sregnas.evaluation.inference import score_full_graph_inference
from sregnas.evaluation.checkpoints import (
    POST_SELECTION_TEST_MODE,
    assert_test_metric_isolated,
    load_retrain_checkpoint,
    save_retrain_checkpoint,
)


GB = 1024**3
CELL_TIMING_FIELDS = (
    "process_spawn_s",
    "cuda_init_s",
    "data_load_h2d_s",
    "train_compute_s",
    "teardown_s",
)


def load_graph_payload(path):
    if str(path).endswith('.x18-shared.json'):
        from sregnas.icde27.x18.shared_graph import load_payload
        return load_payload(path)
    return torch.load(path,map_location='cpu',weights_only=False)


class PersistentCellRuntime:
    pass

    def __init__(self):
        cuda_started = time.perf_counter()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(0)
            torch.cuda.init()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
            torch.cuda.synchronize(self.device)
        self.startup_cuda_init_s = time.perf_counter() - cuda_started
        self.model_module = load_model_module(force_native_scatter=True)
        self._startup_timing_unclaimed = True
        self._payloads: dict[tuple[str, str], dict[str, Any]] = {}
        self.proxy_top1: dict[str, Any] | None = None

    def claim_startup_cuda_init_s(self) -> float:
        if not self._startup_timing_unclaimed:
            return 0.0
        self._startup_timing_unclaimed = False
        return self.startup_cuda_init_s

    def load_payload(
        self, graph_path: str, mode: str
    ) -> tuple[dict[str, Any], float, bool]:
        kind = (
            "proxy"
            if mode == "proxy"
            else "test" if mode == POST_SELECTION_TEST_MODE else "full"
        )
        key = (str(Path(graph_path).resolve()), kind)
        cached = self._payloads.get(key)
        if cached is not None:
            return cached, 0.0, True
        started = time.perf_counter()
        source = load_graph_payload(graph_path)
        common = {
            "x": source["x"].to(self.device),
            "labels": source["labels"].to(self.device),
            "num_features": int(source["num_features"]),
            "num_classes": int(source["num_classes"]),
        }
        if kind == "proxy":
            common.update(
                {
                    "train_mask": source["train_mask"].to(self.device),
                    "val_mask": source["val_mask"].to(self.device),
                    "edge_index_layer1": source["edge_index_layer1"].to(self.device),
                    "edge_index_layer2": source["edge_index_layer2"].to(self.device),
                    "expected_e_msg_sub": int(source["expected_e_msg_sub"]),
                    "msg1": int(source["msg1"]),
                    "msg2": int(source["msg2"]),
                }
            )
            from .fixed_proxy import BATCH_ENV, payload_index
            import os
            if BATCH_ENV in os.environ:
                if "proxy_fixed_index" in source:
                    common["proxy_fixed_index"] = source["proxy_fixed_index"]
                payload_index(common, [common["edge_index_layer1"], common["edge_index_layer2"]], self.device)
        elif kind == "test":
            common.update(
                {
                    "test_mask": source["test_mask"].to(self.device),
                    "edge_index": source["edge_index"].to(self.device),
                }
            )
        else:
            common.update(
                {
                    "train_mask": source["train_mask"].to(self.device),
                    "val_mask": source["val_mask"].to(self.device),
                    "edge_index": source["edge_index"].to(self.device),
                }
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        self._payloads[key] = common
        return common, elapsed, False


class PerLayerCanonicalModel(torch.nn.Module):
    pass

    def __init__(self, canonical_model: torch.nn.Module):
        super().__init__()
        self.canonical_model = canonical_model

    def forward(
        self, x: torch.Tensor, edge_indices: list[torch.Tensor]
    ) -> torch.Tensor:
        model = self.canonical_model
        full_degrees = getattr(edge_indices, "nf_full_degrees", None)
        for index, (conv, act_name) in enumerate(zip(model.convs, model._act_names)):
            x = F.dropout(x, p=model.dropout, training=model.training)
            edge_index = edge_indices[index]
            if full_degrees is None:
                x = model._act(conv(x, edge_index), act_name)
            else:
                from sregnas.models.normalization_fix import forward
                x = model._act(forward(conv, x, edge_index, full_degrees), act_name)
        x = F.dropout(x, p=model.dropout, training=model.training)
        return model.classifier(x)


def fingerprint_seed(fingerprint: str) -> int:
    return int(fingerprint[:8], 16) % (2**31)


def fingerprint_retrain_replica_seed(fingerprint: str, replica: int) -> int:
    pass
    payload = json.dumps(
        (fingerprint, "retrain", int(replica)),
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big"
    ) % (2**31)


def _set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _accuracy(out: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum()) == 0:
        raise AssertionError("accuracy mask is empty")
    return float((out[mask].argmax(1) == labels[mask]).float().mean().item())


@dataclass(frozen=True)
class ProxyTrainResult:
    sub_acc: float
    epochs: int


def train_proxy_fixed_epochs(
    model: PerLayerCanonicalModel,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    edge_indices: list[torch.Tensor],
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    *,
    epochs: int,
) -> ProxyTrainResult:
    pass
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(x, edge_indices)
        loss = F.cross_entropy(out[train_mask], labels[train_mask])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite proxy loss: {loss.item()}")
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        sub_acc = _accuracy(model(x, edge_indices), labels, val_mask)
    return ProxyTrainResult(sub_acc=sub_acc, epochs=epochs)


def _memory(device: torch.device) -> tuple[float | None, float | None]:
    if device.type != "cuda":
        return None, None
    return (
        torch.cuda.max_memory_allocated(device) / GB,
        torch.cuda.max_memory_reserved(device) / GB,
    )


def retain_proxy_top1(runtime, spec, canonical, value):
    pass
    if runtime is None or spec.get("mode") != "proxy":
        raise ValueError("Proxy retention requires a persistent proxy worker")
    previous = runtime.proxy_top1
    if previous is not None and previous["shard"] != int(spec["shard"]):
        raise ValueError("Proxy top1 retention requires one shard per worker")
    rank = (-float(value), spec["fingerprint"])
    if previous is not None and rank >= (-previous["sub_acc"], previous["fingerprint"]):
        return False
    runtime.proxy_top1 = dict(
        schema="reddit_proxy_top1_v1",
        dataset=spec["dataset"],
        shard=int(spec["shard"]),
        fingerprint=spec["fingerprint"],
        architecture=spec["architecture"],
        sub_acc=float(value),
        epochs=int(spec["epochs"]),
        state_dict={
            key: tensor.detach().cpu().clone()
            for key, tensor in canonical.state_dict().items()
        },
        test_evaluations=0,
    )
    return True


def _record_retrain_validation(model, *, device, **score_kwargs):
    pass
    import random

    import numpy as np

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    python_state, numpy_state = random.getstate(), np.random.get_state()
    training_modes = [(module, module.training) for module in model.modules()]
    devices = [device.index or 0] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            inferred = score_full_graph_inference(
                model, device=device, splits=("val",), **score_kwargs
            )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for module, training in training_modes:
            module.training = training
    if inferred.val_acc is None:
        raise AssertionError("intermediate validation produced no val_acc")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return inferred.val_acc, time.perf_counter() - started


def run_cell(
    spec: dict[str, Any], runtime: PersistentCellRuntime | None = None
) -> CellResult:
    pass
    from .fixed_proxy import apply_options
    spec = apply_options(spec)
    phase = spec["phase"]
    mode = spec.get("mode", phase)
    record_validation = bool(spec.get("record_retrain_validation", False))
    if record_validation and mode != "retrain":
        raise ValueError("record_retrain_validation is supported only for retrain")
    validation_curve: list[dict[str, Any]] = []
    fingerprint = spec["fingerprint"]
    arch_desc = spec["architecture"]
    parent_launch = float(spec.get("parent_launch_perf_s", time.perf_counter()))
    timings = {field: 0.0 for field in CELL_TIMING_FIELDS}
    device = (
        runtime.device
        if runtime is not None
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    model: torch.nn.Module | None = None
    canonical: torch.nn.Module | None = None
    payload: dict[str, Any] | None = None
    opt: torch.optim.Optimizer | None = None
    x = labels = train_mask = val_mask = test_mask = edge_index = None
    edge_indices = None
    out = loss = None
    compute_started: float | None = None
    status = Status.OK
    value: float | None = None
    error: str | None = None
    metrics: dict[str, Any] = {}
    allocated: float | None = None
    reserved: float | None = None
    seed: int | None = None
    seed_policy = "fingerprint"
    cuda_boundary_synchronized = device.type != "cuda"
    try:
        if "retrain_seed" in spec:
            if mode != "retrain":
                raise AssertionError(
                    "explicit retrain_seed is valid only for retrain mode"
                )
            seed = int(spec["retrain_seed"])
            seed_policy = "explicit_retrain_seed"
        elif (
            mode in {"retrain", POST_SELECTION_TEST_MODE}
            and spec.get("retrain_seed_policy") == "fingerprint_retrain_replica"
        ):
            seed = fingerprint_retrain_replica_seed(
                fingerprint, int(spec.get("replica", 0))
            )
            seed_policy = "H(fingerprint,retrain,replica);side_excluded"
        elif (
            mode == "proxy"
            and spec.get("proxy_replica_seed", False)
            and int(spec.get("replica", 0)) != 0
        ):

            from sregnas.training.minibatch import _hash_seed

            seed = _hash_seed(fingerprint, "search", int(spec["replica"]))
            seed_policy = "H(fingerprint,search,replica);side_shard_excluded"
        else:
            seed = fingerprint_seed(fingerprint)
        _set_seed(seed)
        model_module = (
            runtime.model_module
            if runtime is not None
            else load_model_module(force_native_scatter=True)
        )
        arch = Architecture(arch_description=arch_desc)
        if arch.fingerprint() != fingerprint:
            raise AssertionError("cell architecture/fingerprint mismatch")

        cuda_started = time.perf_counter()
        if runtime is not None:
            timings["cuda_init_s"] = runtime.claim_startup_cuda_init_s()
        elif device.type == "cuda":


            torch.cuda.set_device(0)
            torch.cuda.init()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
            torch.cuda.synchronize(device)
            timings["cuda_init_s"] = time.perf_counter() - cuda_started

        data_started = time.perf_counter()
        graph_cache_hit = False
        if runtime is not None:
            payload, load_elapsed, graph_cache_hit = runtime.load_payload(
                spec["graph_path"], mode
            )
            timings["data_load_h2d_s"] = load_elapsed
        else:
            payload = load_graph_payload(spec['graph_path'])
        if mode == "proxy":
            msg1 = int(payload["edge_index_layer1"].size(1))
            msg2 = int(payload["edge_index_layer2"].size(1))
            expected = int(payload["expected_e_msg_sub"])
            if msg1 != int(payload["msg1"]) or msg2 != int(payload["msg2"]):
                raise AssertionError("stored per-layer message metadata mismatch")
            if msg1 + msg2 != expected:
                raise AssertionError(
                    f"TRAINING E_msg_sub {msg1}+{msg2} != scout-locked {expected}"
                )
            x = payload["x"]
            labels = payload["labels"]
            train_mask = payload["train_mask"]
            val_mask = payload["val_mask"]
            edge_indices = [
                payload["edge_index_layer1"],
                payload["edge_index_layer2"],
            ]
            if runtime is None:
                x = x.to(device)
                labels = labels.to(device)
                train_mask = train_mask.to(device)
                val_mask = val_mask.to(device)
                edge_indices = [edge.to(device) for edge in edge_indices]
        elif mode == POST_SELECTION_TEST_MODE:
            x = payload["x"]
            labels = payload["labels"]
            test_mask = payload["test_mask"]
            edge_index = payload["edge_index"]
            if runtime is None:
                x = x.to(device)
                labels = labels.to(device)
                test_mask = test_mask.to(device)
                edge_index = edge_index.to(device)
        else:
            x = payload["x"]
            labels = payload["labels"]
            train_mask = payload["train_mask"]
            val_mask = payload["val_mask"]
            edge_index = payload["edge_index"]
            if runtime is None:
                x = x.to(device)
                labels = labels.to(device)
                train_mask = train_mask.to(device)
                val_mask = val_mask.to(device)
                edge_index = edge_index.to(device)
        if device.type == "cuda" and runtime is None:
            torch.cuda.synchronize(device)
        if runtime is None:
            timings["data_load_h2d_s"] = time.perf_counter() - data_started
        elif device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(0)

        if mode == "proxy":
            canonical = model_module.build_model(
                arch,
                int(payload["num_features"]),
                int(payload["num_classes"]),
                dropout=float(spec["dropout"]),
            ).to(device)
            model = PerLayerCanonicalModel(canonical).to(device)
            assert_model(model, model_module)
            epochs = int(spec["epochs"])
            opt = torch.optim.Adam(
                model.parameters(),
                lr=float(spec["lr"]),
                weight_decay=float(spec["weight_decay"]),
            )
        else:
            canonical = model_module.build_model(
                arch,
                int(payload["num_features"]),
                int(payload["num_classes"]),
                dropout=float(spec["dropout"]),
            ).to(device)
            model = canonical
            assert_model(model, model_module)
            if mode == POST_SELECTION_TEST_MODE:
                load_retrain_checkpoint(
                    spec["checkpoint_path"],
                    model,
                    fingerprint=fingerprint,
                    architecture=arch_desc,
                    dataset=str(spec["dataset"]),
                )
            else:
                opt = torch.optim.Adam(
                    model.parameters(),
                    lr=float(spec["lr"]),
                    weight_decay=float(spec["weight_decay"]),
                )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compute_started = time.perf_counter()
        timings["process_spawn_s"] = max(
            0.0,
            compute_started
            - parent_launch
            - timings["cuda_init_s"]
            - timings["data_load_h2d_s"],
        )
        if mode == POST_SELECTION_TEST_MODE:
            inferred = score_full_graph_inference(
                model,
                dataset=str(spec["dataset"]),
                device=device,
                splits=("test",),
                x=x,
                edge_index=edge_index,
                labels=labels,
                masks={"test": test_mask},
            )
            if inferred.test_acc is None:
                raise AssertionError("post-selection test produced no test_acc")
            value = inferred.test_acc
            metrics = {
                "test_acc": value,
                "scorer": "full_test_post_selection",
                "post_selection_test": True,
                "checkpoint_path": str(spec["checkpoint_path"]),
                "inference_implementation": inferred.implementation,
                "exact_inference_s": inferred.inference_s,
                "graph_cache_hit": graph_cache_hit,
            }
        elif mode == "proxy":
            epochs = int(spec["epochs"])
            proxy_train = train_proxy_fixed_epochs
            proxy_kwargs = {}
            if spec.get("proxy_batch_targets") is not None:
                from .fixed_proxy import payload_index, train_proxy_batches
                proxy_train = train_proxy_batches
                proxy_kwargs = dict(batch_targets=int(spec["proxy_batch_targets"]), seed=seed,
                    ap_targets=bool(spec.get("proxy_ap_targets", False)),
                    index=payload_index(payload, edge_indices, device))
            trained = proxy_train(
                model,
                opt,
                x,
                edge_indices,
                labels,
                train_mask,
                val_mask,
                epochs=epochs,
                **proxy_kwargs,
            )
            value = trained.sub_acc
            metrics = {
                "sub_acc": value,
                "epochs_ran": epochs,
                "msg1": msg1,
                "msg2": msg2,
                "E_msg_sub": expected,
                "scorer": f"pv@{epochs}",
                "cross_shard_comparable": False,
                "graph_cache_hit": graph_cache_hit,
            }
            if spec.get("proxy_steps") is not None or spec.get("proxy_batch_targets") is not None:
                metrics.update(proxy_steps=epochs, proxy_batch_targets=spec.get("proxy_batch_targets"),
                    proxy_target_seed=seed, proxy_ap_targets=bool(spec.get("proxy_ap_targets", False)),
                    proxy_target_rng=("ap_fresh_loader_first_batch_v1; isolated CPU RNG(model_seed)"
                        if spec.get("proxy_ap_targets") else "independent torch.Generator(model_seed)"),
                    fixed_index_load_s=payload.get("_proxy_fixed_index_load_s", 0.0))
            if spec.get("retain_proxy_top1", False):
                retain_started = time.perf_counter()
                metrics["proxy_top1_retained"] = retain_proxy_top1(
                    runtime, spec, canonical, value
                )
                metrics["proxy_top1_retention_s"] = time.perf_counter() - retain_started
        else:
            if mode == "probe":
                model.train()
                opt.zero_grad(set_to_none=True)
                out = model(x, edge_index)
                loss = F.cross_entropy(out[train_mask], labels[train_mask])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite probe loss: {loss.item()}")
                loss.backward()
                epochs = 0
            else:
                epochs = int(spec["epochs"])
                for epoch_index in range(epochs):
                    model.train()
                    opt.zero_grad(set_to_none=True)
                    out = model(x, edge_index)
                    loss = F.cross_entropy(out[train_mask], labels[train_mask])
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite full loss: {loss.item()}")
                    loss.backward()
                    opt.step()
                    if (
                        record_validation
                        and (epoch_index + 1) % 5 == 0
                        and epoch_index + 1 < epochs
                    ):
                        accuracy, evaluation_wall_s = _record_retrain_validation(
                            model,
                            dataset=str(spec["dataset"]),
                            device=device,
                            x=x,
                            edge_index=edge_index,
                            labels=labels,
                            masks={"val": val_mask},
                        )
                        validation_curve.append(
                            {
                                "epoch": epoch_index + 1,
                                "val_acc": accuracy,
                                "evaluation_wall_s": evaluation_wall_s,
                            }
                        )
            if mode == "probe":
                value = 1.0
                metrics = {
                    "probe_forward_backward": True,
                    "scorer": "feasibility",
                    "cross_shard_comparable": False,
                    "graph_cache_hit": graph_cache_hit,
                }
            else:
                inferred = score_full_graph_inference(
                    model,
                    dataset=str(spec["dataset"]),
                    device=device,
                    splits=("val",),
                    x=x,
                    edge_index=edge_index,
                    labels=labels,
                    masks={"val": val_mask},
                )
                if inferred.val_acc is None:
                    raise AssertionError("full validation produced no val_acc")
                value = inferred.val_acc
                metrics = {
                    "val_acc": value,
                    "epochs_ran": epochs,
                    "scorer": f"full_val@{epochs}",
                    "cross_shard_comparable": True,
                    "inference_implementation": inferred.implementation,
                    "exact_inference_s": inferred.inference_s,
                    "graph_cache_hit": graph_cache_hit,
                }
                if mode == "retrain":
                    if record_validation:
                        metrics["retrain_validation_recording_overhead_s"] = sum(
                            point["evaluation_wall_s"] for point in validation_curve
                        )
                        validation_curve.append(
                            {
                                "epoch": epochs,
                                "val_acc": value,
                                "evaluation_wall_s": inferred.inference_s,
                            }
                        )
                        metrics["retrain_validation_curve"] = validation_curve
                        metrics["retrain_validation_interval"] = 5
                    checkpoint = save_retrain_checkpoint(
                        spec["checkpoint_path"],
                        model,
                        fingerprint=fingerprint,
                        architecture=arch_desc,
                        dataset=str(spec["dataset"]),
                    )
                    metrics.update(
                        {
                            "checkpoint_path": str(checkpoint),
                            "test_isolation": "RETRAIN_VAL_ONLY",
                        }
                    )
        assert_test_metric_isolated(mode, metrics)
        metrics["seed"] = seed
        metrics["seed_policy"] = seed_policy
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings["train_compute_s"] = time.perf_counter() - compute_started
        metrics["train_wall_s"] = timings["train_compute_s"]
        allocated, reserved = _memory(device)
    except FloatingPointError as exc:
        status = Status.DIVERGE
        error = str(exc)
        if compute_started is not None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings["train_compute_s"] = time.perf_counter() - compute_started
        allocated, reserved = _memory(device)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        status = Status.PROBE_OOM if mode == "probe" else Status.OOM
        error = str(exc)[:1000]
        if compute_started is not None:
            timings["train_compute_s"] = time.perf_counter() - compute_started
        allocated, reserved = _memory(device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    finally:
        if compute_started is None:
            timings["process_spawn_s"] = max(
                0.0,
                time.perf_counter()
                - parent_launch
                - timings["cuda_init_s"]
                - timings["data_load_h2d_s"],
            )
        teardown_started = time.perf_counter()
        out = loss = None
        opt = None
        model = None
        canonical = None
        payload = None
        x = labels = train_mask = val_mask = test_mask = edge_index = None
        edge_indices = None
        if device.type == "cuda" and runtime is None:
            torch.cuda.empty_cache()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            cuda_boundary_synchronized = True
        timings["teardown_s"] = time.perf_counter() - teardown_started

    metrics.update(timings)
    metrics["cuda_boundary_synchronized"] = cuda_boundary_synchronized
    metrics["cuda_boundary_contract"] = (
        "torch.cuda.synchronize immediately before cell return"
        if device.type == "cuda"
        else "cpu_no_cuda_work"
    )
    return CellResult(
        status=status,
        fingerprint=fingerprint,
        phase=phase,
        gpu=int(spec["gpu"]),
        round=int(spec["round"]),
        shard=int(spec["shard"]),
        architecture=arch_desc,
        value=value,
        wall_s=sum(timings.values()),
        peak_allocated_gb=allocated,
        peak_reserved_gb=reserved,
        error=error,
        metrics=metrics,
    )
