

from __future__ import annotations

import time
from typing import Any

import torch

from sregnas.core import Architecture

from sregnas.models.canonical import assert_model, load_model_module
from sregnas.core.contracts import CellResult, Status
from sregnas.evaluation.inference import score_full_graph_inference
from sregnas.data.products import PRODUCTS_INFERENCE_BATCH_SIZE
from sregnas.training.minibatch import ProductsRuntimeCache
from sregnas.training.fullbatch import (
    GB,
    PerLayerCanonicalModel,
    PersistentCellRuntime,
    _set_seed,
    fingerprint_seed,
    train_proxy_fixed_epochs,
)


ALLOCATION_SCORE_MODE = "allocation_score"
ALLOCATION_TIMING_SCHEMA = "allocation_score_v1"


def _peak_memory(device: torch.device) -> tuple[float | None, float | None]:
    if device.type != "cuda":
        return None, None
    return (
        torch.cuda.max_memory_allocated(device) / GB,
        torch.cuda.max_memory_reserved(device) / GB,
    )


def run_allocation_score_cell(
    spec: dict[str, Any],
    *,
    device: torch.device | None = None,
    runtime: PersistentCellRuntime | None = None,
    products_runtime: ProductsRuntimeCache | None = None,
) -> CellResult:
    pass
    if str(spec.get("mode")) != ALLOCATION_SCORE_MODE:
        raise AssertionError("allocation scorer received a non-allocation mode")
    if spec.get("checkpoint_path") is not None:
        raise AssertionError("allocation scoring must not receive a checkpoint path")

    started = time.perf_counter()
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fingerprint = str(spec["fingerprint"])
    architecture = spec["architecture"]
    dataset = str(spec["dataset"])
    status: str = Status.OK
    error: str | None = None
    value: float | None = None
    allocated = reserved = None
    metrics: dict[str, Any] = {
        "timing_schema": ALLOCATION_TIMING_SCHEMA,
        "checkpoint_persistence": "none_in_memory_only",
        "recorded_search_sub_acc": float(spec["recorded_search_sub_acc"]),
    }

    model: PerLayerCanonicalModel | None = None
    canonical: torch.nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    try:
        seed = fingerprint_seed(fingerprint)
        _set_seed(seed)
        metrics.update(
            {
                "seed": seed,
                "seed_policy": "fingerprint",
                "proxy_protocol": "authoritative_fixed_epoch_shard_proxy",
            }
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        load_started = time.perf_counter()
        local_runtime = runtime or PersistentCellRuntime()
        proxy, proxy_load_s, proxy_cache_hit = local_runtime.load_payload(
            str(spec["proxy_graph_path"]), "proxy"
        )
        metrics.update(
            {
                "proxy_graph_load_s": proxy_load_s,
                "proxy_graph_cache_hit": proxy_cache_hit,
                "proxy_graph_path": str(spec["proxy_graph_path"]),
            }
        )

        products_graph = products_sampler = None
        full = None
        if dataset.lower() == "ogbn-products":
            product_cache = products_runtime or ProductsRuntimeCache()
            products_graph, products_sampler, products_cache_hit = product_cache.get(
                str(spec["products_graph_manifest_path"]),
                device=device,
                **{
                    key: spec[key]
                    for key in (
                        "sampler_impl",
                        "feature_residency",
                        "inference_residency",
                    )
                    if key in spec
                },
            )
            metrics.update(
                {
                    "products_graph_cache_hit": products_cache_hit,
                    "products_graph_manifest_path": str(
                        spec["products_graph_manifest_path"]
                    ),
                }
            )
        else:
            full, full_load_s, full_cache_hit = local_runtime.load_payload(
                str(spec["full_graph_path"]), "full"
            )
            metrics.update(
                {
                    "full_graph_load_s": full_load_s,
                    "full_graph_cache_hit": full_cache_hit,
                    "full_graph_path": str(spec["full_graph_path"]),
                }
            )
        metrics["graph_load_total_s"] = time.perf_counter() - load_started

        model_module = local_runtime.model_module
        arch = Architecture(arch_description=architecture)
        if arch.fingerprint() != fingerprint:
            raise AssertionError("allocation architecture/fingerprint mismatch")
        canonical = model_module.build_model(
            arch,
            int(proxy["num_features"]),
            int(proxy["num_classes"]),
            dropout=float(spec["dropout"]),
        ).to(device)
        model = PerLayerCanonicalModel(canonical).to(device)
        assert_model(model, model_module)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(spec["lr"]),
            weight_decay=float(spec["weight_decay"]),
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_started = time.perf_counter()
        reproduced = train_proxy_fixed_epochs(
            model,
            optimizer,
            proxy["x"],
            [proxy["edge_index_layer1"], proxy["edge_index_layer2"]],
            proxy["labels"],
            proxy["train_mask"],
            proxy["val_mask"],
            epochs=int(spec["epochs"]),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        allocation_train_s = time.perf_counter() - train_started

        recorded = float(spec["recorded_search_sub_acc"])
        metrics.update(
            {
                "reproduced_sub_acc": reproduced.sub_acc,
                "sub_acc_delta": reproduced.sub_acc - recorded,
                "sub_acc_abs_delta": abs(reproduced.sub_acc - recorded),
                "epochs_ran": reproduced.epochs,
                "allocation_train_s": allocation_train_s,
            }
        )

        if dataset.lower() == "ogbn-products":
            inferred = score_full_graph_inference(
                canonical,
                dataset=dataset,
                device=device,
                splits=("val",),
                products_graph=products_graph,
                products_sampler=products_sampler,
                products_batch_size=PRODUCTS_INFERENCE_BATCH_SIZE,
            )
        else:
            assert full is not None
            inferred = score_full_graph_inference(
                canonical,
                dataset=dataset,
                device=device,
                splits=("val",),
                x=full["x"],
                edge_index=full["edge_index"],
                labels=full["labels"],
                masks={"val": full["val_mask"]},
            )
        if inferred.val_acc is None:
            raise AssertionError("allocation full-graph inference returned no val_acc")
        value = inferred.val_acc
        metrics.update(
            {
                "fullgraph_score": value,
                "scorer": "fullval_after_identical_proxy_retrain",
                "cross_shard_comparable": True,
                "exact_inference_s": inferred.inference_s,
                "inference_layer_batches": inferred.layer_batches,
                "inference_implementation": inferred.implementation,
            }
        )
        allocated, reserved = _peak_memory(device)
    except FloatingPointError as exc:
        status = Status.DIVERGE
        error = str(exc)
        allocated, reserved = _peak_memory(device)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        status = Status.PROBE_OOM
        error = str(exc)[:1000]
        allocated, reserved = _peak_memory(device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    finally:
        optimizer = None
        model = None
        canonical = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    metrics["cuda_boundary_synchronized"] = True
    metrics["cell_total_s"] = time.perf_counter() - started
    return CellResult(
        status=status,
        fingerprint=fingerprint,
        phase=str(spec["phase"]),
        gpu=int(spec.get("gpu", 0)),
        round=int(spec.get("round", 0)),
        shard=int(spec["shard"]),
        architecture=architecture,
        value=value,
        wall_s=float(metrics["cell_total_s"]),
        peak_allocated_gb=allocated,
        peak_reserved_gb=reserved,
        error=error,
        metrics=metrics,
    )
