

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np
import torch
import torch.nn.functional as F

from sregnas.core import Architecture

from sregnas.data.products import (
    PRODUCTS_BATCH_SIZE,
    PRODUCTS_FANOUT,
    PRODUCTS_INFERENCE_BATCH_SIZE,
    PRODUCTS_RETRAIN_PASSES,
    InferenceCandidateSet,
    ProductsMmapGraph,
    ProductsNeighborBatch,
    ReusableProductsNeighborSampler,
)
from sregnas.models.canonical import assert_model, load_model_module
from sregnas.core.contracts import CellResult, Status
from sregnas.evaluation.inference import score_full_graph_inference
from sregnas.training.fullbatch import PerLayerCanonicalModel
from sregnas.evaluation.checkpoints import (
    POST_SELECTION_TEST_MODE,
    assert_test_metric_isolated,
    load_retrain_checkpoint,
    save_retrain_checkpoint,
)


PRODUCTS_PROBE_SCHEMA = "products_retrain_shape_probe_v1"
PRODUCTS_TIMING_SCHEMA = "products_minibatch_v1"
PRODUCTS_REFERENCE_FANOUT = (15, 10, 5)
PRODUCTS_REFERENCE_BATCH_SIZE = 1024
PRODUCTS_REFERENCE_PRODUCTION_BATCH_SIZE = 4096
PRODUCTS_REFERENCE_BATCH_SIZES = (
    PRODUCTS_REFERENCE_BATCH_SIZE,
    PRODUCTS_REFERENCE_PRODUCTION_BATCH_SIZE,
)
PRODUCTS_REFERENCE_PASSES = 25
PRODUCTS_REFERENCE_FOLLOWUP_PASSES = 50
PRODUCTS_REFERENCE_CONTROL_CONTRACTS = {
    "A_published_batch1024": (
        PRODUCTS_REFERENCE_BATCH_SIZE,
        PRODUCTS_REFERENCE_PASSES,
    ),
    "B_production_batch4096": (
        PRODUCTS_REFERENCE_PRODUCTION_BATCH_SIZE,
        PRODUCTS_REFERENCE_PASSES,
    ),
    "C_production_batch4096_pass50": (
        PRODUCTS_REFERENCE_PRODUCTION_BATCH_SIZE,
        PRODUCTS_REFERENCE_FOLLOWUP_PASSES,
    ),
}
PRODUCTS_REFERENCE_DROPOUT = 0.5
PRODUCTS_REFERENCE_ARCH_DESC = [
    {
        "encoding_op": "const",
        "aggregation_op": "mean",
        "activation": "relu",
        "num_heads": 1,
        "hidden_dim": 256,
    }
    for _ in range(3)
]
PRODUCTS_REFERENCE_ARCH_DIFFERENCES = (
    "GNAS const/mean uses one shared linear projection and mean over sampled "
    "neighbors plus an added self-loop; canonical SAGEConv has distinct root "
    "and neighbor transforms. DistNAS also keeps three 256-wide message-passing "
    "layers followed by a separate linear classifier, and applies ReLU after "
    "the third layer. It is therefore a pipeline control, not a claim of exact "
    "GraphSAGE operator identity."
)


def _hash_seed(*parts: object) -> int:
    payload = json.dumps(parts, separators=(",", ":"), ensure_ascii=True)
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big"
    ) % (2**31)


def retrain_root_seed(fingerprint: str, replica: int) -> int:
    pass
    return _hash_seed(fingerprint, "retrain", int(replica))


def search_sampling_seed(
    fingerprint: str,
    *,
    side: str,
    controller: int,
    replica: int = 0,
) -> int:
    pass
    return _hash_seed(
        fingerprint,
        "search",
        str(side),
        int(controller),
        int(replica),
    )


def _subseed(root: int, stream: str, *coordinates: int) -> int:
    return _hash_seed(int(root), stream, *(int(value) for value in coordinates))


def set_retrain_seed(root_seed: int) -> None:
    random.seed(root_seed)
    np.random.seed(root_seed)
    torch.manual_seed(root_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(root_seed)


def sampled_inference_batch_size(dataset: str) -> int:
    return 128 if dataset.lower() == "reddit" else PRODUCTS_INFERENCE_BATCH_SIZE


def products_probe_signature(
    graph: ProductsMmapGraph,
    architecture: list[dict[str, Any]],
) -> dict[str, Any]:
    signature = {
        "schema": PRODUCTS_PROBE_SCHEMA,
        "dataset": graph.manifest["dataset"],
        "graph_sha256": graph.manifest["graph_sha256"],
        "fanout": list(PRODUCTS_FANOUT),
        "train_batch_size": PRODUCTS_BATCH_SIZE,
        "inference": "exact_layerwise_full_neighbor",
        "inference_batch_size": sampled_inference_batch_size(graph.manifest["dataset"]),
        "architecture": architecture,
        "dtype": "float32",
    }
    from sregnas.models import normalization_fix
    if normalization_fix.enabled():
        from pathlib import Path
        signature['normalization_fix'] = dict(schema=normalization_fix.SCHEMA,
            source_sha256=hashlib.sha256(Path(normalization_fix.__file__).read_bytes()).hexdigest())
        if normalization_fix.softmax_max_enabled():
            signature['pi_softmax_max_fix'] = 'pi_softmax_max_v1'
    return signature


@dataclass(frozen=True)
class _ShapeOnlyFeatures:
    width: int


def _copy_rows(
    array: np.ndarray | torch.Tensor | _ShapeOnlyFeatures,
    ids: np.ndarray,
) -> torch.Tensor:
    if isinstance(array, _ShapeOnlyFeatures):
        return torch.zeros((len(ids), array.width), dtype=torch.float32)
    if isinstance(array, torch.Tensor):
        return array.index_select(0, torch.from_numpy(ids).long()).contiguous()


    return torch.from_numpy(np.array(array[ids], copy=True)).contiguous()


def _torch_edge(edge_index: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(edge_index).long().contiguous().to(device)


@dataclass
class ProductsTrainResult:
    passes: int
    optimizer_steps: int
    train_seed_nodes: int
    final_batch_size: int
    final_loss: float
    sampler_loader_init_s: float
    sampling_s: float
    h2d_s: float
    train_compute_s: float
    total_s: float


def train_products_minibatch(
    model: PerLayerCanonicalModel,
    graph: ProductsMmapGraph,
    sampler: ReusableProductsNeighborSampler,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    root_seed: int,
    passes: int = PRODUCTS_RETRAIN_PASSES,
    pass_callback=None,
) -> ProductsTrainResult:
    pass
    if hasattr(sampler, "executor_options"):
        from sregnas.training.executor import train_resident

        return train_resident(
            model,
            graph,
            sampler,
            optimizer,
            device=device,
            root_seed=root_seed,
            passes=passes,
            pass_callback=pass_callback,
        )
    if passes <= 0:
        raise ValueError("products passes must be positive")
    started = time.perf_counter()
    sampling_s = h2d_s = train_compute_s = 0.0
    steps = seen = 0
    final_loss = float("nan")
    final_batch_size = 0
    expected_train_nodes = int(graph.train_mask.sum())

    for pass_index in range(passes):
        pass_started = time.perf_counter()
        before = (sampling_s, h2d_s, train_compute_s)
        pass_seen = 0
        loader_started = time.perf_counter()
        batches = sampler.seed_batches(
            "train",
            seed=_subseed(root_seed, "train_order", pass_index),
            shuffle=True,
        )

        sampling_s += time.perf_counter() - loader_started
        for batch_index, seed_nodes in enumerate(batches):
            final_batch_size = int(seed_nodes.size)
            sample_started = time.perf_counter()
            batch = sampler.sample(
                seed_nodes,
                seed=_subseed(root_seed, "neighbor_sample", pass_index, batch_index),
            )
            sampling_s += time.perf_counter() - sample_started

            h2d_started = time.perf_counter()
            x = _copy_rows(graph.x, batch.node_ids).to(device)
            labels = _copy_rows(graph.labels, batch.seed_nodes).long().to(device)
            edges = [
                _torch_edge(batch.edge_index_layer1, device),
                _torch_edge(batch.edge_index_layer2, device),
            ]
            from sregnas.models.normalization_fix import enabled, attach_degrees
            if enabled():
                edges = attach_degrees(graph, sampler, batch, edges, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            h2d_s += time.perf_counter() - h2d_started

            compute_started = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, edges)[: batch.batch_size]
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite products loss: {float(loss.item())}"
                )
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            train_compute_s += time.perf_counter() - compute_started
            final_loss = float(loss.detach().item())
            steps += 1
            pass_seen += batch.batch_size
        if pass_seen != expected_train_nodes:
            raise AssertionError(
                f"products pass {pass_index} saw {pass_seen} train seeds; "
                f"expected {expected_train_nodes}"
            )
        seen += pass_seen
        if pass_callback:
            pass_callback(
                dict(
                    pass_index=pass_index,
                    wall_s=time.perf_counter() - pass_started,
                    sampling_s=sampling_s - before[0],
                    h2d_s=h2d_s - before[1],
                    train_compute_s=train_compute_s - before[2],
                )
            )

    total_s = time.perf_counter() - started
    return ProductsTrainResult(
        passes=passes,
        optimizer_steps=steps,
        train_seed_nodes=seen,
        final_batch_size=final_batch_size,
        final_loss=final_loss,
        sampler_loader_init_s=max(0.0, total_s - sampling_s - h2d_s - train_compute_s),
        sampling_s=sampling_s,
        h2d_s=h2d_s,
        train_compute_s=train_compute_s,
        total_s=total_s,
    )


def train_products_reference_control(
    model: PerLayerCanonicalModel,
    graph: ProductsMmapGraph,
    sampler: ReusableProductsNeighborSampler,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    root_seed: int,
    batch_size: int,
    passes: int = PRODUCTS_REFERENCE_PASSES,
) -> ProductsTrainResult:
    pass
    if batch_size not in PRODUCTS_REFERENCE_BATCH_SIZES:
        raise ValueError(
            "products reference batch size must be literature 1024 or "
            "production 4096"
        )
    if passes not in {
        PRODUCTS_REFERENCE_PASSES,
        PRODUCTS_REFERENCE_FOLLOWUP_PASSES,
    }:
        raise ValueError("products reference passes must be 25 or 50")
    started = time.perf_counter()
    sampling_s = h2d_s = train_compute_s = 0.0
    steps = seen = 0
    final_loss = float("nan")
    final_batch_size = 0
    expected_train_nodes = int(graph.train_mask.sum())
    for pass_index in range(passes):
        pass_seen = 0
        batches = sampler.seed_batches_with_size(
            "train",
            seed=_subseed(root_seed, "reference_train_order", pass_index),
            shuffle=True,
            batch_size=batch_size,
        )
        for batch_index, seed_nodes in enumerate(batches):
            final_batch_size = int(seed_nodes.size)
            sample_started = time.perf_counter()
            batch = sampler.sample_with_fanouts(
                seed_nodes,
                fanouts=PRODUCTS_REFERENCE_FANOUT,
                seed=_subseed(
                    root_seed,
                    "reference_neighbor_sample",
                    pass_index,
                    batch_index,
                ),
            )
            sampling_s += time.perf_counter() - sample_started
            h2d_started = time.perf_counter()
            x = _copy_rows(graph.x, batch.node_ids).to(device)
            labels = _copy_rows(graph.labels, batch.seed_nodes).long().to(device)
            edges = [_torch_edge(edge, device) for edge in batch.edge_indices]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            h2d_s += time.perf_counter() - h2d_started
            compute_started = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, edges)[: batch.batch_size]
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite products reference loss: {float(loss.item())}"
                )
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            train_compute_s += time.perf_counter() - compute_started
            final_loss = float(loss.detach().item())
            steps += 1
            pass_seen += batch.batch_size
        if pass_seen != expected_train_nodes:
            raise AssertionError("reference pass did not cover every train seed")
        seen += pass_seen
    total_s = time.perf_counter() - started
    return ProductsTrainResult(
        passes=passes,
        optimizer_steps=steps,
        train_seed_nodes=seen,
        final_batch_size=final_batch_size,
        final_loss=final_loss,
        sampler_loader_init_s=max(0.0, total_s - sampling_s - h2d_s - train_compute_s),
        sampling_s=sampling_s,
        h2d_s=h2d_s,
        train_compute_s=train_compute_s,
        total_s=total_s,
    )


def _exact_gcn_target_forward(
    conv: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    global_nodes: np.ndarray,
    target_count: int,
    global_degrees_with_loops: torch.Tensor,
) -> torch.Tensor:
    pass
    h = conv.lin(x)
    source, target = edge_index
    targets = torch.arange(target_count, device=x.device)
    source = torch.cat([source, targets])
    target = torch.cat([target, targets])
    node_ids = (
        global_nodes.long()
        if isinstance(global_nodes, torch.Tensor)
        else torch.from_numpy(global_nodes).long()
    ).to(global_degrees_with_loops.device)
    degrees = global_degrees_with_loops.index_select(0, node_ids).to(x.device)
    norm = degrees.to(dtype=h.dtype).pow(-0.5)
    messages = h.index_select(0, source) * (
        norm.index_select(0, source) * norm.index_select(0, target)
    ).unsqueeze(1)
    out = torch.zeros((target_count, conv.out_channels), dtype=h.dtype, device=x.device)
    out.index_add_(0, target, messages)
    if conv.aggregation_op == "mlp":
        out = conv.mlp_agg(out)
    return out + conv.bias


def _exact_layer_batch(
    conv: torch.nn.Module,
    activation_name: str,
    features: np.ndarray | torch.Tensor | _ShapeOnlyFeatures,
    graph: ProductsMmapGraph,
    sampler: ReusableProductsNeighborSampler,
    targets: np.ndarray,
    *,
    device: torch.device,
    global_degrees_with_loops: torch.Tensor,
) -> torch.Tensor:
    if isinstance(targets, torch.Tensor) and targets.is_cuda:
        nodes, edge_index = sampler.full_neighbor_edges_device(targets)
        x = (
            torch.zeros((nodes.numel(), features.width), device=device)
            if isinstance(features, _ShapeOnlyFeatures)
            else features.index_select(0, nodes).float()
        )
    else:
        nodes, edge_np = sampler.full_neighbor_edges(targets)
        x = _copy_rows(features, nodes).float().to(device)
        edge_index = _torch_edge(edge_np, device)
    if conv.encoding_op == "gcn":
        out = _exact_gcn_target_forward(
            conv,
            x,
            edge_index,
            nodes,
            len(targets),
            global_degrees_with_loops,
        )
    else:
        out = conv(x, edge_index)[: len(targets)]

    activation = {
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "relu": torch.relu,
        "none": lambda value: value,
        "softplus": F.softplus,
        "leaky_relu": F.leaky_relu,
        "relu6": F.relu6,
        "elu": F.elu,
    }[activation_name]
    return activation(out)


@dataclass
class ProductsInferenceResult:
    val_acc: float | None
    test_acc: float | None
    layer_batches: int
    inference_s: float
    logits: torch.Tensor | None = None


def exact_layerwise_full_neighbor_inference(
    canonical_model: torch.nn.Module,
    graph: ProductsMmapGraph,
    sampler: ReusableProductsNeighborSampler,
    *,
    device: torch.device,
    splits: tuple[Literal["val", "test"], ...],
    batch_size: int = PRODUCTS_INFERENCE_BATCH_SIZE,
    collect_logits: bool = False,
) -> ProductsInferenceResult:
    pass
    if batch_size <= 0:
        raise ValueError("inference batch size must be positive")
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("inference splits must be non-empty and unique")
    if not set(splits) <= {"val", "test"}:
        raise ValueError(f"unsupported inference splits: {splits}")
    if getattr(sampler, "executor_options", {}).get("inference_residency") == "gpu":
        from sregnas.training.executor import exact_resident

        return exact_resident(
            canonical_model,
            graph,
            sampler,
            device=device,
            splits=splits,
            batch_size=batch_size,
            collect_logits=collect_logits,
        )
    started = time.perf_counter()
    canonical_model.eval()
    features: np.ndarray | torch.Tensor = graph.x
    layer_batches = 0
    degrees = torch.from_numpy(np.array(np.diff(graph.rowptr) + 1, copy=True)).float()
    with torch.no_grad():
        for conv, activation_name in zip(
            canonical_model.convs, canonical_model._act_names
        ):
            output = torch.empty(
                (graph.num_nodes, conv.out_channels),
                dtype=torch.float32,
                device="cpu",
            )
            for targets in sampler.inference_batches(batch_size):
                values = _exact_layer_batch(
                    conv,
                    activation_name,
                    features,
                    graph,
                    sampler,
                    targets,
                    device=device,
                    global_degrees_with_loops=degrees,
                )
                output[torch.from_numpy(targets).long()] = values.cpu()
                layer_batches += 1
            features = output

        correct = {split: 0 for split in splits}
        count = {split: 0 for split in splits}
        logits_out = (
            torch.empty((graph.num_nodes, graph.num_classes))
            if collect_logits
            else None
        )
        for targets in sampler.inference_batches(batch_size):
            hidden = _copy_rows(features, targets).to(device)
            logits = canonical_model.classifier(hidden)
            predictions = logits.argmax(dim=1).cpu()
            labels = _copy_rows(graph.labels, targets).long()
            for split in splits:
                mask_array = graph.val_mask if split == "val" else graph.test_mask
                mask = _copy_rows(mask_array, targets).bool()
                correct[split] += int((predictions[mask] == labels[mask]).sum())
                count[split] += int(mask.sum())
            if logits_out is not None:
                logits_out[torch.from_numpy(targets).long()] = logits.cpu()
    empty = [split for split in splits if not count[split]]
    if empty:
        raise AssertionError(f"products exact inference split is empty: {empty}")
    return ProductsInferenceResult(
        val_acc=(correct["val"] / count["val"] if "val" in splits else None),
        test_acc=(correct["test"] / count["test"] if "test" in splits else None),
        layer_batches=layer_batches,
        inference_s=time.perf_counter() - started,
        logits=logits_out,
    )


def analytical_inference_bound_bytes(
    architecture: list[dict[str, Any]],
    candidate_set: InferenceCandidateSet,
    *,
    input_dim: int,
    dtype_bytes: int = 4,
) -> dict[str, Any]:
    pass
    edge_count = (
        candidate_set.analytical_combined_bound["incident_edges"]
        + candidate_set.analytical_combined_bound["target_nodes"]
    )
    unique_sources = candidate_set.analytical_combined_bound["unique_source_nodes"]
    target_nodes = candidate_set.analytical_combined_bound["target_nodes"]
    rows = []
    in_width = int(input_dim)
    for layer_index, layer in enumerate(architecture):
        out_width = int(layer["hidden_dim"])
        heads = int(layer["num_heads"])
        source_features = unique_sources * in_width * dtype_bytes
        transformed_sources = unique_sources * out_width * dtype_bytes
        messages = edge_count * out_width * dtype_bytes
        attention_scores = (
            edge_count * heads * dtype_bytes
            if layer["encoding_op"] in {"gat", "sym-gat"}
            else 0
        )
        target_output = target_nodes * out_width * dtype_bytes
        combined = (
            source_features
            + transformed_sources
            + messages
            + attention_scores
            + target_output
        )
        rows.append(
            {
                "layer": layer_index,
                "input_width": in_width,
                "output_width": out_width,
                "source_feature_bytes": source_features,
                "transformed_source_bytes": transformed_sources,
                "message_bytes": messages,
                "attention_score_bytes": attention_scores,
                "target_output_bytes": target_output,
                "combined_bytes": combined,
            }
        )
        in_width = out_width
    return {
        "kind": "analytical_combined_Emax_Umax_bound",
        "edge_count_including_self_loops": edge_count,
        "unique_source_nodes": unique_sources,
        "layers": rows,
        "peak_layer_bytes": max(row["combined_bytes"] for row in rows),
    }


@dataclass
class ProductsProbeResult:
    mode: str
    training_steps: int
    inference_layer_batches: int
    analytical_bound: dict[str, Any]
    total_s: float
    candidate_set_s: float
    training_shape_s: float
    inference_shape_s: float


def probe_products_retrain_shapes(
    model: PerLayerCanonicalModel,
    graph: ProductsMmapGraph,
    sampler: ReusableProductsNeighborSampler,
    architecture: list[dict[str, Any]],
    *,
    device: torch.device,
    root_seed: int,
    mode: Literal["worst_case_upper_bound", "full_mirror"],
) -> ProductsProbeResult:
    pass
    started = time.perf_counter()
    candidate_started = time.perf_counter()
    inference_batch_size = sampled_inference_batch_size(graph.manifest["dataset"])
    candidates = sampler.inference_candidate_set(
        min(inference_batch_size, graph.num_nodes)
    )
    candidate_s = time.perf_counter() - candidate_started
    analytical = analytical_inference_bound_bytes(
        architecture, candidates, input_dim=graph.num_features
    )

    train_started = time.perf_counter()
    train_seeds = next(
        sampler.seed_batches(
            "train", seed=_subseed(root_seed, "probe_train_order"), shuffle=True
        )
    )
    if hasattr(sampler, "executor_options"):
        from sregnas.training.executor import (
            GPUProductsNeighborSampler,
            training_inputs,
        )

        sample_fn = (
            sampler.sample_device
            if isinstance(sampler, GPUProductsNeighborSampler)
            else sampler.sample
        )
        sampled = sample_fn(
            train_seeds, seed=_subseed(root_seed, "probe_neighbor_sample")
        )
        x, edges, labels = training_inputs(graph, sampler, sampled, device=device)
    else:
        sampled = sampler.sample(
            train_seeds, seed=_subseed(root_seed, "probe_neighbor_sample")
        )
        x = _copy_rows(graph.x, sampled.node_ids).to(device)
        labels = _copy_rows(graph.labels, sampled.seed_nodes).long().to(device)
        edges = [
            _torch_edge(sampled.edge_index_layer1, device),
            _torch_edge(sampled.edge_index_layer2, device),
        ]
    model.train()
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x, edges)[: sampled.batch_size], labels)
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_s = time.perf_counter() - train_started

    inference_started = time.perf_counter()
    canonical = model.canonical_model
    canonical.eval()
    layer_batches = 0
    degrees = torch.from_numpy(np.array(np.diff(graph.rowptr) + 1, copy=True)).float()
    resident_inference = (
        getattr(sampler, "executor_options", {}).get("inference_residency") == "gpu"
    )
    if resident_inference:
        from sregnas.training.executor import device_sampler, resident_features

        inference_sampler = device_sampler(sampler)
        initial_features, _ = resident_features(sampler, graph, device)
        degrees = degrees.to(device)
        resident_layer_input = initial_features
    else:
        inference_sampler, initial_features = sampler, graph.x
    with torch.no_grad():
        features: np.ndarray | torch.Tensor | _ShapeOnlyFeatures = initial_features
        for conv, activation_name in zip(canonical.convs, canonical._act_names):
            if resident_inference:



                resident_layer_output = torch.empty(
                    (graph.num_nodes, conv.out_channels),
                    dtype=torch.float32,
                    device=device,
                )
            targets_iter: Iterable[np.ndarray]
            if mode == "full_mirror":
                targets_iter = sampler.inference_batches(
                    min(inference_batch_size, graph.num_nodes)
                )
            elif mode == "worst_case_upper_bound":
                targets_iter = (candidates.edge_batch, candidates.source_batch)
            else:
                raise ValueError(f"unknown products probe mode: {mode}")
            for targets in targets_iter:
                if resident_inference:
                    targets = torch.as_tensor(targets, device=device, dtype=torch.int64)
                _exact_layer_batch(
                    conv,
                    activation_name,
                    features,
                    graph,
                    inference_sampler,
                    targets,
                    device=device,
                    global_degrees_with_loops=degrees,
                )
                layer_batches += 1


            features = _ShapeOnlyFeatures(conv.out_channels)
            if resident_inference:
                resident_layer_input = resident_layer_output
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_s = time.perf_counter() - inference_started
    return ProductsProbeResult(
        mode=mode,
        training_steps=1,
        inference_layer_batches=layer_batches,
        analytical_bound=analytical,
        total_s=time.perf_counter() - started,
        candidate_set_s=candidate_s,
        training_shape_s=training_s,
        inference_shape_s=inference_s,
    )


class ProductsRuntimeCache:
    pass

    def __init__(self):
        self._entries: dict[
            str, tuple[ProductsMmapGraph, ReusableProductsNeighborSampler]
        ] = {}

    def get(self, manifest_path: str, *, device=None, **executor_options) -> tuple[
        ProductsMmapGraph,
        ReusableProductsNeighborSampler,
        bool,
    ]:
        from sregnas.training.executor import DEFAULTS, configure_executor

        opts = {k: executor_options.get(k, v) for k, v in DEFAULTS.items()}
        key = (str(manifest_path), str(device), *opts.values())
        cached = self._entries.get(key)
        if cached is not None:
            return cached[0], cached[1], True
        graph = ProductsMmapGraph.open(manifest_path)
        sampler = configure_executor(
            graph, ReusableProductsNeighborSampler(graph), device=device, **opts
        )
        self._entries[key] = (graph, sampler)
        return graph, sampler, False


def _products_peak_memory(
    device: torch.device,
) -> tuple[float | None, float | None]:
    if device.type != "cuda":
        return None, None
    gib = float(1024**3)
    return (
        torch.cuda.max_memory_allocated(device) / gib,
        torch.cuda.max_memory_reserved(device) / gib,
    )


def run_products_cell(
    spec: dict[str, Any],
    *,
    device: torch.device | None = None,
    runtime: ProductsRuntimeCache | None = None,
) -> CellResult:
    pass
    cell_started = time.perf_counter()
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mode = str(spec["mode"])
    dataset = str(spec.get("dataset", "ogbn-products")).lower()
    if dataset not in {"ogbn-products", "reddit"}:
        raise ValueError("sampled cells support only Products and Reddit")
    if dataset == "reddit" and mode not in {
        "retrain",
        "probe",
        POST_SELECTION_TEST_MODE,
    }:
        raise ValueError(
            "Reddit full-neighbor baseline search requires a separate adapter"
        )
    fingerprint = str(spec["fingerprint"])
    architecture = spec["architecture"]
    status: str = Status.OK
    error: str | None = None
    value: float | None = None
    allocated = reserved = None
    metrics: dict[str, Any] = {
        "timing_schema": PRODUCTS_TIMING_SCHEMA,
        "timing_boundary": "before_graph_mmap_sampler_loader_init",
    }
    try:
        loader_started = time.perf_counter()
        cache = runtime or ProductsRuntimeCache()
        graph, sampler, cache_hit = cache.get(
            str(spec["products_graph_manifest_path"]),
            device=device,
            **{
                k: spec[k]
                for k in ("sampler_impl", "feature_residency", "inference_residency")
                if k in spec
            },
        )
        metrics["sampler_loader_init_s"] = time.perf_counter() - loader_started
        metrics["products_graph_cache_hit"] = cache_hit
        if graph.manifest["dataset"] != dataset:
            raise ValueError("sampled graph dataset does not match the requested cell")
        from sregnas.training.executor import DEFAULTS

        metrics["executor_options"] = {
            key: spec.get(key, value) for key, value in DEFAULTS.items()
        }

        replica = int(spec.get("replica", 0))
        if mode in {"retrain", "probe", POST_SELECTION_TEST_MODE}:
            root_seed = retrain_root_seed(fingerprint, replica)
            seed_policy = "H(fingerprint,phase=retrain,replica);side_excluded"
        elif mode == "reference_control":
            root_seed = _hash_seed(fingerprint, "products_reference_control", replica)
            seed_policy = "H(fingerprint,products_reference_control,replica)"
        else:
            root_seed = search_sampling_seed(
                fingerprint,
                side=str(spec.get("side", "baseline")),
                controller=int(spec.get("controller", spec.get("shard", 0))),
                replica=replica,
            )
            seed_policy = "fingerprint_search_side_controller_replica"
        set_retrain_seed(root_seed)

        model_module = load_model_module(force_native_scatter=True)
        arch = Architecture(architecture)
        if arch.fingerprint() != fingerprint:
            raise AssertionError("products cell architecture/fingerprint mismatch")
        canonical = model_module.build_model(
            arch,
            graph.num_features,
            graph.num_classes,
            dropout=float(spec.get("dropout", 0.0)),
        ).to(device)
        model = PerLayerCanonicalModel(canonical).to(device)
        assert_model(model, model_module)
        metrics.update(
            {
                "seed": root_seed,
                "seed_policy": seed_policy,
                "replica": replica,
                "fanout": list(PRODUCTS_FANOUT),
                "batch_size": PRODUCTS_BATCH_SIZE,
            }
        )
        from sregnas.models.normalization_fix import enabled, SCHEMA, softmax_max_enabled
        if enabled():
            metrics['normalization_fix'] = SCHEMA
            if softmax_max_enabled():
                metrics['pi_softmax_max_fix'] = 'pi_softmax_max_v1'

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if mode == POST_SELECTION_TEST_MODE:
            load_retrain_checkpoint(
                spec["checkpoint_path"],
                model,
                fingerprint=fingerprint,
                architecture=architecture,
                dataset=str(spec.get("dataset", "ogbn-products")),
            )
            inferred = score_full_graph_inference(
                canonical,
                dataset=str(spec.get("dataset", "ogbn-products")),
                device=device,
                splits=("test",),
                products_graph=graph,
                products_sampler=sampler,
                products_batch_size=sampled_inference_batch_size(dataset),
            )
            if inferred.test_acc is None:
                raise AssertionError("post-selection test produced no test_acc")
            value = inferred.test_acc
            metrics.update(
                {
                    "test_acc": inferred.test_acc,
                    "exact_inference_s": inferred.inference_s,
                    "inference_layer_batches": inferred.layer_batches,
                    "evaluation": "exact_layerwise_full_neighbor_test_only",
                    "inference_implementation": inferred.implementation,
                    "post_selection_test": True,
                    "checkpoint_path": str(spec["checkpoint_path"]),
                }
            )
        elif mode == "probe":
            probe = probe_products_retrain_shapes(
                model,
                graph,
                sampler,
                architecture,
                device=device,
                root_seed=root_seed,
                mode=str(spec.get("probe_mode", "worst_case_upper_bound")),
            )
            value = 1.0
            device_total_bytes = (
                int(torch.cuda.get_device_properties(device).total_memory)
                if device.type == "cuda"
                else None
            )
            analytical_within_device = (
                None
                if device_total_bytes is None
                else int(probe.analytical_bound["peak_layer_bytes"])
                <= device_total_bytes
            )
            if analytical_within_device is False:
                status = Status.PROBE_OOM
                value = None
                error = (
                    "analytical combined Emax/Umax inference bound exceeds "
                    "device total memory"
                )
            metrics.update(
                {
                    "scorer": "products_retrain_shape_feasibility",
                    "probe_signature": products_probe_signature(graph, architecture),
                    "probe_mode": probe.mode,
                    "probe_total_s": probe.total_s,
                    "probe_candidate_set_s": probe.candidate_set_s,
                    "probe_training_shape_s": probe.training_shape_s,
                    "probe_inference_shape_s": probe.inference_shape_s,
                    "probe_training_steps": probe.training_steps,
                    "probe_inference_layer_batches": (probe.inference_layer_batches),
                    "probe_analytical_bound": probe.analytical_bound,
                    "probe_device_total_bytes": device_total_bytes,
                    "probe_analytical_bound_within_device": (analytical_within_device),
                }
            )
        elif mode == "reference_control":
            if architecture != PRODUCTS_REFERENCE_ARCH_DESC:
                raise AssertionError(
                    "products reference_control arch_desc is not the named control"
                )
            control_name = str(spec["control_name"])
            if control_name not in PRODUCTS_REFERENCE_CONTROL_CONTRACTS:
                raise AssertionError(
                    f"unknown products reference control: {control_name}"
                )
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=float(spec.get("lr", 0.003)),
                weight_decay=float(spec.get("weight_decay", 0.0)),
            )
            reference_batch_size = int(
                spec.get("batch_size", PRODUCTS_REFERENCE_BATCH_SIZE)
            )
            reference_passes = int(spec.get("epochs", PRODUCTS_REFERENCE_PASSES))
            expected_contract = PRODUCTS_REFERENCE_CONTROL_CONTRACTS[control_name]
            if (reference_batch_size, reference_passes) != expected_contract:
                raise AssertionError(
                    "products reference control batch/pass contract mismatch: "
                    f"{control_name} expected {expected_contract}, got "
                    f"{(reference_batch_size, reference_passes)}"
                )
            trained = train_products_reference_control(
                model,
                graph,
                sampler,
                optimizer,
                device=device,
                root_seed=root_seed,
                batch_size=reference_batch_size,
                passes=reference_passes,
            )
            inferred = score_full_graph_inference(
                canonical,
                dataset=str(spec.get("dataset", "ogbn-products")),
                device=device,
                splits=("val", "test"),
                products_graph=graph,
                products_sampler=sampler,
                products_batch_size=sampled_inference_batch_size(dataset),
            )
            if inferred.val_acc is None or inferred.test_acc is None:
                raise AssertionError("reference control requires val and test")
            value = inferred.val_acc
            metrics.update(
                {
                    "control_name": control_name,
                    "arch_desc": architecture,
                    "arch_desc_differences_from_canonical_graphsage": (
                        PRODUCTS_REFERENCE_ARCH_DIFFERENCES
                    ),
                    "fanout": list(PRODUCTS_REFERENCE_FANOUT),
                    "batch_size": reference_batch_size,
                    "final_batch_size": trained.final_batch_size,
                    "passes_ran": trained.passes,
                    "optimizer_steps": trained.optimizer_steps,
                    "train_seed_nodes": trained.train_seed_nodes,
                    "official_train_nodes": int(graph.train_mask.sum()),
                    "split": "ogb_official",
                    "loss_scope": "seed_nodes_only",
                    "sampling_s": trained.sampling_s,
                    "data_h2d_s": trained.h2d_s,
                    "train_compute_s": trained.train_compute_s,
                    "train_total_s": trained.total_s,
                    "val_acc": inferred.val_acc,
                    "test_acc": inferred.test_acc,
                    "exact_inference_s": inferred.inference_s,
                    "evaluation": "exact_layerwise_full_neighbor",
                    "inference_implementation": inferred.implementation,
                    "published_test_reference_pct": 78.70,
                }
            )
        else:
            passes = int(spec.get("epochs", PRODUCTS_RETRAIN_PASSES))
            if mode == "retrain" and passes != PRODUCTS_RETRAIN_PASSES:
                raise AssertionError("products retrain is locked to 25 passes")
            if mode == "cost_calibration" and passes != 1:
                raise AssertionError("products cost calibration is one pass")
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=float(spec.get("lr", 0.01)),
                weight_decay=float(spec.get("weight_decay", 5e-4)),
            )
            from sregnas.training.sampled_clip import final_training_clip
            with final_training_clip(optimizer, dataset=dataset, mode=mode) as clipping:
                trained = train_products_minibatch(
                    model,
                    graph,
                    sampler,
                    optimizer,
                    device=device,
                    root_seed=root_seed,
                    passes=passes,
                )
            metrics["sampled_grad_clip"] = clipping.accounting(trained.optimizer_steps)
            metrics.update(
                {
                    "passes_ran": trained.passes,
                    "optimizer_steps": trained.optimizer_steps,
                    "train_seed_nodes": trained.train_seed_nodes,
                    "final_batch_size": trained.final_batch_size,
                    "final_loss": trained.final_loss,
                    "sampling_s": trained.sampling_s,
                    "data_h2d_s": trained.h2d_s,
                    "train_compute_s": trained.train_compute_s,
                    "train_total_s": trained.total_s,
                }
            )
            if mode in {"full", "retrain", "cost_calibration"}:
                inferred = score_full_graph_inference(
                    canonical,
                    dataset=str(spec.get("dataset", "ogbn-products")),
                    device=device,
                    splits=("val",),
                    products_graph=graph,
                    products_sampler=sampler,
                    products_batch_size=sampled_inference_batch_size(dataset),
                )
                if inferred.val_acc is None:
                    raise AssertionError("validation inference produced no val_acc")
                value = inferred.val_acc if mode != "cost_calibration" else 1.0
                metrics.update(
                    {
                        "val_acc": inferred.val_acc,
                        "exact_inference_s": inferred.inference_s,
                        "inference_layer_batches": inferred.layer_batches,
                        "evaluation": "exact_layerwise_full_neighbor_val_only",
                        "inference_implementation": inferred.implementation,
                    }
                )
                if mode == "retrain":
                    checkpoint = save_retrain_checkpoint(
                        spec["checkpoint_path"],
                        model,
                        fingerprint=fingerprint,
                        architecture=architecture,
                        dataset=str(spec.get("dataset", "ogbn-products")),
                    )
                    metrics.update(
                        {
                            "checkpoint_path": str(checkpoint),
                            "test_isolation": "RETRAIN_VAL_ONLY",
                        }
                    )
                if mode == "cost_calibration":
                    metrics["scorer"] = "one_pass_cost_calibration"
        assert_test_metric_isolated(mode, metrics)
        allocated, reserved = _products_peak_memory(device)
    except FloatingPointError as exc:
        status = Status.DIVERGE
        error = str(exc)
        allocated, reserved = _products_peak_memory(device)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        status = Status.PROBE_OOM if mode == "probe" else Status.OOM
        error = str(exc)[:1000]
        allocated, reserved = _products_peak_memory(device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    metrics["cuda_boundary_synchronized"] = True
    metrics["cuda_boundary_contract"] = (
        "torch.cuda.synchronize immediately before cell return"
        if device.type == "cuda"
        else "cpu_no_cuda_work"
    )
    metrics["cell_total_s"] = time.perf_counter() - cell_started
    metrics["timing_sum_s"] = metrics["cell_total_s"]
    return CellResult(
        status=status,
        fingerprint=fingerprint,
        phase=str(spec["phase"]),
        gpu=int(spec.get("gpu", 0)),
        round=int(spec.get("round", 0)),
        shard=int(spec.get("shard", 0)),
        architecture=architecture,
        value=value,
        wall_s=float(metrics["cell_total_s"]),
        peak_allocated_gb=allocated,
        peak_reserved_gb=reserved,
        error=error,
        metrics=metrics,
    )
