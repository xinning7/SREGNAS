

from __future__ import annotations

import numpy as np
import pytest
import torch

from sregnas.core import Architecture
from sregnas.models.canonical import load_model_module
from sregnas.data.products import (
    ProductsMmapGraph,
    ReusableProductsNeighborSampler,
    build_products_mmap_graph,
)
from sregnas.training.minibatch import (
    exact_layerwise_full_neighbor_inference,
    probe_products_retrain_shapes,
    retrain_root_seed,
    run_products_cell,
    search_sampling_seed,
    train_products_minibatch,
)
from sregnas.training.fullbatch import PerLayerCanonicalModel
from sregnas.evaluation.checkpoints import POST_SELECTION_TEST_MODE


def _dataset():
    edges = torch.tensor(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 4],
            [2, 5],
            [3, 4],
            [3, 6],
            [4, 7],
            [5, 6],
            [6, 7],
        ],
        dtype=torch.long,
    ).t()
    return {
        "x": torch.tensor(
            [[i / 10, (i % 3) / 3, 1.0] for i in range(8)],
            dtype=torch.float32,
        ),
        "edge_index": edges,
        "labels": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        "train_mask": torch.tensor([1, 1, 0, 0, 1, 1, 0, 0]).bool(),
        "val_mask": torch.tensor([0, 0, 1, 0, 0, 0, 1, 0]).bool(),
        "test_mask": torch.tensor([0, 0, 0, 1, 0, 0, 0, 1]).bool(),
        "num_nodes": 8,
        "num_features": 3,
        "num_classes": 2,
    }


def _arch(encoding: str):
    return [
        {
            "encoding_op": encoding,
            "aggregation_op": "sum",
            "activation": "relu",
            "num_heads": 2 if encoding in {"gat", "sym-gat"} else 1,
            "hidden_dim": 4,
        },
        {
            "encoding_op": encoding,
            "aggregation_op": "mean",
            "activation": "elu",
            "num_heads": 2 if encoding in {"gat", "sym-gat"} else 1,
            "hidden_dim": 4,
        },
    ]


def _model(arch):
    model_module = load_model_module(force_native_scatter=True)
    canonical = model_module.build_model(Architecture(arch), 3, 2, dropout=0.0)
    return canonical, PerLayerCanonicalModel(canonical)


def _graph(tmp_path):
    build_products_mmap_graph(_dataset(), tmp_path)
    graph = ProductsMmapGraph.open(tmp_path / "manifest.json")
    return graph, ReusableProductsNeighborSampler(graph)


@pytest.mark.unit
def test_retrain_seed_excludes_side_but_search_seed_keeps_it():
    fp = "7cc89ca7e28804c3"
    assert retrain_root_seed(fp, 0) == retrain_root_seed(fp, 0)
    assert retrain_root_seed(fp, 0) != retrain_root_seed(fp, 1)
    assert search_sampling_seed(fp, side="system", controller=0) != (
        search_sampling_seed(fp, side="baseline", controller=0)
    )


@pytest.mark.unit
def test_minibatch_pass_count_and_optimizer_steps(tmp_path):
    graph, sampler = _graph(tmp_path)
    canonical, model = _model(_arch("const"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    result = train_products_minibatch(
        model,
        graph,
        sampler,
        optimizer,
        device=torch.device("cpu"),
        root_seed=retrain_root_seed("0123456789abcdef", 0),
        passes=2,
    )
    assert result.passes == 2
    assert result.optimizer_steps == 2
    assert result.train_seed_nodes == 2 * int(graph.train_mask.sum())
    assert np.isfinite(result.final_loss)
    del canonical


@pytest.mark.unit
@pytest.mark.parametrize("encoding", ["const", "gcn", "gat"])
def test_layerwise_full_neighbor_matches_canonical_full_graph(tmp_path, encoding):
    graph, sampler = _graph(tmp_path)
    canonical, _ = _model(_arch(encoding))
    row = np.repeat(np.arange(graph.num_nodes, dtype=np.int64), np.diff(graph.rowptr))
    full_edge = torch.from_numpy(
        np.stack([np.asarray(graph.col, dtype=np.int64), row])
    ).long()
    canonical.eval()
    with torch.no_grad():
        expected = canonical(torch.from_numpy(np.array(graph.x)), full_edge)
    actual = exact_layerwise_full_neighbor_inference(
        canonical,
        graph,
        sampler,
        device=torch.device("cpu"),
        splits=("val", "test"),
        batch_size=3,
        collect_logits=True,
    )
    assert actual.layer_batches == 2 * 3
    assert actual.logits is not None
    assert torch.allclose(actual.logits, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.unit
def test_exact_inference_accesses_only_explicit_split(tmp_path):
    graph, sampler = _graph(tmp_path)
    canonical, _ = _model(_arch("const"))
    val_only = exact_layerwise_full_neighbor_inference(
        canonical,
        graph,
        sampler,
        device=torch.device("cpu"),
        splits=("val",),
        batch_size=3,
    )
    assert val_only.val_acc is not None
    assert val_only.test_acc is None


@pytest.mark.unit
def test_candidate_probe_is_four_layer_batches_and_full_audit_is_two(tmp_path):
    graph, sampler = _graph(tmp_path)
    arch = _arch("gat")
    _, model = _model(arch)
    root_seed = retrain_root_seed("fedcba9876543210", 0)
    candidate = probe_products_retrain_shapes(
        model,
        graph,
        sampler,
        arch,
        device=torch.device("cpu"),
        root_seed=root_seed,
        mode="worst_case_upper_bound",
    )
    full = probe_products_retrain_shapes(
        model,
        graph,
        sampler,
        arch,
        device=torch.device("cpu"),
        root_seed=root_seed,
        mode="full_mirror",
    )
    assert candidate.training_steps == full.training_steps == 1
    assert candidate.inference_layer_batches == 4
    assert full.inference_layer_batches == 2
    assert candidate.analytical_bound["kind"] == ("analytical_combined_Emax_Umax_bound")
    assert candidate.analytical_bound["peak_layer_bytes"] > 0


@pytest.mark.unit
def test_cell_timer_starts_before_sampler_loader_initialization(tmp_path):
    graph, _ = _graph(tmp_path)
    arch = _arch("const")
    fingerprint = Architecture(arch).fingerprint()
    result = run_products_cell(
        {
            "phase": "products:baseline:cost_calibration",
            "mode": "cost_calibration",
            "fingerprint": fingerprint,
            "architecture": arch,
            "products_graph_manifest_path": str(graph.root / "manifest.json"),
            "side": "baseline",
            "controller": 0,
            "replica": 0,
            "gpu": 0,
            "round": 0,
            "shard": 0,
            "epochs": 1,
            "lr": 0.01,
            "weight_decay": 5e-4,
            "dropout": 0.0,
        },
        device=torch.device("cpu"),
    )
    assert result.status == "OK"
    assert result.metrics["timing_boundary"] == (
        "before_graph_mmap_sampler_loader_init"
    )
    assert result.metrics["sampler_loader_init_s"] > 0
    assert result.metrics["sampling_s"] >= 0
    assert result.metrics["cell_total_s"] >= result.metrics["train_total_s"]


@pytest.mark.unit
def test_products_retrain_is_val_only_then_explicit_test(tmp_path):
    graph, _ = _graph(tmp_path)
    arch = _arch("const")
    fingerprint = Architecture(arch).fingerprint()
    checkpoint = tmp_path / "products_winner.pt"
    common = {
        "fingerprint": fingerprint,
        "architecture": arch,
        "products_graph_manifest_path": str(graph.root / "manifest.json"),
        "dataset": "ogbn-products",
        "side": "system",
        "controller": 0,
        "replica": 0,
        "gpu": 0,
        "round": 0,
        "shard": 0,
        "lr": 0.01,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "checkpoint_path": str(checkpoint),
    }
    retrain = run_products_cell(
        {
            **common,
            "phase": "products:system:retrain",
            "mode": "retrain",
            "epochs": 25,
        },
        device=torch.device("cpu"),
    )
    assert retrain.status == "OK"
    assert "test_acc" not in retrain.metrics
    assert retrain.metrics["test_isolation"] == "RETRAIN_VAL_ONLY"
    assert checkpoint.exists()

    tested = run_products_cell(
        {
            **common,
            "phase": "products:system:post_selection_test",
            "mode": POST_SELECTION_TEST_MODE,
            "epochs": 0,
        },
        device=torch.device("cpu"),
    )
    assert tested.status == "OK"
    assert tested.metrics["post_selection_test"] is True
    assert "val_acc" not in tested.metrics
    assert tested.value == tested.metrics["test_acc"]
