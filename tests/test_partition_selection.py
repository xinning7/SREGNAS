import json

import numpy as np
import pytest
import torch

from sregnas.config import GraphConfig, load_config
from sregnas.data.partition_selection import (
    balanced_random_membership,
    partition_index,
    select_partition,
)
from sregnas.execution.orchestrator import _graph_cache_key, _validate_partition_cache


def test_balanced_random_matches_frozen_randomstate_construction():
    actual = balanced_random_membership(10, 3, seed=42)
    assert actual.tolist() == [0, 0, 1, 2, 2, 0, 2, 1, 0, 1]
    assert np.bincount(actual).tolist() == [4, 3, 3]
    assert actual.dtype == np.int64


def test_fi_closed_form_and_isolated_node_convention():
    rowptr = np.array([0, 1, 2, 3, 4, 4, 4])
    col = np.array([1, 0, 3, 2])
    labels = np.array([0, 1, 0, 1, 0, 0])
    membership = np.array([0, 0, 1, 1, 0, 1])
    result = partition_index(
        rowptr, col, labels, np.ones(6, dtype=bool), membership, n=2
    )
    assert result["SNR"] == result["LA"] == result["FI"] == 1.0
    assert result["isolated_labeled"] == 2
    assert result["per_shard_labeled_count"] == [3, 3]


def test_fi_does_not_use_test_labels_and_ties_prefer_metis(monkeypatch):
    cfg = GraphConfig(partition_strategy="fi", num_gpus=2)
    membership = balanced_random_membership(6, 2)
    monkeypatch.setattr(
        "sregnas.data.sampling.metis_partition", lambda *a: membership.copy()
    )
    rowptr, col = np.zeros(7, dtype=np.int64), np.array([], dtype=np.int64)
    train = np.array([1, 0, 1, 0, 0, 0], dtype=bool)
    val = np.array([0, 1, 0, 1, 0, 0], dtype=bool)
    labels = np.array([0, 1, 0, 1, 0, 0])
    chosen, before = select_partition(cfg, rowptr, col, labels, train, val)
    labels[4:] = [999, -100]
    after_chosen, after = select_partition(cfg, rowptr, col, labels, train, val)
    assert before == after
    assert before["selected"] == "metis"
    assert not before["test_labels_used"]
    np.testing.assert_array_equal(chosen, after_chosen)
    cfg.partition_label_scope = "train"
    labels[1] = labels[3] = -200
    _, train_only = select_partition(cfg, rowptr, col, labels, train, val)
    assert train_only["scores"]["metis"]["labeled_count"] == 2


def test_fi_rejects_empty_labeled_shard():
    with pytest.raises(AssertionError, match="empty labeled shard"):
        partition_index(
            np.zeros(5, dtype=int),
            np.array([], dtype=int),
            np.zeros(4, dtype=int),
            np.array([1, 1, 0, 0], dtype=bool),
            np.array([0, 0, 1, 1]),
            n=2,
        )


def test_partition_cache_separates_seed_and_label_scope():
    from pathlib import Path

    cfg = load_config(Path(__file__).parents[1] / "configs/arxiv/a6000_smoke_n2.yaml")
    cfg.graph.partition_strategy = "fi"
    key = _graph_cache_key(cfg)
    cfg.graph.partition_seed += 1
    assert _graph_cache_key(cfg) != key
    with pytest.raises(ValueError, match="partition protocol"):
        _validate_partition_cache(cfg, {"dataset": "ogbn-arxiv", "num_shards": 2})


@pytest.mark.parametrize("strategy", ["metis", "hierarchical", "random", "fi"])
def test_cache_validates_graph_identity_for_every_partition_strategy(strategy):
    from sregnas.config import ExperimentConfig

    cfg = ExperimentConfig(graph=GraphConfig(partition_strategy=strategy))
    manifest = dict(vars(cfg.graph), num_shards=cfg.graph.num_gpus,
                    partition_selection=dict(requested_strategy=strategy,
                                             random_seed=42, label_scope="train_val"))
    _validate_partition_cache(cfg, manifest)
    for field, wrong in [("dataset", "flickr"), ("num_shards", 8),
                         ("partition_strategy", "other"), ("hierarchical_r", 4),
                         ("fanout_a", 10), ("fanout_b", 10),
                         ("permutation_seed", 43), ("f_int", 10), ("f_ext", 5)]:
        with pytest.raises(ValueError, match="partition protocol"):
            _validate_partition_cache(cfg, dict(manifest, **{field: wrong}))


def test_uniform_graph_cache_rejects_other_dataset_and_resolves_moved_payload(tmp_path):
    from sregnas.data.graph import build_uniform_sampled_full_graph, _sha256

    payload = tmp_path / "full_graph_uniform_seed42_a5_b5.pt"
    payload.write_bytes(b"cached graph")
    manifest = dict(schema="cost_aligned_uniform_full_graph_v1", dataset="cora",
                    permutation_seed=42, fanouts=[5, 5], payload_path="/old/full.pt",
                    payload_sha256=_sha256(payload))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="dataset/protocol"):
        build_uniform_sampled_full_graph("flickr", tmp_path)
    cached = build_uniform_sampled_full_graph("Cora", tmp_path)
    assert cached["payload_path"] == str(payload.resolve())


@pytest.mark.parametrize("strategy", ["random", "fi"])
def test_graph_preparation_records_selected_partition(monkeypatch, tmp_path, strategy):
    from pathlib import Path
    from sregnas.data import graph
    from test_products_training import _dataset

    cfg = load_config(Path(__file__).parents[1] / "configs/arxiv/a6000_smoke_n2.yaml")
    cfg.graph.dataset = "cora"
    cfg.graph.partition_strategy = strategy
    references = tmp_path / "references"
    references.mkdir()
    (references / "permutation_hashes.json").write_text("{}")
    monkeypatch.setattr(graph, "REFERENCE_DIR", references)
    monkeypatch.setattr(graph, "_load_dataset", lambda name: _dataset())
    monkeypatch.setattr(
        "sregnas.data.sampling.metis_partition", lambda *a: np.arange(8) % 2
    )
    result = graph.build_graph_stage(cfg, tmp_path / "prepared")
    assert result["partition_selection"]["selected"] in {"metis", "random"}
    assert result["assertions"]["e_msg_sub_closed_form"] == "PASS"
    assert result["assertions"]["e_msg_sub_reference"] == "NOT_APPLICABLE"
    membership = np.load(tmp_path / "prepared/graph/membership.npy")
    for record in result["shards"]:
        payload = torch.load(record["path"], weights_only=False)
        nodes = payload["node_ids"].numpy()
        assert np.all(membership[nodes] == record["shard"])
        assert (
            payload["edge_index_layer1"].shape[1]
            + payload["edge_index_layer2"].shape[1]
            == record["E_msg_sub"]
        )
