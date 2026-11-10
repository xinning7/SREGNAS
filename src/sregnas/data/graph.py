

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sregnas.config import ExperimentConfig
from sregnas.data.products import ProductsMmapGraph, build_products_mmap_graph


from sregnas.paths import PROJECT_ROOT as ROOT
from . import sampling as scout
from .partition import two_level_membership
from sregnas.paths import DATA_DIR, REFERENCE_DIR


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_dataset(name: str) -> dict[str, Any]:
    from sregnas.data.dataset_manager import load_dataset

    return load_dataset(name, data_dir=str(DATA_DIR))


def _dataset_artifact_name(name: str) -> str:
    from sregnas.data.dataset_manager import get_dataset_artifact_name

    return get_dataset_artifact_name(name)


def _load_products_raw() -> dict[str, Any]:
    pass
    from ogb.nodeproppred import NodePropPredDataset

    dataset = NodePropPredDataset(name="ogbn-products", root=str(DATA_DIR))
    graph, labels = dataset[0]
    num_nodes = int(graph["num_nodes"])
    split = dataset.get_idx_split()
    masks = {}
    for target, source in (
        ("train_mask", "train"),
        ("val_mask", "valid"),
        ("test_mask", "test"),
    ):
        mask = np.zeros(num_nodes, dtype=np.bool_)
        mask[np.asarray(split[source], dtype=np.int64)] = True
        masks[target] = mask
    node_labels = np.asarray(labels).reshape(-1).astype(np.int64, copy=False)
    return {
        "x": np.asarray(graph["node_feat"], dtype=np.float32),
        "edge_index": np.asarray(graph["edge_index"], dtype=np.int64),
        "labels": node_labels,
        **masks,
        "num_features": int(graph["node_feat"].shape[1]),
        "num_classes": int(node_labels.max()) + 1,
        "num_nodes": num_nodes,
    }


def _tensor_rows(value: Any, node_ids: np.ndarray) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value[node_ids].contiguous()
    return torch.from_numpy(np.array(value[node_ids], copy=True)).contiguous()


def _reference_row(cfg: ExperimentConfig) -> tuple[dict[str, str], str]:
    if cfg.graph.partition_strategy == "metis":
        path = REFERENCE_DIR / "scout.tsv"
        rows = csv.DictReader(path.open(), delimiter="\t")
        for row in rows:
            if (
                row["dataset"].lower() == cfg.graph.dataset.lower()
                and int(row["S"]) == cfg.graph.num_gpus
                and row["mode"] == "locality"
                and int(row["f_int"]) == cfg.graph.f_int
                and int(row["f_ext"]) == cfg.graph.f_ext
            ):
                return row, str(path)
    else:
        path = REFERENCE_DIR / "hier_scout.tsv"
        rows = csv.DictReader(path.open(), delimiter="\t")
        for row in rows:
            if (
                row["dataset"].lower() == cfg.graph.dataset.lower()
                and int(row["R"]) == cfg.graph.hierarchical_r
                and int(row["N"]) == cfg.graph.num_gpus
                and int(row["a"]) == cfg.graph.fanout_a
                and int(row["b"]) == cfg.graph.fanout_b
            ):
                return row, str(path)
    raise AssertionError("no scout reference row for active graph configuration")


def _per_layer_capped_edges(
    rowptr: np.ndarray,
    col_perm: np.ndarray,
    membership: np.ndarray,
    part: int,
    cap: int,
) -> tuple[torch.Tensor, int]:
    source_ids = np.flatnonzero(membership == part).astype(np.int64)
    deg = (rowptr[1:] - rowptr[:-1]).astype(np.int64)
    caps = deg[source_ids]
    pos = scout._ragged_positions(rowptr[source_ids], caps)
    if len(pos) == 0:
        return torch.empty((2, 0), dtype=torch.long), 0
    node_of = np.repeat(np.arange(len(source_ids), dtype=np.int64), caps)
    nbr = col_perm[pos].astype(np.int64)
    inside = membership[nbr] == part
    inside_rank = scout._segmented_exclusive_count(node_of, inside)
    keep = inside & (inside_rank < np.int64(cap))
    src = source_ids[node_of[keep]]
    nbr = nbr[keep]

    src_local = np.searchsorted(source_ids, src)
    nbr_local = np.searchsorted(source_ids, nbr)
    if not np.array_equal(source_ids[nbr_local], nbr):
        raise AssertionError("capped edge escaped its shard")

    edge = np.stack([nbr_local, src_local]).astype(np.int64)
    return torch.from_numpy(edge).long().contiguous(), int(keep.sum())


def build_uniform_sampled_full_graph(
    dataset: str,
    out_dir: Path,
    *,
    permutation_seed: int = 42,
    fanouts: tuple[int, int] = (5, 5),
) -> dict[str, Any]:
    pass
    if permutation_seed != 42 or fanouts != (5, 5):
        raise ValueError("cost-aligned baseline is locked to seed-42 [5,5]")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload_path = out_dir / "full_graph_uniform_seed42_a5_b5.pt"
    manifest_path = out_dir / "manifest.json"
    if payload_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("schema") != "cost_aligned_uniform_full_graph_v1"
            or manifest.get("dataset") != dataset.lower()
            or manifest.get("permutation_seed") != permutation_seed
            or manifest.get("fanouts") != list(fanouts)
        ):
            raise ValueError(
                "sampled full-graph cache does not match the requested dataset/protocol"
            )
        if manifest.get("payload_sha256") != _sha256(payload_path):
            raise AssertionError("sampled full-graph payload checksum drift")
        manifest["payload_path"] = str(payload_path.resolve())
        return manifest

    ds = _load_dataset(dataset)
    num_nodes = int(ds["num_nodes"])
    rowptr, col, degree = scout.build_csr(ds["edge_index"].cpu().numpy(), num_nodes)
    dataset_file = _dataset_artifact_name(dataset)
    perm_dir = out_dir / "sampling_perms"
    perm_dir.mkdir(exist_ok=True)
    hashes = json.loads((REFERENCE_DIR / "permutation_hashes.json").read_text())
    stored_permutations = scout.build_layer_perms(
        rowptr, col, permutation_seed, num_layers=2
    )
    permutation_paths = []
    for layer, permutation in enumerate(stored_permutations, 1):
        path = perm_dir / f"{dataset_file}_seed42_layer{layer}.npy"
        np.save(path, permutation)
        expected_hash = hashes.get(path.name)
        if expected_hash and _sha256(path) != expected_hash:
            raise AssertionError(f"full-graph permutation layer {layer} drifted")
        permutation_paths.append(path)

    membership = np.zeros(num_nodes, dtype=np.int64)
    edge1, msg1 = _per_layer_capped_edges(
        rowptr, stored_permutations[0], membership, 0, fanouts[0]
    )
    edge2, msg2 = _per_layer_capped_edges(
        rowptr, stored_permutations[1], membership, 0, fanouts[1]
    )
    expected1 = int(np.minimum(degree, fanouts[0]).sum())
    expected2 = int(np.minimum(degree, fanouts[1]).sum())
    if (msg1, msg2) != (expected1, expected2):
        raise AssertionError(
            "sampled full-graph message counts do not match the degree cap"
        )
    if torch.equal(edge1, edge2):
        raise AssertionError("the two frozen per-layer edge tensors collapsed")
    if bool((edge1[0] == edge1[1]).any()) or bool((edge2[0] == edge2[1]).any()):
        raise AssertionError("sampled full graph contains pre-forward self loops")

    torch.save(
        {
            "x": ds["x"].cpu().contiguous(),
            "edge_index_layer1": edge1,
            "edge_index_layer2": edge2,
            "labels": ds["labels"].cpu().contiguous(),
            "train_mask": ds["train_mask"].cpu().contiguous(),
            "val_mask": ds["val_mask"].cpu().contiguous(),
            "test_mask": ds["test_mask"].cpu().contiguous(),
            "msg1": msg1,
            "msg2": msg2,
            "expected_e_msg_sub": msg1 + msg2,
            "num_features": int(ds["num_features"]),
            "num_classes": int(ds["num_classes"]),
        },
        payload_path,
    )
    manifest = {
        "schema": "cost_aligned_uniform_full_graph_v1",
        "dataset": dataset.lower(),
        "num_nodes": num_nodes,
        "permutation_seed": permutation_seed,
        "fanouts": list(fanouts),
        "msg1": msg1,
        "msg2": msg2,
        "combined_messages": msg1 + msg2,
        "payload_path": str(payload_path.resolve()),
        "payload_sha256": _sha256(payload_path),
        "permutation_sha256": {path.name: _sha256(path) for path in permutation_paths},
        "build_count": 1,
        "self_loops": 0,
        "distinct_per_layer_edge_tensors": True,
        "self_loop_policy": "added_in_forward_only",
        "assertions": {
            "stored_permutations_bit_consistent": "PASS",
            "closed_form_message_counts": "PASS",
            "two_distinct_layer_edges": "PASS",
            "self_loops_absent_pre_forward": "PASS",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def build_graph_stage(cfg: ExperimentConfig, out_dir: Path) -> dict[str, Any]:
    from sregnas.icde27.x18.cold_prep import enabled as memory_prep_enabled
    if memory_prep_enabled():
        from sregnas.icde27.x18.cold_prep import build
        return build(cfg,out_dir)
    cfg.validate()
    if cfg.graph.dataset == "reddit":
        from sregnas.data.reddit import build_reddit_graph_stage

        return build_reddit_graph_stage(cfg, out_dir)
    graph_dir = out_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    from sregnas.icde27.s0 import span
    products_manifest_path: Path | None = None
    if cfg.graph.dataset == "ogbn-products":
        with span("read_graph"):
            raw = _load_products_raw()
        products_dir = graph_dir / "products_mmap"
        with span("csr_materialization"):
            build_products_mmap_graph(raw, products_dir)
        products_manifest_path = products_dir / "manifest.json"
        products_graph = ProductsMmapGraph.open(products_manifest_path)
        ds = {
            "x": products_graph.x,
            "labels": products_graph.labels,
            "train_mask": products_graph.train_mask,
            "val_mask": products_graph.val_mask,
            "test_mask": products_graph.test_mask,
            "num_features": products_graph.num_features,
            "num_classes": products_graph.num_classes,
            "num_nodes": products_graph.num_nodes,
        }
        rowptr, col = products_graph.rowptr, products_graph.col
    else:
        with span("read_graph"):
            ds = _load_dataset(cfg.graph.dataset)
        edge_np = ds["edge_index"].cpu().numpy()
        with span("csr_materialization"):
            rowptr, col, _ = scout.build_csr(edge_np, int(ds["num_nodes"]))
    rebuilt_perms = scout.build_layer_perms(
        rowptr, col, cfg.graph.permutation_seed, num_layers=2
    )
    dataset_file = _dataset_artifact_name(cfg.graph.dataset)
    perm_dir = graph_dir / "sampling_perms"
    perm_dir.mkdir(exist_ok=True)
    checked_paths = []
    hashes = json.loads((REFERENCE_DIR / "permutation_hashes.json").read_text())
    for layer, permutation in enumerate(rebuilt_perms, 1):
        path = perm_dir / f"{dataset_file}_seed42_layer{layer}.npy"
        np.save(path, permutation)
        expected_hash = hashes.get(path.name)
        if expected_hash and _sha256(path) != expected_hash:
            raise AssertionError(
                f"seed-42 permutation layer {layer} differs from reference"
            )
        checked_paths.append(path)

    partition_selection = None
    if cfg.graph.partition_strategy in {"random", "fi"}:
        from .partition_selection import select_partition

        membership, partition_selection = select_partition(
            cfg.graph, rowptr, col, ds["labels"], ds["train_mask"], ds["val_mask"]
        )
        macro = np.zeros(len(membership), dtype=np.int64)
    elif cfg.graph.partition_strategy == "metis":
        membership = scout.metis_partition(rowptr, col, cfg.graph.num_gpus)
        macro = np.zeros(len(membership), dtype=np.int64)
    else:
        membership, macro = two_level_membership(
            rowptr, col, cfg.graph.hierarchical_r, cfg.graph.num_gpus
        )

    row = np.repeat(
        np.arange(len(rowptr) - 1, dtype=np.int64), np.diff(rowptr).astype(np.int64)
    )
    internal = membership[row] == membership[col.astype(np.int64)]
    deg_int = np.bincount(row[internal], minlength=len(membership)).astype(np.int64)
    expected_per_shard = []
    shard_records = []
    def materialize(part):
        node_ids = np.flatnonzero(membership == part).astype(np.int64)
        edge1, msg1 = _per_layer_capped_edges(
            rowptr, rebuilt_perms[0], membership, part, cfg.graph.fanout_a
        )
        edge2, msg2 = _per_layer_capped_edges(
            rowptr, rebuilt_perms[1], membership, part, cfg.graph.fanout_b
        )
        expected = int(
            (
                np.minimum(deg_int[node_ids], cfg.graph.fanout_a)
                + np.minimum(deg_int[node_ids], cfg.graph.fanout_b)
            ).sum()
        )
        if msg1 + msg2 != expected:
            raise AssertionError(
                f"shard {part}: capped messages {msg1}+{msg2} != closed form {expected}"
            )
        if bool((edge1[0] == edge1[1]).any()) or bool((edge2[0] == edge2[1]).any()):
            raise AssertionError("graph-stage edge tensor contains a self-loop")
        payload = {
            "x": _tensor_rows(ds["x"], node_ids),
            "labels": _tensor_rows(ds["labels"], node_ids),
            "train_mask": _tensor_rows(ds["train_mask"], node_ids),
            "val_mask": _tensor_rows(ds["val_mask"], node_ids),
            "test_mask": _tensor_rows(ds["test_mask"], node_ids),
            "node_ids": torch.from_numpy(node_ids),
            "edge_index_layer1": edge1,
            "edge_index_layer2": edge2,
            "msg1": msg1,
            "msg2": msg2,
            "expected_e_msg_sub": expected,
            "num_features": int(ds["num_features"]),
            "num_classes": int(ds["num_classes"]),
        }
        shard_path = graph_dir / f"shard_{part}.pt"
        from sregnas.training.fixed_proxy import add_prepared_index
        add_prepared_index(payload, shard=part)
        torch.save(payload, shard_path)
        return (
            {
                "shard": part,
                "nodes": len(node_ids),
                "core_nodes": len(node_ids),
                "touched_nodes": len(node_ids),
                "rho_v": 1.0,
                "msg1": msg1,
                "msg2": msg2,
                "E_msg_sub": expected,
                "path": str(shard_path),
                **({"train_nodes": int(payload["train_mask"].sum()),
                    "validation_nodes": int(payload["val_mask"].sum()),
                    "train_coverage": float(payload["train_mask"].sum()) / int(np.asarray(ds["train_mask"]).sum()),
                    "validation_coverage": float(payload["val_mask"].sum()) / int(np.asarray(ds["val_mask"]).sum())}
                   if __import__("os").environ.get("SREGNAS_QUALITY_ONLY") == "1" else {}),
            }
        )

    from sregnas.icde27.s0 import ordered_map, span
    def measured_materialize(part):
        with span("offline_materialization", shard=part):
            return materialize(part)
    shard_records = ordered_map(measured_materialize, range(cfg.graph.num_gpus))
    expected_per_shard = [r["E_msg_sub"] for r in shard_records]

    computed = {
        "min": min(expected_per_shard),
        "med": float(np.median(expected_per_shard)),
        "max": max(expected_per_shard),
    }
    reference = ref_path = None
    if partition_selection is None:
        ref, ref_path = _reference_row(cfg)
        reference = {
            "min": int(ref["E_msg_sub_min"]),
            "med": float(ref["E_msg_sub_med"]),
            "max": int(ref["E_msg_sub_max"]),
        }
        if computed != reference:
            raise AssertionError(
                f"rebuilt E_msg_sub aggregates {computed} != scout reference {reference}"
            )

    if cfg.graph.dataset == "ogbn-products":
        assert products_manifest_path is not None
        full_path = products_manifest_path
    else:
        full_edge = (
            torch.from_numpy(np.stack([col.astype(np.int64), row]).astype(np.int64))
            .long()
            .contiguous()
        )
        if bool((full_edge[0] == full_edge[1]).any()):
            raise AssertionError("full graph contains pre-added self loops")
        full_path = graph_dir / "full_graph.pt"
        torch.save(
            {
                "x": ds["x"].cpu().contiguous(),
                "edge_index": full_edge,
                "labels": ds["labels"].cpu().contiguous(),
                "train_mask": ds["train_mask"].cpu().contiguous(),
                "val_mask": ds["val_mask"].cpu().contiguous(),
                "test_mask": ds["test_mask"].cpu().contiguous(),
                "num_features": int(ds["num_features"]),
                "num_classes": int(ds["num_classes"]),
            },
            full_path,
        )
    manifest = {
        "dataset": cfg.graph.dataset,
        "partition_strategy": cfg.graph.partition_strategy,
        "partition_selection": partition_selection,
        "hierarchical_r": cfg.graph.hierarchical_r,
        "num_shards": cfg.graph.num_gpus,
        "permutation_seed": 42,
        "fanout_a": cfg.graph.fanout_a,
        "fanout_b": cfg.graph.fanout_b,
        "f_int": 5,
        "f_ext": 0,
        "reference_path": ref_path,
        "reference": reference,
        "computed": computed,
        "permutation_sha256": {
            checked_paths[0].name: _sha256(checked_paths[0]),
            checked_paths[1].name: _sha256(checked_paths[1]),
        },
        "shards": shard_records,
        "full_graph_path": str(full_path),
        "products_graph_manifest_path": (
            str(products_manifest_path) if products_manifest_path is not None else None
        ),
        "assertions": {
            "cache_bit_consistency": "PASS",
            "e_msg_sub_reference": (
                "PASS" if reference is not None else "NOT_APPLICABLE"
            ),
            "e_msg_sub_closed_form": "PASS",
            "self_loops_absent_pre_forward": "PASS",
            "zero_halo_rho_v_exactly_one": "PASS",
        },
    }
    (graph_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    np.save(graph_dir / "membership.npy", membership)
    np.save(graph_dir / "macro.npy", macro)
    return manifest
