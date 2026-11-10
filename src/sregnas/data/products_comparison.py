

import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .products import ProductsMmapGraph, build_products_mmap_graph
from sregnas.paths import DATA_DIR


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_products_graph(root=None):
    from .graph import _load_products_raw

    root = (
        Path(root)
        if root is not None
        else DATA_DIR / "ogbn_products/sregnas_comparison_v1"
    )
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "full/manifest.json"
    with (root / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not manifest.exists():
            build_products_mmap_graph(_load_products_raw(), root / "full")
        graph = ProductsMmapGraph.open(manifest)
        if (graph.num_nodes, graph.num_features, graph.num_classes) != (
            2449029,
            100,
            47,
        ):
            raise AssertionError("unexpected Products graph variant")
        if tuple(
            int(getattr(graph, s + "_mask").sum()) for s in ("train", "val", "test")
        ) != (196615, 39323, 2213091):
            raise AssertionError("Products official split changed")
    return str(manifest)


def baseline_payload(manifest, out_dir, *, sampled):
    from .graph import _per_layer_capped_edges, _tensor_rows, scout

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / ("sampled_5_5.pt" if sampled else "full.pt")
    receipt_path = path.with_suffix(".json")
    graph = ProductsMmapGraph.open(manifest)
    identity = dict(
        graph_manifest_sha256=sha256(manifest),
        sampled=sampled,
        permutation_seeds=[42, 43] if sampled else None,
    )
    with (root / "payload.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists() and receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["identity"] != identity or receipt["payload_sha256"] != sha256(
                path
            ):
                raise AssertionError("Products comparison payload changed")
            return str(path)
        nodes = np.arange(graph.num_nodes, dtype=np.int64)
        payload = {
            k: _tensor_rows(getattr(graph, k), nodes)
            for k in ("x", "labels", "train_mask", "val_mask", "test_mask")
        }
        payload.update(
            num_nodes=graph.num_nodes,
            num_features=graph.num_features,
            num_classes=graph.num_classes,
        )
        permutations = []
        if sampled:
            membership = np.zeros(graph.num_nodes, dtype=np.int64)
            expected = int(np.minimum(np.diff(graph.rowptr), 5).sum())
            for layer, seed in enumerate((42, 43), 1):
                perm = scout.shuffle_csr_segments(graph.rowptr, graph.col, seed)
                permutations.append(
                    hashlib.sha256(memoryview(perm).cast("B")).hexdigest()
                )
                edge, count = _per_layer_capped_edges(
                    graph.rowptr, perm, membership, 0, 5
                )
                if count != expected or bool((edge[0] == edge[1]).any()):
                    raise AssertionError(
                        "Products fixed [5,5] sampling contract failed"
                    )
                payload[f"edge_index_layer{layer}"] = edge
                payload[f"msg{layer}"] = count
            payload["expected_e_msg_sub"] = 2 * expected
        else:
            edge = torch.empty((2, graph.col.size), dtype=torch.int64)
            edge[0] = torch.from_numpy(np.array(graph.col, dtype=np.int64))
            edge[1] = torch.from_numpy(np.repeat(nodes, np.diff(graph.rowptr)))
            payload["edge_index"] = edge
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        receipt_path.write_text(
            json.dumps(
                dict(
                    identity=identity,
                    payload_sha256=sha256(path),
                    permutation_sha256=permutations,
                ),
                indent=2,
            )
        )
    return str(path)


def native_products_data():
    graph = ProductsMmapGraph.open(prepare_products_graph())
    nodes = np.arange(graph.num_nodes, dtype=np.int64)

    edge = torch.empty((2, graph.col.size), dtype=torch.int64)
    edge[0] = torch.from_numpy(np.repeat(nodes, np.diff(graph.rowptr)))
    edge[1] = torch.from_numpy(np.array(graph.col, dtype=np.int64))
    result = {
        k: torch.from_numpy(np.array(getattr(graph, k), copy=True))
        for k in ("x", "labels", "train_mask", "val_mask", "test_mask")
    }
    return dict(
        result,
        edge_index=edge,
        num_nodes=graph.num_nodes,
        num_features=graph.num_features,
        num_classes=graph.num_classes,
    )
