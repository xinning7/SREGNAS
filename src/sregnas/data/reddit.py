

import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from sregnas.data.products import ProductsMmapGraph, build_products_mmap_graph

from sregnas.paths import PROJECT_ROOT as ROOT, DATA_DIR


def cache_root():
    return Path(
        os.environ.get(
            "REDDIT_GRAPH_CACHE", str(DATA_DIR / "Reddit/sregnas_cache_v1")
        )
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_signature():
    return {
        name: {
            "bytes": (DATA_DIR / "Reddit/raw" / name).stat().st_size,
            "mtime_ns": (DATA_DIR / "Reddit/raw" / name).stat().st_mtime_ns,
        }
        for name in ("reddit_data.npz", "reddit_graph.npz")
    }


def prepare_reddit_graph(root=None):
    from sregnas.data import load_dataset
    from sregnas.data.graph import scout

    root = Path(root) if root is not None else cache_root()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ready = root / "ready.json"
        signature = raw_signature()
        if ready.exists():
            record = json.loads(ready.read_text())
            if record["raw_files"] != signature:
                raise RuntimeError("Reddit raw files changed since graph preparation")
            return record
        print("REDDIT_PREPARE loading public dataset", flush=True)
        dataset = load_dataset("reddit", data_dir=str(DATA_DIR))
        if (dataset["num_nodes"], dataset["num_features"], dataset["num_classes"]) != (
            232965,
            602,
            41,
        ):
            raise ValueError("Unexpected Reddit dataset variant")
        print("REDDIT_PREPARE building undirected CPU CSR", flush=True)
        build_products_mmap_graph(dataset, root / "full", dataset_name="reddit")
        del dataset
        graph = ProductsMmapGraph.open(root / "full/manifest.json")
        permutations = {}
        for layer in range(2):
            print(f"REDDIT_PREPARE frozen permutation layer {layer + 1}", flush=True)
            values = scout.shuffle_csr_segments(graph.rowptr, graph.col, 42 + layer)
            path = root / f"seed42_layer{layer + 1}.npy"
            np.save(path, values)
            permutations[path.name] = sha256(path)
            del values
        if raw_signature() != signature:
            raise RuntimeError("Reddit raw upload changed during preparation")
        record = dict(
            dataset="reddit",
            raw_files=signature,
            graph_manifest=str(root / "full/manifest.json"),
            permutation_sha256=permutations,
            permutation_seed=42,
            num_nodes=graph.num_nodes,
            num_features=graph.num_features,
            num_classes=graph.num_classes,
            directed_messages=int(graph.col.size),
            splits={
                k: int(getattr(graph, k + "_mask").sum())
                for k in ["train", "val", "test"]
            },
        )
        temporary = ready.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(ready)
        print("REDDIT_PREPARE complete " + json.dumps(record), flush=True)
        return record


def build_reddit_graph_stage(cfg, out_dir):
    from sregnas.data.graph import _per_layer_capped_edges, _tensor_rows, scout

    cfg.validate()
    root = cache_root()
    from sregnas.icde27.s0 import span
    with span("read_graph_and_csr"):
        prepared = prepare_reddit_graph(root)
        graph = ProductsMmapGraph.open(prepared["graph_manifest"])
    graph_dir = Path(out_dir) / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    permutations = []
    for name, digest in prepared["permutation_sha256"].items():
        if sha256(root / name) != digest:
            raise RuntimeError("Reddit frozen sampling permutation checksum drift")
        permutations.append(np.load(root / name, mmap_mode="r"))
    print(
        f"REDDIT_GRAPH {cfg.graph.partition_strategy} N={cfg.graph.num_gpus}",
        flush=True,
    )
    partition_selection = None
    if cfg.graph.partition_strategy in {"random", "fi"}:
        from .partition_selection import select_partition

        membership, partition_selection = select_partition(
            cfg.graph,
            graph.rowptr,
            graph.col,
            graph.labels,
            graph.train_mask,
            graph.val_mask,
        )
    else:
        membership = scout.metis_partition(graph.rowptr, graph.col, cfg.graph.num_gpus)
    assert membership.shape == (graph.num_nodes,)
    assert np.array_equal(np.unique(membership), np.arange(cfg.graph.num_gpus))

    degrees = np.diff(graph.rowptr)
    internal_degrees = np.empty(graph.num_nodes, dtype=np.int64)
    for start in range(0, graph.num_nodes, 4096):
        stop = min(start + 4096, graph.num_nodes)
        rows = np.repeat(np.arange(start, stop), degrees[start:stop])
        cols = graph.col[graph.rowptr[start] : graph.rowptr[stop]]
        inside = membership[rows] == membership[cols]
        internal_degrees[start:stop] = np.bincount(
            rows[inside] - start, minlength=stop - start
        )
    shards = []
    def materialize(part):
        nodes = np.flatnonzero(membership == part).astype(np.int64)
        edge1, msg1 = _per_layer_capped_edges(
            graph.rowptr, permutations[0], membership, part, 5
        )
        edge2, msg2 = _per_layer_capped_edges(
            graph.rowptr, permutations[1], membership, part, 5
        )
        expected = int(2 * np.minimum(internal_degrees[nodes], 5).sum())
        assert msg1 + msg2 == expected
        for edge in (edge1, edge2):
            assert not bool((edge[0] == edge[1]).any())
            assert not edge.numel() or (
                int(edge.min()) >= 0 and int(edge.max()) < len(nodes)
            )
        payload = {
            key: _tensor_rows(getattr(graph, key), nodes)
            for key in ["x", "labels", "train_mask", "val_mask", "test_mask"]
        }
        assert int(payload["train_mask"].sum()) and int(payload["val_mask"].sum())
        payload.update(
            node_ids=torch.from_numpy(nodes),
            edge_index_layer1=edge1,
            edge_index_layer2=edge2,
            msg1=msg1,
            msg2=msg2,
            expected_e_msg_sub=expected,
            num_features=graph.num_features,
            num_classes=graph.num_classes,
        )
        path = graph_dir / f"shard_{part}.pt"
        from sregnas.training.fixed_proxy import add_prepared_index
        add_prepared_index(payload, shard=part)
        torch.save(payload, path)
        record = (
            dict(
                shard=part,
                nodes=len(nodes),
                core_nodes=len(nodes),
                touched_nodes=len(nodes),
                rho_v=1.0,
                msg1=msg1,
                msg2=msg2,
                E_msg_sub=expected,
                path=str(path),
            )
        )
        print(
            f"REDDIT_GRAPH shard={part} nodes={len(nodes)} messages={expected}",
            flush=True,
        )
        return record
    from sregnas.icde27.s0 import ordered_map, span
    def measured_materialize(part):
        with span("offline_materialization", shard=part):
            return materialize(part)
    shards = ordered_map(measured_materialize, range(cfg.graph.num_gpus))
    counts = [s["E_msg_sub"] for s in shards]
    computed = dict(min=min(counts), med=float(np.median(counts)), max=max(counts))
    np.save(graph_dir / "membership.npy", membership)
    np.save(graph_dir / "macro.npy", np.zeros(graph.num_nodes, dtype=np.int64))
    manifest = dict(
        dataset="reddit",
        partition_strategy=cfg.graph.partition_strategy,
        partition_selection=partition_selection,
        hierarchical_r=1,
        num_shards=cfg.graph.num_gpus,
        permutation_seed=42,
        fanout_a=5,
        fanout_b=5,
        f_int=5,
        f_ext=0,
        shards=shards,
        computed=computed,
        full_graph_path=prepared["graph_manifest"],
        reddit_graph_manifest_path=prepared["graph_manifest"],
        permutation_sha256=prepared["permutation_sha256"],
        membership_sha256=sha256(graph_dir / "membership.npy"),
        reference_method="independent full-CSR internal-degree closed form; new dataset has no historical scout row",
        assertions=dict(
            cache_checksum_consistency="PASS",
            e_msg_sub_closed_form="PASS",
            self_loops_absent_pre_forward="PASS",
            zero_halo_rho_v_exactly_one="PASS",
        ),
    )
    (graph_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare_baseline_payload(sampled=True):
    pass
    from sregnas.data.graph import _per_layer_capped_edges, _tensor_rows

    root = cache_root()
    prepared = prepare_reddit_graph(root)
    path = root / ("baseline_sampled_5_5.pt" if sampled else "baseline_full.pt")
    with (root / "baseline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            return path
        graph = ProductsMmapGraph.open(prepared["graph_manifest"])
        nodes = np.arange(graph.num_nodes)
        payload = {
            key: _tensor_rows(getattr(graph, key), nodes)
            for key in ("x", "labels", "train_mask", "val_mask", "test_mask")
        }
        payload.update(num_features=graph.num_features, num_classes=graph.num_classes)
        if sampled:
            membership = np.zeros(graph.num_nodes, dtype=np.int64)
            edges = []
            for layer in (1, 2):
                name = f"seed42_layer{layer}.npy"
                assert sha256(root / name) == prepared["permutation_sha256"][name]
                edge, count = _per_layer_capped_edges(
                    graph.rowptr, np.load(root / name, mmap_mode="r"), membership, 0, 5
                )
                edges.append((edge, count))
            payload.update(
                edge_index_layer1=edges[0][0],
                edge_index_layer2=edges[1][0],
                msg1=edges[0][1],
                msg2=edges[1][1],
                expected_e_msg_sub=sum(e[1] for e in edges),
            )
        else:
            dst = np.repeat(nodes, np.diff(graph.rowptr))
            payload["edge_index"] = torch.stack(
                (
                    torch.from_numpy(np.array(graph.col, dtype=np.int64)),
                    torch.from_numpy(dst),
                )
            )
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
    return path
