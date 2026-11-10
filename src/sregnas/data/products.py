

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch


PRODUCTS_FANOUT = (5, 5)
PRODUCTS_BATCH_SIZE = 4096
PRODUCTS_RETRAIN_PASSES = 25
PRODUCTS_INFERENCE_BATCH_SIZE = 4096
PRODUCTS_GRAPH_SCHEMA = "products_mmap_csr_v1"


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _write_npy_atomic(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    array = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=value.dtype,
        shape=value.shape,
    )
    array[...] = value
    array.flush()
    del array
    temporary.replace(path)


def edge_index_to_undirected_csr(
    edge_index: np.ndarray, num_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    pass
    import os
    if os.environ.get('SREGNAS_X18_FAST_CSR') == '1':
        from sregnas.icde27.x18.prep import undirected_csr
        return undirected_csr(edge_index, num_nodes)
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    src, dst = edge_index
    valid = (src != dst) & (src >= 0) & (dst >= 0)
    valid &= (src < num_nodes) & (dst < num_nodes)
    src, dst = src[valid], dst[valid]
    forward = src * np.int64(num_nodes) + dst
    reverse = dst * np.int64(num_nodes) + src
    packed = np.unique(np.concatenate([forward, reverse]))
    rows = packed // np.int64(num_nodes)
    col = (packed % np.int64(num_nodes)).astype(np.int32, copy=False)
    counts = np.bincount(rows, minlength=num_nodes)
    rowptr = np.empty(num_nodes + 1, dtype=np.int64)
    rowptr[0] = 0
    np.cumsum(counts, out=rowptr[1:])
    return rowptr, col


def _array_record(path: Path, value: np.ndarray) -> dict[str, Any]:
    return {
        "path": path.name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def build_products_mmap_graph(
    dataset: Mapping[str, Any], out_dir: Path, *, dataset_name: str = "ogbn-products"
) -> dict[str, Any]:
    pass
    out_dir.mkdir(parents=True, exist_ok=True)
    num_nodes = int(dataset["num_nodes"])
    x = np.ascontiguousarray(_as_numpy(dataset["x"]), dtype=np.float32)
    labels = np.ascontiguousarray(
        _as_numpy(dataset["labels"]).reshape(-1), dtype=np.int64
    )
    masks = {
        key: np.ascontiguousarray(_as_numpy(dataset[key]), dtype=np.bool_)
        for key in ("train_mask", "val_mask", "test_mask")
    }
    if x.shape[0] != num_nodes or labels.shape != (num_nodes,):
        raise ValueError("products features/labels do not match num_nodes")
    if any(mask.shape != (num_nodes,) for mask in masks.values()):
        raise ValueError("products split masks do not match num_nodes")
    rowptr, col = edge_index_to_undirected_csr(
        _as_numpy(dataset["edge_index"]), num_nodes
    )

    arrays = {
        "x": x,
        "labels": labels,
        "train_mask": masks["train_mask"],
        "val_mask": masks["val_mask"],
        "test_mask": masks["test_mask"],
        "rowptr": rowptr,
        "col": col,
    }
    records: dict[str, dict[str, Any]] = {}
    for name, value in arrays.items():
        path = out_dir / f"{name}.npy"
        _write_npy_atomic(path, value)
        records[name] = _array_record(path, value)

    graph_digest = hashlib.sha256()
    graph_digest.update(memoryview(rowptr).cast("B"))
    graph_digest.update(memoryview(col).cast("B"))
    manifest = {
        "schema": PRODUCTS_GRAPH_SCHEMA,
        "dataset": dataset_name,
        "num_nodes": num_nodes,
        "num_features": int(dataset["num_features"]),
        "num_classes": int(dataset["num_classes"]),
        "num_directed_messages": int(col.size),
        "graph_sha256": graph_digest.hexdigest(),
        "arrays": records,
        "storage": "cpu_numpy_readonly_mmap",
        "adjacency_builder": "direct_csr_no_GraphStore._build_adj",
    }
    temporary = out_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    temporary.replace(out_dir / "manifest.json")
    return manifest


@dataclass(frozen=True)
class ProductsMmapGraph:
    root: Path
    manifest: dict[str, Any]
    x: np.ndarray
    labels: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    rowptr: np.ndarray
    col: np.ndarray

    @classmethod
    def open(cls, manifest_path: str | Path) -> "ProductsMmapGraph":
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != PRODUCTS_GRAPH_SCHEMA:
            raise ValueError("unsupported products graph manifest schema")

        def load(name: str) -> np.ndarray:
            declared = Path(manifest["arrays"][name]["path"])
            path = (
                declared if declared.is_absolute() else manifest_path.parent / declared
            )
            return np.load(path, mmap_mode="r")

        graph = cls(
            root=manifest_path.parent,
            manifest=manifest,
            x=load("x"),
            labels=load("labels"),
            train_mask=load("train_mask"),
            val_mask=load("val_mask"),
            test_mask=load("test_mask"),
            rowptr=load("rowptr"),
            col=load("col"),
        )
        graph.validate()
        return graph

    @property
    def num_nodes(self) -> int:
        return int(self.manifest["num_nodes"])

    @property
    def num_features(self) -> int:
        return int(self.manifest["num_features"])

    @property
    def num_classes(self) -> int:
        return int(self.manifest["num_classes"])

    def validate(self) -> None:
        if self.x.shape != (self.num_nodes, self.num_features):
            raise ValueError("products mmap feature shape mismatch")
        if self.rowptr.shape != (self.num_nodes + 1,):
            raise ValueError("products mmap rowptr shape mismatch")
        if int(self.rowptr[-1]) != int(self.col.size):
            raise ValueError("products mmap CSR endpoint mismatch")
        if any(
            value.shape != (self.num_nodes,)
            for value in (
                self.labels,
                self.train_mask,
                self.val_mask,
                self.test_mask,
            )
        ):
            raise ValueError("products mmap labels/masks shape mismatch")


def _ordered_unique(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if values.size == 0:
        return values
    _, first = np.unique(values, return_index=True)
    return values[np.sort(first)]


def _local_indices(nodes: np.ndarray, values: np.ndarray) -> np.ndarray:
    order = np.argsort(nodes)
    sorted_nodes = nodes[order]
    positions = np.searchsorted(sorted_nodes, values)
    if np.any(positions == sorted_nodes.size) or not np.array_equal(
        sorted_nodes[positions], values
    ):
        raise AssertionError("sampled endpoint absent from compact node set")
    return order[positions].astype(np.int64, copy=False)


@dataclass(frozen=True)
class ProductsNeighborBatch:
    node_ids: np.ndarray
    seed_nodes: np.ndarray
    edge_index_layer1: np.ndarray
    edge_index_layer2: np.ndarray

    @property
    def batch_size(self) -> int:
        return int(self.seed_nodes.size)


@dataclass(frozen=True)
class ProductsKhopBatch:
    node_ids: np.ndarray
    seed_nodes: np.ndarray
    edge_indices: tuple[np.ndarray, ...]

    @property
    def batch_size(self) -> int:
        return int(self.seed_nodes.size)


@dataclass(frozen=True)
class InferenceCandidateSet:
    edge_batch: np.ndarray
    source_batch: np.ndarray
    max_incident_edges: int
    max_unique_sources: int
    analytical_combined_bound: dict[str, int]


class ReusableProductsNeighborSampler:
    pass

    def __init__(
        self,
        graph: ProductsMmapGraph,
        fanout: Sequence[int] = PRODUCTS_FANOUT,
        batch_size: int = PRODUCTS_BATCH_SIZE,
    ):
        if tuple(int(value) for value in fanout) != PRODUCTS_FANOUT:
            raise ValueError("products training fanout is locked to [5,5]")
        if int(batch_size) != PRODUCTS_BATCH_SIZE:
            raise ValueError("products training batch size is locked to 4096")
        self.graph = graph
        self.fanout = PRODUCTS_FANOUT
        self.batch_size = PRODUCTS_BATCH_SIZE
        self._candidate_sets: dict[int, InferenceCandidateSet] = {}

    def split_nodes(self, split: str) -> np.ndarray:
        try:
            mask = {
                "train": self.graph.train_mask,
                "val": self.graph.val_mask,
                "test": self.graph.test_mask,
            }[split]
        except KeyError as exc:
            raise ValueError(f"unknown products split: {split}") from exc
        return np.flatnonzero(mask).astype(np.int64, copy=False)

    def seed_batches(
        self,
        split: str,
        *,
        seed: int,
        shuffle: bool,
    ) -> Iterator[np.ndarray]:
        nodes = self.split_nodes(split).copy()
        if shuffle:
            np.random.default_rng(seed).shuffle(nodes)
        for start in range(0, nodes.size, self.batch_size):

            yield nodes[start : start + self.batch_size]

    def seed_batches_with_size(
        self,
        split: str,
        *,
        seed: int,
        shuffle: bool,
        batch_size: int,
    ) -> Iterator[np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        nodes = self.split_nodes(split).copy()
        if shuffle:
            np.random.default_rng(seed).shuffle(nodes)
        for start in range(0, nodes.size, batch_size):
            yield nodes[start : start + batch_size]

    def inference_batches(
        self, batch_size: int = PRODUCTS_INFERENCE_BATCH_SIZE
    ) -> Iterator[np.ndarray]:
        if batch_size <= 0:
            raise ValueError("inference batch size must be positive")
        nodes = np.arange(self.graph.num_nodes, dtype=np.int64)
        for start in range(0, nodes.size, batch_size):
            yield nodes[start : start + batch_size]

    def _neighbors(
        self,
        targets: np.ndarray,
        fanout: int | None,
        rng: np.random.Generator | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        rowptr, col = self.graph.rowptr, self.graph.col
        degrees = rowptr[targets + 1] - rowptr[targets]
        counts = degrees if fanout is None else np.minimum(degrees, fanout)
        total = int(counts.sum())
        sources = np.empty(total, dtype=np.int64)
        destinations = np.repeat(targets, counts.astype(np.int64))
        cursor = 0
        for target, degree, count in zip(
            targets.tolist(), degrees.tolist(), counts.tolist()
        ):
            start, stop = int(rowptr[target]), int(rowptr[target + 1])
            neighbors = col[start:stop]
            if count == degree:
                chosen = neighbors
            else:
                if rng is None:
                    raise AssertionError("finite fanout requires a generator")
                chosen = neighbors[
                    rng.choice(int(degree), size=int(count), replace=False)
                ]
            sources[cursor : cursor + count] = chosen
            cursor += count
        return sources, destinations

    def sample(self, seed_nodes: np.ndarray, *, seed: int) -> ProductsNeighborBatch:
        generic = self.sample_with_fanouts(seed_nodes, fanouts=self.fanout, seed=seed)
        return ProductsNeighborBatch(
            node_ids=generic.node_ids,
            seed_nodes=generic.seed_nodes,
            edge_index_layer1=generic.edge_indices[0],
            edge_index_layer2=generic.edge_indices[1],
        )

    def sample_with_fanouts(
        self,
        seed_nodes: np.ndarray,
        *,
        fanouts: Sequence[int],
        seed: int,
    ) -> ProductsKhopBatch:
        pass
        fanouts = tuple(int(value) for value in fanouts)
        if not fanouts or any(value <= 0 for value in fanouts):
            raise ValueError("fanouts must be a non-empty sequence of positives")
        seed_nodes = _ordered_unique(np.asarray(seed_nodes, dtype=np.int64))
        rng = np.random.default_rng(seed)
        targets = seed_nodes
        output_to_input_edges: list[tuple[np.ndarray, np.ndarray]] = []
        for fanout in reversed(fanouts):
            sources, destinations = self._neighbors(targets, fanout, rng)
            output_to_input_edges.append((sources, destinations))
            targets = _ordered_unique(np.concatenate([targets, sources]))
        nodes = targets
        edges = tuple(
            np.stack(
                [
                    _local_indices(nodes, sources),
                    _local_indices(nodes, destinations),
                ]
            )
            for sources, destinations in reversed(output_to_input_edges)
        )
        if not np.array_equal(nodes[: seed_nodes.size], seed_nodes):
            raise AssertionError("seed nodes must be the leading compact nodes")
        return ProductsKhopBatch(
            node_ids=nodes,
            seed_nodes=seed_nodes,
            edge_indices=edges,
        )

    def full_neighbor_edges(
        self, target_nodes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        targets = np.asarray(target_nodes, dtype=np.int64)
        sources, destinations = self._neighbors(targets, None, None)
        nodes = _ordered_unique(np.concatenate([targets, sources]))
        edge_index = np.stack(
            [_local_indices(nodes, sources), _local_indices(nodes, destinations)]
        )
        return nodes, edge_index

    def inference_candidate_set(
        self, batch_size: int = PRODUCTS_INFERENCE_BATCH_SIZE
    ) -> InferenceCandidateSet:
        pass
        if batch_size <= 0 or batch_size > self.graph.num_nodes:
            raise ValueError("invalid inference candidate batch size")
        cached = self._candidate_sets.get(int(batch_size))
        if cached is not None:
            return cached
        degrees = np.diff(self.graph.rowptr).astype(np.int64, copy=False)
        node_ids = np.arange(self.graph.num_nodes, dtype=np.int64)
        order = np.lexsort((node_ids, -degrees))
        edge_batch = order[:batch_size].astype(np.int64, copy=False)
        max_incident_edges = int(degrees[edge_batch].sum())

        source_batch: np.ndarray | None = None
        max_unique_sources = -1
        source_batch_edges = 0
        for batch in self.inference_batches(batch_size):
            sources, _ = self._neighbors(batch, None, None)
            unique_sources = int(np.unique(np.concatenate([batch, sources])).size)
            if unique_sources > max_unique_sources:
                source_batch = batch.copy()
                max_unique_sources = unique_sources
                source_batch_edges = int(sources.size)
        assert source_batch is not None
        result = InferenceCandidateSet(
            edge_batch=edge_batch,
            source_batch=source_batch,
            max_incident_edges=max_incident_edges,
            max_unique_sources=max_unique_sources,
            analytical_combined_bound={
                "target_nodes": int(batch_size),
                "incident_edges": max_incident_edges,
                "unique_source_nodes": max_unique_sources,
                "source_witness_edges": source_batch_edges,
            },
        )
        self._candidate_sets[int(batch_size)] = result
        return result
