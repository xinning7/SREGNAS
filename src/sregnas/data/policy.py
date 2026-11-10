

from __future__ import annotations

from dataclasses import dataclass

from sregnas.search.memory_gate import (
    STATIC_GATE_PREDICATE,
    assert_static_gate_predicate_approved,
)


CANONICAL_TWO_LAYER_ARCHITECTURES = 10_863_616


@dataclass(frozen=True)
class DatasetPolicy:
    permitted_r: frozenset[int]
    static_gate_predicate: str | None
    static_gate_purpose: str | None


_R1_ONLY = frozenset({1})
_HIERARCHICAL_VIABLE = frozenset({1, 2, 4, 8})

DATASET_POLICIES: dict[str, DatasetPolicy] = {
    "cora": DatasetPolicy(_R1_ONLY, None, None),
    "citeseer": DatasetPolicy(_R1_ONLY, None, None),
    "pubmed": DatasetPolicy(_R1_ONLY, None, None),
    "flickr": DatasetPolicy(_R1_ONLY, None, None),
    "reddit": DatasetPolicy(_R1_ONLY, None, None),
    "ogbn-arxiv": DatasetPolicy(
        _HIERARCHICAL_VIABLE,
        STATIC_GATE_PREDICATE,
        "feasibility_only",
    ),
    "ogbn-products": DatasetPolicy(_HIERARCHICAL_VIABLE, None, None),
}


def dataset_policy(name: str) -> DatasetPolicy:
    try:
        return DATASET_POLICIES[name.lower()]
    except KeyError as exc:
        raise ValueError(f"no DistNAS v2 policy for dataset: {name}") from exc


def system_static_gate_enabled(name: str) -> bool:
    pass
    predicate = dataset_policy(name).static_gate_predicate
    if predicate is None:
        return False
    assert_static_gate_predicate_approved(predicate)
    return True
