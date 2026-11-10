

from __future__ import annotations

from collections.abc import Callable
from typing import Any


EVEN_SPLIT = "even-split"
GLOBAL_BY_LOCAL_SCORE = "global-by-local-score"
GLOBAL_BY_FULLGRAPH_SCORE = "global-by-fullgraph-score"
ALLOCATION_STRATEGIES = (
    EVEN_SPLIT,
    GLOBAL_BY_LOCAL_SCORE,
    GLOBAL_BY_FULLGRAPH_SCORE,
)


def _eligible(row: dict[str, Any]) -> bool:
    return row["status"] == "OK" and row.get("value") is not None


def rank_per_shard(
    rows: list[dict[str, Any]], num_shards: int
) -> dict[int, list[dict]]:
    ranked: dict[int, list[dict]] = {}
    for shard in range(num_shards):
        local = [r for r in rows if int(r["shard"]) == shard and _eligible(r)]
        ranked[shard] = sorted(
            local,
            key=lambda row: (-float(row["value"]), row["fingerprint"]),
        )
    return ranked


def plan1_nominate(
    rows: list[dict[str, Any]], num_shards: int, k: int
) -> tuple[list[dict], dict[int, list[dict]]]:
    ranked = rank_per_shard(rows, num_shards)
    selected: list[dict] = []
    global_seen: set[str] = set()
    for shard in range(num_shards):
        for row in ranked[shard]:
            if row["fingerprint"] in global_seen:
                continue
            selected.append(row)
            global_seen.add(row["fingerprint"])
            if sum(int(x["shard"]) == shard for x in selected) == k:
                break
        if sum(int(x["shard"]) == shard for x in selected) != k:
            raise AssertionError(f"shard {shard} cannot provide k={k} unique nominees")
    if len(selected) != num_shards * k:
        raise AssertionError("Plan1 did not produce X=N*k unique nominees")
    return selected, ranked


def _unique_global_top(
    rows: list[dict[str, Any]], *, x: int, score_key: str
) -> list[dict[str, Any]]:
    eligible = [
        row for row in rows if _eligible(row) and row.get(score_key) is not None
    ]
    ordered = sorted(
        eligible,
        key=lambda row: (-float(row[score_key]), row["fingerprint"]),
    )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in ordered:
        fingerprint = str(row["fingerprint"])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        selected.append(row)
        if len(selected) == x:
            break
    if len(selected) != x:
        raise AssertionError(
            f"allocation cannot provide X={x} unique rows by {score_key}"
        )
    return selected


def allocate_nominees(
    candidates: list[dict[str, Any]],
    num_shards: int,
    k: int,
    strategy: str = EVEN_SPLIT,
    *,
    fullgraph_score_key: str = "fullgraph_score",
) -> list[dict[str, Any]]:
    pass
    if strategy not in ALLOCATION_STRATEGIES:
        raise ValueError(f"unknown allocation strategy: {strategy}")
    if strategy == EVEN_SPLIT:
        selected, _ranked = plan1_nominate(candidates, num_shards, k)
        return selected
    x = int(num_shards) * int(k)
    if strategy == GLOBAL_BY_LOCAL_SCORE:
        return _unique_global_top(candidates, x=x, score_key="value")
    return _unique_global_top(candidates, x=x, score_key=fullgraph_score_key)


def plan2_nominate(
    rows: list[dict[str, Any]], x: int, scorer_metadata: dict[str, Any]
) -> list[dict]:
    scorer = str(scorer_metadata.get("scorer", ""))
    comparable = bool(scorer_metadata.get("cross_shard_comparable", False))
    if scorer.startswith("pv@") or not comparable:
        raise AssertionError(f"Plan2 refuses non-comparable scorer {scorer!r}")
    eligible = [r for r in rows if _eligible(r)]
    return sorted(eligible, key=lambda row: (-float(row["value"]), row["fingerprint"]))[
        :x
    ]


def probe_and_refill(
    nominees: list[dict],
    ranked: dict[int, list[dict]],
    probe: Callable[[dict], bool],
) -> tuple[list[dict], list[dict]]:
    selected = list(nominees)
    audit: list[dict] = []
    selected_fps = {row["fingerprint"] for row in selected}
    if len(selected_fps) != len(selected):
        raise AssertionError("probe nominees must have distinct fingerprints")
    failed_fps: set[str] = set()
    index = 0
    while index < len(selected):
        row = selected[index]
        ok = probe(row)
        audit.append(
            {
                "shard": row["shard"],
                "fingerprint": row["fingerprint"],
                "probe": "PASS" if ok else "OOM",
            }
        )
        if ok:
            index += 1
            continue
        failed_fps.add(row["fingerprint"])
        shard = int(row["shard"])
        replacement = None
        for candidate in ranked[shard]:
            if candidate["fingerprint"] not in selected_fps | failed_fps:
                replacement = candidate
                break
        if replacement is None:
            raise AssertionError(f"shard {shard} exhausted during probe refill")
        selected_fps.remove(row["fingerprint"])
        selected_fps.add(replacement["fingerprint"])
        selected[index] = replacement
        audit[-1]["refill_fingerprint"] = replacement["fingerprint"]
    return selected, audit
