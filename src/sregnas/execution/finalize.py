

from __future__ import annotations


import json


import os


import time


from pathlib import Path

from typing import Any


from sregnas.config import ExperimentConfig

from sregnas.core.contracts import CellResult, EvidenceKind, Status, TaskKey


from sregnas.execution.pool import PersistentWorkerPool


from sregnas.selection.nomination import (
    EVEN_SPLIT,
    allocate_nominees,
    probe_and_refill,
    rank_per_shard,
)

from sregnas.execution.queue import TaskQueue

from sregnas.evaluation.checkpoints import POST_SELECTION_TEST_MODE


from .tasks import (
    _assert_cuda_phase_boundary,
    _cell_spec,
    _execute_batches,
    _execute_task,
    _task_rows,
)


def _evaluate_selected_winner_test(
    *,
    cfg: ExperimentConfig,
    profile_dir: Path,
    db_path: Path,
    graph_manifest: dict[str, Any],
    side: str,
    winner: CellResult,
    worker_pool: PersistentWorkerPool | None,
    phase: str,
) -> CellResult:
    pass
    checkpoint_path = winner.metrics.get("checkpoint_path")
    if not checkpoint_path:
        raise AssertionError(f"{phase}: validation winner lacks a retrain checkpoint")
    physical_gpus = (
        cfg.runtime.system_gpus if side == "system" else cfg.runtime.baseline_gpus
    )
    physical_gpu = physical_gpus[winner.shard % len(physical_gpus)]
    key = TaskKey(phase, 0, winner.shard, winner.fingerprint)
    spec = _cell_spec(
        key=key,
        mode=POST_SELECTION_TEST_MODE,
        shard=winner.shard,
        architecture=winner.architecture,
        graph_path=graph_manifest["full_graph_path"],
        cfg=cfg,
        physical_gpu=physical_gpu,
    )
    spec["checkpoint_path"] = str(checkpoint_path)
    result = _execute_task(
        db_path,
        profile_dir,
        key,
        winner.shard,
        winner.architecture,
        spec,
        worker_pool=worker_pool,
    )
    if result.status != Status.OK or "test_acc" not in result.metrics:
        raise AssertionError(
            f"{phase}: explicit winner test failed: {result.status} {result.error}"
        )
    return result


def _coverage_targets(cfg: ExperimentConfig) -> tuple[int, ...]:
    if (
        cfg.full_budget
        and cfg.full_budget.get("protocol") == "products_cost_aligned_sampled_v1"
    ):
        if cfg.graph.num_gpus != 8 or cfg.fixed_x != 24 or cfg.selection.k != 3:
            raise AssertionError("products cost-aligned coverage contract drifted")
        return (8, 16, 24)
    return (int(cfg.graph.num_gpus * cfg.selection.k),)


def _coverage_subsets(
    feasible: list[dict[str, Any]], cfg: ExperimentConfig
) -> dict[int, list[dict[str, Any]]]:
    targets = _coverage_targets(cfg)
    if len(targets) == 1:
        return {targets[0]: list(feasible)}
    by_shard = {
        shard: [row for row in feasible if int(row["shard"]) == shard]
        for shard in range(cfg.graph.num_gpus)
    }
    if any(len(rows) != cfg.selection.k for rows in by_shard.values()):
        raise AssertionError("products Plan1 feasible set is not k=3 per shard")
    return {
        x: [
            row
            for shard in range(cfg.graph.num_gpus)
            for row in by_shard[shard][: x // cfg.graph.num_gpus]
        ]
        for x in targets
    }


def _post_search_side(
    *,
    cfg: ExperimentConfig,
    profile_dir: Path,
    db_path: Path,
    graph_manifest: dict[str, Any],
    side: str,
    search_info: dict[str, Any],
    worker_pool: PersistentWorkerPool | None = None,
    close_worker_pool: bool = False,
) -> dict[str, Any]:
    probe_phase = f"{cfg.name}:{side}:{cfg.search.generator}:probe"
    retrain_phase = f"{cfg.name}:{side}:{cfg.search.generator}:retrain"
    test_phase = f"{cfg.name}:{side}:{cfg.search.generator}:post_selection_test"
    coverage_targets = _coverage_targets(cfg)
    primary_x = max(coverage_targets)
    queue = TaskQueue(db_path)
    probe_event = queue.conn.execute(
        "SELECT payload_json FROM phase_events "
        "WHERE phase=? AND event='PROBE_COMPLETE' ORDER BY id DESC LIMIT 1",
        (probe_phase,),
    ).fetchone()
    retrain_event = queue.conn.execute(
        "SELECT payload_json FROM phase_events "
        "WHERE phase=? AND event='RETRAIN_COMPLETE' ORDER BY id DESC LIMIT 1",
        (retrain_phase,),
    ).fetchone()
    test_event = queue.conn.execute(
        "SELECT payload_json FROM phase_events "
        "WHERE phase=? AND event='TEST_COMPLETE' ORDER BY id DESC LIMIT 1",
        (test_phase,),
    ).fetchone()
    post_incomplete = int(
        queue.conn.execute(
            "SELECT COUNT(*) n FROM tasks WHERE phase IN (?,?,?) "
            "AND status IN ('PENDING','RUNNING')",
            (probe_phase, retrain_phase, test_phase),
        ).fetchone()["n"]
    )
    if (
        probe_event is not None
        and retrain_event is not None
        and test_event is not None
        and not post_incomplete
    ):
        probe_payload = json.loads(probe_event["payload_json"])
        retrain_payload = json.loads(retrain_event["payload_json"])
        test_payload = json.loads(test_event["payload_json"])
        stored_coverages = test_payload.get("coverages")
        if stored_coverages is None:
            stored_coverages = {
                str(primary_x): {
                    "winner": retrain_payload["winner"],
                    "val_acc": float(retrain_payload["val_acc"]),
                    "test_acc": float(test_payload["test_acc"]),
                }
            }
        result = {
            "probe_wall_s": float(probe_payload["wall_s"]),
            "retrain_wall_s": float(retrain_payload["wall_s"]),
            "test_wall_s": float(test_payload["wall_s"]),
            "winner": retrain_payload["winner"],
            "val_acc": float(retrain_payload["val_acc"]),
            "test_acc": float(test_payload["test_acc"]),
            "coverages": stored_coverages,
            "test_isolation": "POST_SELECTION_ENFORCED",
            "worker_audit": (worker_pool.audit() if worker_pool is not None else None),
        }
        queue.add_event(
            f"{cfg.name}:{side}:{cfg.search.generator}",
            "POST_SEARCH_RESULT_REUSED_AFTER_RESUME",
            EvidenceKind.RAW,
            result,
        )
        queue.close()
        return result
    queue.close()
    rows = _task_rows(db_path, search_info["phase"])
    ranked = rank_per_shard(rows, cfg.graph.num_gpus)
    nominees = allocate_nominees(
        rows,
        cfg.graph.num_gpus,
        cfg.selection.k,
        strategy=EVEN_SPLIT,
    )
    physical_gpus = (
        cfg.runtime.system_gpus if side == "system" else cfg.runtime.baseline_gpus
    )
    nominations_path = profile_dir / f"{side}_nominations.json"
    if probe_event is not None:
        if not nominations_path.exists():
            raise AssertionError(
                f"{probe_phase}: completed probe lacks nominations artifact"
            )
        nomination_payload = json.loads(nominations_path.read_text())
        expected_nominees = [
            (int(row["shard"]), row["fingerprint"]) for row in nominees
        ]
        stored_nominees = [
            (int(row["shard"]), row["fingerprint"])
            for row in nomination_payload["nominees"]
        ]
        if stored_nominees != expected_nominees:
            raise AssertionError(
                f"{probe_phase}: persisted nominations changed on resume"
            )
        probe_payload = json.loads(probe_event["payload_json"])
        feasible = nomination_payload["feasible"]
        probe_audit = nomination_payload["probe_audit"]
        probe_wall_s = float(probe_payload["wall_s"])
    else:
        probe_started = time.perf_counter()

        def probe(candidate: dict) -> bool:
            shard = int(candidate["shard"])
            gpu = physical_gpus[shard % len(physical_gpus)]
            key = TaskKey(probe_phase, 0, shard, candidate["fingerprint"])
            spec = _cell_spec(
                key=key,
                mode="probe",
                shard=shard,
                architecture=candidate["architecture"],
                graph_path=graph_manifest["full_graph_path"],
                cfg=cfg,
                physical_gpu=gpu,
            )
            result = _execute_task(
                db_path,
                profile_dir,
                key,
                shard,
                candidate["architecture"],
                spec,
                worker_pool=worker_pool,
            )
            return result.status == Status.OK

        from sregnas.icde27.s0 import parallel_probe_and_refill
        feasible, probe_audit = parallel_probe_and_refill(
            nominees, ranked, probe,
            lambda row: physical_gpus[int(row["shard"]) % len(physical_gpus)],
        )
        probe_cuda_boundary = _assert_cuda_phase_boundary(
            db_path=db_path,
            phase=probe_phase,
            expected_tasks=len(probe_audit),
        )
        probe_wall_s = time.perf_counter() - probe_started
        nominations_path.write_text(
            json.dumps(
                {
                    "plan": "plan1",
                    "X": cfg.graph.num_gpus * cfg.selection.k,
                    "nominees": nominees,
                    "probe_audit": probe_audit,
                    "feasible": feasible,
                    "subsets": {
                        str(x): rows
                        for x, rows in _coverage_subsets(feasible, cfg).items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        queue = TaskQueue(db_path)
        queue.add_event(
            probe_phase,
            "PROBE_COMPLETE",
            EvidenceKind.MEASURED,
            {
                "wall_s": probe_wall_s,
                "audit": probe_audit,
                "pid": os.getpid(),
                "timing_boundary": "direct_perf_counter_same_process",
                "cuda_boundary_assertion": probe_cuda_boundary,
            },
        )
        queue.close()

    subsets = _coverage_subsets(feasible, cfg)
    if retrain_event is not None:
        retrain_payload = json.loads(retrain_event["payload_json"])
        queue = TaskQueue(db_path)
        retrain_rows = queue.conn.execute(
            "SELECT result_json FROM tasks WHERE phase=? AND status='OK'",
            (retrain_phase,),
        ).fetchall()
        queue.close()
        retrained = [
            CellResult(**json.loads(row["result_json"]))
            for row in retrain_rows
            if row["result_json"] is not None
        ]
        by_fp = {row.fingerprint: row for row in retrained}
        winners = {}
        for x, subset in subsets.items():
            eligible = [
                by_fp[row["fingerprint"]]
                for row in subset
                if row["fingerprint"] in by_fp
            ]
            if len(eligible) != len(subset):
                raise AssertionError(
                    f"{retrain_phase}: completed retrain lacks X={x} results"
                )
            winners[x] = max(
                eligible,
                key=lambda result: (float(result.value), result.fingerprint),
            )
        winner = winners[primary_x]
        if winner.fingerprint != retrain_payload["winner"]:
            raise AssertionError(f"{retrain_phase}: persisted winner changed on resume")
        if float(winner.value) != float(retrain_payload["val_acc"]):
            raise AssertionError(
                f"{retrain_phase}: persisted winner value changed on resume"
            )
        retrain_wall_s = float(retrain_payload["wall_s"])
    else:
        retrain_started = time.perf_counter()
        from sregnas.icde27.ledger import event
        event("retrain_started", source="finalize before retrain dispatch")
        batches: dict[int, list[tuple[TaskKey, int, list[dict], dict]]] = {
            gpu: [] for gpu in physical_gpus
        }
        for index, candidate in enumerate(feasible):
            shard = int(candidate["shard"])
            gpu = physical_gpus[shard % len(physical_gpus)]
            key = TaskKey(retrain_phase, index, shard, candidate["fingerprint"])
            spec = _cell_spec(
                key=key,
                mode="retrain",
                shard=shard,
                architecture=candidate["architecture"],
                graph_path=graph_manifest["full_graph_path"],
                cfg=cfg,
                physical_gpu=gpu,
            )
            batches[gpu].append((key, shard, candidate["architecture"], spec))
        retrained = _execute_batches(
            db_path, profile_dir, batches, worker_pool=worker_pool
        )
        retrain_cuda_boundary = _assert_cuda_phase_boundary(
            db_path=db_path,
            phase=retrain_phase,
            expected_tasks=len(retrained),
        )
        retrain_wall_s = time.perf_counter() - retrain_started
        ok = [result for result in retrained if result.status == Status.OK]
        if not ok:
            raise AssertionError(f"{retrain_phase}: no successful retrain")
        by_fp = {result.fingerprint: result for result in ok}
        winners = {}
        for x, subset in subsets.items():
            eligible = [
                by_fp[row["fingerprint"]]
                for row in subset
                if row["fingerprint"] in by_fp
            ]
            if len(eligible) != len(subset):
                raise AssertionError(
                    f"{retrain_phase}: X={x} subset lacks successful retrains"
                )
            winners[x] = max(
                eligible,
                key=lambda result: (float(result.value), result.fingerprint),
            )
        winner = winners[primary_x]
        queue = TaskQueue(db_path)
        queue.add_event(
            retrain_phase,
            "RETRAIN_COMPLETE",
            EvidenceKind.MEASURED,
            {
                "wall_s": retrain_wall_s,
                "winner": winner.fingerprint,
                "val_acc": winner.value,
                "checkpoint_path": winner.metrics["checkpoint_path"],
                "coverages": {
                    str(x): {
                        "winner": selected.fingerprint,
                        "val_acc": float(selected.value),
                        "checkpoint_path": selected.metrics["checkpoint_path"],
                        "nominee_fingerprints": [
                            row["fingerprint"] for row in subsets[x]
                        ],
                    }
                    for x, selected in winners.items()
                },
                "test_isolation": "RETRAIN_VAL_ONLY",
                "pid": os.getpid(),
                "timing_boundary": "direct_perf_counter_same_process",
                "cuda_boundary_assertion": retrain_cuda_boundary,
            },
        )
        queue.close()

    from .campaign_timing import mark

    mark(profile_dir, "winner_selected")
    from sregnas.icde27.ledger import event
    event("winner_selected", winner=winner.fingerprint, val_acc=float(winner.value),
          tie_occurred=sum(float(r.value) == float(winner.value) for r in retrained if r.status == Status.OK) > 1,
          checkpoint_path=winner.metrics["checkpoint_path"])
    test_started = time.perf_counter()
    tested_by_fp: dict[str, CellResult] = {}
    for selected in {row.fingerprint: row for row in winners.values()}.values():
        tested_by_fp[selected.fingerprint] = _evaluate_selected_winner_test(
            cfg=cfg,
            profile_dir=profile_dir,
            db_path=db_path,
            graph_manifest=graph_manifest,
            side=side,
            winner=selected,
            worker_pool=worker_pool,
            phase=test_phase,
        )
    tested = tested_by_fp[winner.fingerprint]
    from sregnas.icde27.ledger import stop_clock
    stop_clock()
    test_cuda_boundary = _assert_cuda_phase_boundary(
        db_path=db_path, phase=test_phase, expected_tasks=len(tested_by_fp)
    )
    test_wall_s = time.perf_counter() - test_started
    queue = TaskQueue(db_path)
    queue.add_event(
        test_phase,
        "TEST_COMPLETE",
        EvidenceKind.MEASURED,
        {
            "wall_s": test_wall_s,
            "winner": tested.fingerprint,
            "test_acc": tested.metrics["test_acc"],
            "coverages": {
                str(x): {
                    "winner": selected.fingerprint,
                    "val_acc": float(selected.value),
                    "test_acc": float(
                        tested_by_fp[selected.fingerprint].metrics["test_acc"]
                    ),
                    "nominee_fingerprints": [row["fingerprint"] for row in subsets[x]],
                }
                for x, selected in winners.items()
            },
            "selection_metric": "validation_accuracy",
            "pid": os.getpid(),
            "timing_boundary": "direct_perf_counter_same_process",
            "cuda_boundary_assertion": test_cuda_boundary,
        },
    )
    queue.close()
    if close_worker_pool and worker_pool is not None:
        worker_pool.close()
    return {
        "probe_wall_s": probe_wall_s,
        "retrain_wall_s": retrain_wall_s,
        "test_wall_s": test_wall_s,
        "winner": winner.fingerprint,
        "val_acc": float(winner.value),
        "test_acc": float(tested.metrics["test_acc"]),
        "coverages": {
            str(x): {
                "winner": selected.fingerprint,
                "val_acc": float(selected.value),
                "test_acc": float(
                    tested_by_fp[selected.fingerprint].metrics["test_acc"]
                ),
                "nominee_fingerprints": [row["fingerprint"] for row in subsets[x]],
            }
            for x, selected in winners.items()
        },
        "test_isolation": "POST_SELECTION_ENFORCED",
        "worker_audit": (worker_pool.audit() if worker_pool is not None else None),
    }
