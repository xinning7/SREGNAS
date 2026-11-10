

from __future__ import annotations


import json


import os

import queue as thread_queue


import subprocess

import sys

import threading


import time

from concurrent.futures import ThreadPoolExecutor, as_completed


from pathlib import Path

from typing import Any


from sregnas.config import ExperimentConfig

from sregnas.core.contracts import CellResult, EvidenceKind, Status, TaskKey


from sregnas.execution.pool import PersistentWorkerPool


from sregnas.execution.queue import TaskQueue

from sregnas.evaluation.checkpoints import (
    POST_SELECTION_TEST_MODE,
    assert_test_metric_isolated,
)


from sregnas.paths import PROJECT_ROOT as ROOT


CELL_TIMING_FIELDS = (
    "process_spawn_s",
    "cuda_init_s",
    "data_load_h2d_s",
    "train_compute_s",
    "teardown_s",
)


class ProductsCostGateStop(RuntimeError):
    pass

    def __init__(self, payload: dict[str, Any]):
        super().__init__("products cost-aligned projection exceeded the 24 GPU-h gate")
        self.payload = payload


def _safe_name(value: str) -> str:
    return value.replace(":", "_").replace("/", "_")


def _cell_spec(
    *,
    key: TaskKey,
    mode: str,
    shard: int,
    architecture: list[dict[str, Any]],
    graph_path: str,
    cfg: ExperimentConfig,
    physical_gpu: int | None = None,
) -> dict[str, Any]:
    from .options import resolve_executor

    executor_options = resolve_executor(cfg)
    epochs = (
        cfg.train.proxy_epochs
        if mode in {"proxy", "full"}
        else cfg.train.retrain_epochs
    )
    spec = {
        **executor_options,
        "expected_executor": dict(executor_options),
        "comparison_method": cfg.runtime.comparison_method,
        "require_fi_gpu": cfg.runtime.require_fi_gpu,
        "partition_strategy": (
            "none" if cfg.runtime.comparison_method else cfg.graph.partition_strategy
        ),
        "partition_seed": cfg.graph.partition_seed,
        "partition_label_scope": cfg.graph.partition_label_scope,
        "partition_parts": cfg.graph.num_gpus,
        "partition_receipt": getattr(cfg, "_partition_receipt", None),
        "phase": key.phase,
        "mode": mode,
        "fingerprint": key.fingerprint,
        "architecture": architecture,
        "gpu": key.gpu if physical_gpu is None else physical_gpu,
        "logical_gpu": key.gpu,
        "round": key.round,
        "shard": shard,
        "graph_path": graph_path,
        "epochs": epochs,
        "lr": cfg.train.lr,
        "weight_decay": cfg.train.weight_decay,
        "dropout": cfg.train.dropout,
        "dataset": cfg.graph.dataset,
        "side": (
            "system"
            if ":system:" in key.phase
            else "baseline" if ":baseline:" in key.phase else "unknown"
        ),
        "controller": shard,
        "replica": 0,
    }
    if mode in {"retrain", POST_SELECTION_TEST_MODE}:
        spec["retrain_seed_policy"] = cfg.train.retrain_seed_policy
        spec["replica"] = int(cfg.train.retrain_replica)
    if (
        cfg.graph.dataset == "ogbn-products"
        and mode != "proxy"
        and not (mode == "full" and cfg.runtime.comparison_method == "BaselineFull")
    ) or (
        cfg.graph.dataset == "reddit"
        and mode in {"probe", "retrain", POST_SELECTION_TEST_MODE}
    ):
        spec["products_graph_manifest_path"] = graph_path
        if mode == "probe":
            spec["probe_mode"] = "worst_case_upper_bound"
    if mode == "retrain" and cfg.train.retrain_seed is not None:
        spec["retrain_seed"] = int(cfg.train.retrain_seed)
    from sregnas.training.fixed_proxy import apply_options
    return apply_options(spec)


def _baseline_search_graph(cfg: ExperimentConfig) -> tuple[str, str]:
    pass
    if cfg.runtime.baseline_search_graph == "cost_aligned_uniform_5_5":
        reference = cfg.runtime.baseline_sampled_graph_reference
        if not reference:
            raise AssertionError("sampled baseline graph reference is missing")
        path = Path(reference)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(path)
        return "proxy", str(path.resolve())
    return "full", ""


def _share_physical_worker_pool(cfg: ExperimentConfig) -> bool:
    pass
    return (
        cfg.runtime.execution_mode == "persistent_worker"
        and not cfg.runtime.parallel_sides
        and cfg.runtime.system_gpus == cfg.runtime.baseline_gpus
    )


def _retryable_pretrain_failure(result: CellResult) -> bool:
    pass

    error = str(result.error or "")
    return result.status == Status.ERROR and any(
        signature in error
        for signature in (
            "architecture/fingerprint mismatch",
            "Invalid device argument",
        )
    )


def _execute_task(
    db_path: Path,
    run_dir: Path,
    key: TaskKey,
    shard: int,
    architecture: list[dict[str, Any]],
    spec: dict[str, Any],
    worker_pool: PersistentWorkerPool | None = None,
) -> CellResult:
    queue = TaskQueue(db_path)
    physical_gpu = int(spec.get("gpu", key.gpu))
    queue.ensure_task(key, shard, architecture, physical_gpu=physical_gpu)
    existing = queue.task_result(key)
    if existing is not None:
        if _retryable_pretrain_failure(existing):
            queue.requeue(key)
            queue.add_event(
                key.phase,
                "RETRYABLE_PRETRAIN_RESULT_REQUEUED",
                EvidenceKind.RAW,
                {
                    "task_key": key.as_string(),
                    "prior_error": existing.error,
                },
            )
        else:
            queue.close()
            assert_test_metric_isolated(
                str(spec.get("mode", key.phase)), existing.metrics
            )
            return existing
    task_name = _safe_name(key.as_string())
    spec = dict(spec)
    if str(spec.get("mode")) == "retrain":
        spec.setdefault(
            "checkpoint_path",
            str((run_dir / "checkpoints" / f"{task_name}.pt").resolve()),
        )
    spec_path = run_dir / "specs" / f"{task_name}.json"
    out_path = run_dir / "cells" / f"{task_name}.json"
    log_path = run_dir / "cells" / f"{task_name}.log"
    if out_path.exists():
        recovered = CellResult.load(out_path)
        if recovered.fingerprint != key.fingerprint or recovered.phase != key.phase:
            queue.close()
            raise AssertionError(
                f"stale cell output for {key.as_string()}: "
                f"{recovered.phase}/{recovered.fingerprint}"
            )
        if _retryable_pretrain_failure(recovered):
            queue.add_event(
                key.phase,
                "RETRYABLE_PRETRAIN_OUTPUT_IGNORED",
                EvidenceKind.RAW,
                {
                    "task_key": key.as_string(),
                    "out_path": str(out_path),
                    "prior_error": recovered.error,
                },
            )
        else:
            assert_test_metric_isolated(
                str(spec.get("mode", key.phase)), recovered.metrics
            )
            queue.complete(key, recovered)
            queue.add_event(
                key.phase,
                "CELL_OUTPUT_RECOVERED_WITHOUT_RERUN",
                EvidenceKind.RAW,
                {
                    "task_key": key.as_string(),
                    "out_path": str(out_path),
                    "status": recovered.status,
                    "wall_s": recovered.wall_s,
                },
            )
            queue.close()
            return recovered
    task_started_epoch_s = queue.mark_running(key)
    machine_load_start = os.getloadavg()
    if key.phase.endswith(":proxy"):
        queue.record_training_launch(
            phase=key.phase,
            round_idx=key.round,
            gpu=physical_gpu,
            shard=shard,
            fingerprint=key.fingerprint,
            architecture=architecture,
        )
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)


    spec_path.write_text(json.dumps(spec, indent=2))
    if worker_pool is not None:
        result = worker_pool.execute(physical_gpu, spec, out_path)
    else:
        from sregnas.paths import require_gpu_permission

        require_gpu_permission()
        started = time.perf_counter()
        subprocess_spec = dict(spec)
        subprocess_spec["parent_launch_perf_s"] = started
        spec_path.write_text(json.dumps(subprocess_spec, indent=2))
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        with log_path.open("a") as log:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sregnas.execution.cell",
                    "--spec",
                    str(spec_path),
                    "--out",
                    str(out_path),
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        parent_wall_s = time.perf_counter() - started
        if out_path.exists():
            result = CellResult.load(out_path)
            if result.metrics.get("timing_schema") == "products_minibatch_v1":
                result.metrics["parent_dispatch_wall_s"] = parent_wall_s
            else:
                for field in CELL_TIMING_FIELDS:
                    result.metrics.setdefault(field, 0.0)
                measured = sum(
                    float(result.metrics[field]) for field in CELL_TIMING_FIELDS
                )
                result.metrics["teardown_s"] = float(
                    result.metrics["teardown_s"]
                ) + max(0.0, parent_wall_s - measured)
                result.metrics["timing_sum_s"] = sum(
                    float(result.metrics[field]) for field in CELL_TIMING_FIELDS
                )
                result.metrics["timing_schema"] = "subprocess_v1"
            result.wall_s = parent_wall_s
            result.dump(out_path)
        else:
            result = CellResult(
                status=Status.CUDA_FAULT if proc.returncode else Status.ERROR,
                fingerprint=key.fingerprint,
                phase=key.phase,
                gpu=physical_gpu,
                round=key.round,
                shard=shard,
                architecture=architecture,
                wall_s=parent_wall_s,
                error=f"cell exited {proc.returncode} without JSON",
                metrics={
                    "process_spawn_s": parent_wall_s,
                    "cuda_init_s": 0.0,
                    "data_load_h2d_s": 0.0,
                    "train_compute_s": 0.0,
                    "teardown_s": 0.0,
                    "timing_sum_s": parent_wall_s,
                    "timing_schema": "subprocess_v1_missing_result",
                },
            )
    task_finished_epoch_s = time.time()
    machine_load_end = os.getloadavg()
    result.metrics.update(
        {
            "task_start_epoch_s": task_started_epoch_s,
            "task_end_epoch_s": task_finished_epoch_s,
            "task_gpu_s": float(result.wall_s),
            "task_physical_gpu": physical_gpu,
            "machine_load_start": {
                "load1": machine_load_start[0],
                "load5": machine_load_start[1],
                "load15": machine_load_start[2],
            },
            "machine_load_end": {
                "load1": machine_load_end[0],
                "load5": machine_load_end[1],
                "load15": machine_load_end[2],
            },
            "machine_load_logical_cpus": os.cpu_count(),
            "task_timing_contract": (
                "parent dispatch window; task_gpu_s equals parent-observed "
                "cell wall time"
            ),
        }
    )
    assert_test_metric_isolated(str(spec.get("mode", key.phase)), result.metrics)
    from .provenance import commit_hash

    result.commit_hash = commit_hash()
    result.dump(out_path)
    queue.complete(key, result)
    queue.close()
    if os.environ.get("SREGNAS_STOP_ON_INVALID") == "1" and result.status not in {
        Status.OK,
        Status.OOM,
        Status.PROBE_OOM,
    }:
        raise RuntimeError(
            f"invalid cell result {result.status}: {result.error}; evidence: {out_path}"
        )
    return result


def _execute_gpu_batch(
    db_path: Path,
    run_dir: Path,
    batch: list[tuple[TaskKey, int, list[dict[str, Any]], dict[str, Any]]],
    worker_pool: PersistentWorkerPool | None = None,
    scheduler_mode: str | None = None,
    side: str | None = None,
) -> list[CellResult]:
    results = []
    for key, shard, architecture, spec in batch:
        existing_queue = TaskQueue(db_path)
        existing = existing_queue.task_result(key)
        task_name = _safe_name(key.as_string())
        out_path = run_dir / "cells" / f"{task_name}.json"
        existing_queue.close()
        if scheduler_mode is not None and existing is None and not out_path.exists():
            dispatch = TaskQueue(db_path)
            dispatch.record_dispatch(
                task_id=key.as_string(),
                phase=key.phase,
                side=str(side),
                scheduler_mode=scheduler_mode,
                worker_id=int(spec["gpu"]),
                task_index=int(spec["dispatch_task_index"]),
                static_worker_id=int(spec["static_worker_id"]),
            )
            dispatch.close()
        results.append(
            _execute_task(
                db_path,
                run_dir,
                key,
                shard,
                architecture,
                spec,
                worker_pool=worker_pool,
            )
        )
    return results


def _execute_batches(
    db_path: Path,
    run_dir: Path,
    batches: dict[int, list[tuple[TaskKey, int, list[dict], dict]]],
    worker_pool: PersistentWorkerPool | None = None,
    scheduler_mode: str | None = None,
    side: str | None = None,
) -> list[CellResult]:
    results: list[CellResult] = []
    with ThreadPoolExecutor(max_workers=len(batches)) as pool:
        futures = [
            pool.submit(
                _execute_gpu_batch,
                db_path,
                run_dir,
                batch,
                worker_pool,
                scheduler_mode,
                side,
            )
            for batch in batches.values()
            if batch
        ]
        for future in as_completed(futures):
            results.extend(future.result())
    return results


def _execute_dynamic_batches(
    db_path: Path,
    run_dir: Path,
    physical_gpus: list[int],
    items: list[tuple[TaskKey, int, list[dict[str, Any]], dict[str, Any]]],
    worker_pool: PersistentWorkerPool | None = None,
    completion_callback: Any | None = None,
) -> list[CellResult]:
    pass
    results: list[CellResult] = []
    pending: thread_queue.Queue[
        tuple[
            int,
            tuple[TaskKey, int, list[dict[str, Any]], dict[str, Any]],
        ]
    ] = thread_queue.Queue()
    for task_index, item in enumerate(items):
        existing_queue = TaskQueue(db_path)
        existing = existing_queue.task_result(item[0])
        existing_queue.close()
        if existing is None:
            pending.put((task_index, item))
        else:
            results.append(existing)

    stop = {"requested": False, "payload": None}
    stop_lock = threading.Lock()

    def worker(physical_gpu: int) -> list[CellResult]:
        local = []
        while True:
            with stop_lock:
                if stop["requested"]:
                    break
            try:
                task_index, item = pending.get_nowait()
            except thread_queue.Empty:
                break
            key, shard, architecture, template = item
            spec = dict(template)
            spec["gpu"] = physical_gpu
            audit = TaskQueue(db_path)
            audit.record_dispatch(
                task_id=key.as_string(),
                phase=key.phase,
                side="baseline",
                scheduler_mode="dynamic_work_stealing",
                worker_id=physical_gpu,
                task_index=task_index,
                static_worker_id=physical_gpus[task_index % len(physical_gpus)],
            )
            audit.add_proposal(
                phase=key.phase,
                round_idx=key.round,
                gpu=physical_gpu,
                shard=shard,
                fingerprint=key.fingerprint,
                architecture=architecture,
                status="DYNAMIC_ASSIGNED",
                reward=None,
                consumes_q=False,
                retry_index=0,
            )
            audit.close()
            result = _execute_task(
                db_path,
                run_dir,
                key,
                shard,
                architecture,
                spec,
                worker_pool=worker_pool,
            )
            local.append(result)
            if completion_callback is not None:
                payload = completion_callback(result)
                if payload is not None and payload.get("decision") == "STOP":
                    with stop_lock:
                        stop["requested"] = True
                        stop["payload"] = payload
            pending.task_done()
        return local

    with ThreadPoolExecutor(max_workers=len(physical_gpus)) as pool:
        futures = [pool.submit(worker, physical_gpu) for physical_gpu in physical_gpus]
        for future in as_completed(futures):
            results.extend(future.result())
    if stop["requested"]:
        raise ProductsCostGateStop(dict(stop["payload"] or {}))
    if len(results) != len(items):
        raise AssertionError(
            f"dynamic baseline results={len(results)} != items={len(items)}"
        )
    return results


def _products_cost_gate_callback(
    cfg: ExperimentConfig, db_path: Path, phase: str
) -> Any | None:
    pass
    if (
        cfg.graph.dataset == "ogbn-products"
        and cfg.runtime.products_verification_reference
    ):
        reference = Path(cfg.runtime.products_verification_reference)
        if not reference.is_absolute():
            reference = ROOT / reference
        if (
            reference.exists()
            and json.loads(reference.read_text()).get("schema")
            == "products_user_directive_v1"
        ):
            from sregnas.comparison.products_protocol import confirmed_protocol

            confirmed_protocol(cfg.runtime.comparison_method or "SREGNAS", reference)
            return None
    if (
        not cfg.full_budget
        or cfg.full_budget.get("protocol") != "products_cost_aligned_sampled_v1"
    ):
        return None
    lock = threading.Lock()

    def callback(_result: CellResult) -> dict[str, Any] | None:
        with lock:
            queue = TaskQueue(db_path)
            prior = queue.get_state("products_cost_gate_first50")
            if prior is not None:
                queue.close()
                return prior if prior.get("decision") == "STOP" else None
            rows = queue.conn.execute(
                "SELECT wall_s,finished_at FROM tasks WHERE phase=? "
                "AND result_json IS NOT NULL ORDER BY finished_at,task_key LIMIT 50",
                (phase,),
            ).fetchall()
            if len(rows) < 50:
                queue.close()
                return None
            system_phase = f"{cfg.name}:system"
            system_event = queue.conn.execute(
                "SELECT payload_json FROM phase_events WHERE phase=? "
                "AND event='DIRECT_SIDE_E2E_STOP' ORDER BY id DESC LIMIT 1",
                (system_phase,),
            ).fetchone()
            search_event = queue.conn.execute(
                "SELECT payload_json FROM phase_events WHERE phase=? "
                "AND event='SEARCH_COMPLETE' ORDER BY id DESC LIMIT 1",
                (f"{cfg.name}:system:{cfg.search.generator}:proxy",),
            ).fetchone()
            if system_event is None or search_event is None:
                queue.close()
                raise AssertionError(
                    "first-50 cost gate lacks the completed fresh system anchor"
                )
            system_wall = float(json.loads(system_event["payload_json"])["wall_s"])
            system_search_wall = float(
                json.loads(search_event["payload_json"])["wall_s"]
            )
            first50_task_gpu_s = sum(float(row["wall_s"] or 0.0) for row in rows)
            physical = len(cfg.runtime.baseline_gpus)
            system_reserved_gpu_h = system_wall * physical / 3600.0
            baseline_search_projected_gpu_h = (
                first50_task_gpu_s / 50.0 * cfg.search.q / 3600.0
            )
            baseline_post_projected_gpu_h = (
                max(0.0, system_wall - system_search_wall) * physical / 3600.0
            )
            projected_total = (
                system_reserved_gpu_h
                + baseline_search_projected_gpu_h
                + baseline_post_projected_gpu_h
            )
            payload = {
                "schema": "products_cost_aligned_first50_gate_v1",
                "baseline_cells": 50,
                "first50_task_busy_gpu_s": first50_task_gpu_s,
                "mean_task_gpu_s": first50_task_gpu_s / 50.0,
                "system_reserved_gpu_h_measured": system_reserved_gpu_h,
                "baseline_search_gpu_h_projected_Q400": baseline_search_projected_gpu_h,
                "baseline_post_gpu_h_projected_from_fresh_system": baseline_post_projected_gpu_h,
                "projected_total_gpu_h": projected_total,
                "fresh_cost_gate_gpu_h": 24.0,
                "decision": "STOP" if projected_total > 24.0 else "CONTINUE",
                "projection_method": (
                    "fresh system reserved GPU-h + first-50 baseline task-busy "
                    "mean scaled to Q=400 + fresh system post-search reserved "
                    "GPU-h as the equal-retrain baseline anchor"
                ),
                "evaluated_at_epoch_s": time.time(),
            }
            queue.set_state("products_cost_gate_first50", payload)
            queue.set_state("cost_gate", payload)
            queue.add_event(
                phase,
                "FIRST_50_BASELINE_COST_PROJECTION",
                EvidenceKind.PROJECTED,
                payload,
            )
            queue.close()
            return payload if payload["decision"] == "STOP" else None

    return callback


def _runtime_scheduler_mode_audit(
    *,
    db_path: Path,
    phase: str,
    expected_mode: str,
    expected_tasks: int,
) -> dict[str, Any]:
    pass
    queue = TaskQueue(db_path)
    rows = queue.conn.execute(
        """
        SELECT scheduler_mode, COUNT(*) attempts,
               COUNT(DISTINCT task_id) tasks,
               COUNT(DISTINCT worker_id) workers
        FROM dispatch_events WHERE phase=? GROUP BY scheduler_mode
        ORDER BY scheduler_mode
        """,
        (phase,),
    ).fetchall()
    modes = {str(row["scheduler_mode"]) for row in rows}
    distinct_tasks = int(
        queue.conn.execute(
            "SELECT COUNT(DISTINCT task_id) n FROM dispatch_events " "WHERE phase=?",
            (phase,),
        ).fetchone()["n"]
    )
    detail = {
        "phase": phase,
        "expected_scheduler_mode": expected_mode,
        "observed_scheduler_modes": sorted(modes),
        "distinct_dispatched_tasks": distinct_tasks,
        "expected_tasks": expected_tasks,
        "mode_rows": [dict(row) for row in rows],
    }
    passed = modes == {expected_mode} and distinct_tasks == expected_tasks
    if not passed:
        queue.close()
        raise AssertionError(
            "runtime scheduler-mode assertion failed: "
            f"{json.dumps(detail, sort_keys=True)}"
        )
    queue.add_event(
        phase,
        "RUNTIME_SCHEDULER_MODE_ASSERTION_PASS",
        EvidenceKind.RAW,
        detail,
    )
    queue.close()
    return detail


def _start_timed_phase_segment(
    *, db_path: Path, phase: str
) -> tuple[float, float, float]:
    pass
    queue = TaskQueue(db_path)
    carries = queue.get_state("phase_wall_carry_s", {})
    carry_s = float(carries.get(phase, 0.0))
    epoch_s = time.time()
    perf_s = time.perf_counter()
    queue.add_event(
        phase,
        "TIMED_PHASE_SEGMENT_START",
        EvidenceKind.RAW,
        {
            "pid": os.getpid(),
            "start_epoch_s": epoch_s,
            "prior_active_wall_carry_s": carry_s,
        },
    )
    queue.close()
    return perf_s, epoch_s, carry_s


def _finish_timed_phase_segment(
    *,
    db_path: Path,
    phase: str,
    perf_start_s: float,
    epoch_start_s: float,
    carry_s: float,
) -> tuple[float, float]:
    active_s = time.perf_counter() - perf_start_s
    total_s = carry_s + active_s
    queue = TaskQueue(db_path)
    carries = queue.get_state("phase_wall_carry_s", {})
    carries[phase] = 0.0
    queue.set_state("phase_wall_carry_s", carries)
    queue.add_event(
        phase,
        "TIMED_PHASE_SEGMENT_STOP",
        EvidenceKind.RAW,
        {
            "pid": os.getpid(),
            "start_epoch_s": epoch_start_s,
            "stop_epoch_s": time.time(),
            "active_segment_wall_s": active_s,
            "prior_active_wall_carry_s": carry_s,
            "total_active_wall_s": total_s,
        },
    )
    queue.close()
    return total_s, active_s


def _completed_search_result(
    *, db_path: Path, phase: str, expected_tasks: int, scheduler: str
) -> dict[str, Any] | None:
    pass
    queue = TaskQueue(db_path)
    event = queue.conn.execute(
        "SELECT payload_json FROM phase_events "
        "WHERE phase=? AND event='SEARCH_COMPLETE' "
        "ORDER BY id DESC LIMIT 1",
        (phase,),
    ).fetchone()
    task_state = queue.conn.execute(
        "SELECT COUNT(*) n, "
        "SUM(CASE WHEN status IN ('PENDING','RUNNING') THEN 1 ELSE 0 END) incomplete "
        "FROM tasks WHERE phase=?",
        (phase,),
    ).fetchone()
    if (
        event is None
        or int(task_state["n"]) != expected_tasks
        or int(task_state["incomplete"] or 0) != 0
    ):
        queue.close()
        return None
    payload = json.loads(event["payload_json"])
    result = {
        "phase": phase,
        "wall_s": float(payload["wall_s"]),
        "launches": int(payload["trained_launches"]),
        "scheduler": scheduler,
        "sync_contract": payload.get("sync_contract"),
    }
    queue.add_event(
        phase,
        "SEARCH_RESULT_REUSED_AFTER_RESUME",
        EvidenceKind.RAW,
        result,
    )
    queue.close()
    return result


def _task_rows(db_path: Path, phase: str) -> list[dict[str, Any]]:
    queue = TaskQueue(db_path)
    rows = []
    for row in queue.conn.execute("SELECT * FROM tasks WHERE phase=?", (phase,)):
        rows.append(
            {
                "phase": row["phase"],
                "round": int(row["round"]),
                "gpu": int(row["gpu"]),
                "shard": int(row["shard"]),
                "fingerprint": row["fingerprint"],
                "architecture": json.loads(row["architecture_json"]),
                "status": row["status"],
                "value": row["value"],
            }
        )
    queue.close()
    return rows


def _assert_cuda_phase_boundary(
    *, db_path: Path, phase: str, expected_tasks: int
) -> dict[str, Any]:
    pass
    queue = TaskQueue(db_path)
    rows = queue.conn.execute(
        "SELECT fingerprint,status,result_json FROM tasks WHERE phase=?",
        (phase,),
    ).fetchall()
    missing: list[dict[str, str]] = []
    for row in rows:
        result = json.loads(row["result_json"] or "{}")
        synchronized = result.get("metrics", {}).get("cuda_boundary_synchronized")
        if synchronized is not True:
            missing.append(
                {
                    "fingerprint": str(row["fingerprint"]),
                    "status": str(row["status"]),
                }
            )
    assertion = {
        "phase": phase,
        "expected_tasks": int(expected_tasks),
        "observed_tasks": len(rows),
        "synchronized_tasks": len(rows) - len(missing),
        "missing": missing,
        "contract": (
            "every worker executed torch.cuda.synchronize immediately "
            "before returning its cell result"
        ),
        "pid": os.getpid(),
    }
    if len(rows) != expected_tasks or missing:
        queue.close()
        raise AssertionError(
            f"{phase}: CUDA phase-boundary assertion failed: {assertion}"
        )
    queue.add_event(
        phase,
        "CUDA_PHASE_BOUNDARY_ASSERTION_PASS",
        EvidenceKind.RAW,
        assertion,
    )
    queue.close()
    return assertion
