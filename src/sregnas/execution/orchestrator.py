

from __future__ import annotations


import json


import os


import time

from concurrent.futures import ThreadPoolExecutor


from pathlib import Path

from typing import Any


from sregnas.config import ExperimentConfig

from sregnas.core.contracts import EvidenceKind


from sregnas.data.graph import build_graph_stage

from sregnas.execution.pool import PersistentWorkerPool

from sregnas.reporting.report import generate_report


from sregnas.execution.queue import TaskQueue


from sregnas.paths import PROJECT_ROOT as ROOT


from .tasks import _share_physical_worker_pool
from .search import (
    _build_shared_random_pool,
    _run_cost_balanced_shared_system_search,
    _run_dynamic_shared_baseline_search,
    _run_independent_bound_system_search,
    _run_independent_dynamic_baseline_search,
    _run_search_side,
    _run_shared_search_side,
)
from .finalize import _post_search_side


def _graph_cache_key(cfg: ExperimentConfig) -> str:
    key = (
        f"{cfg.graph.dataset}_{cfg.graph.partition_strategy}_"
        f"R{cfg.graph.hierarchical_r}_N{cfg.graph.num_gpus}_a5_b5"
    )
    if cfg.graph.partition_strategy in {"random", "fi"}:
        key += f"_seed{cfg.graph.partition_seed}_{cfg.graph.partition_label_scope}"
    from sregnas.data.proxy_partition_mode import partition_mode
    if partition_mode() == "proxy_aware" and cfg.graph.partition_strategy == "fi":
        key += "_proxy_aware_v1"
    return key


def _validate_partition_cache(cfg, manifest):
    expected = {
        "dataset": cfg.graph.dataset,
        "partition_strategy": cfg.graph.partition_strategy,
        "num_shards": cfg.graph.num_gpus,
        "hierarchical_r": cfg.graph.hierarchical_r,
        "permutation_seed": cfg.graph.permutation_seed,
        "fanout_a": cfg.graph.fanout_a,
        "fanout_b": cfg.graph.fanout_b,
        "f_int": cfg.graph.f_int,
        "f_ext": cfg.graph.f_ext,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("graph cache does not match the requested partition protocol")
    if cfg.graph.partition_strategy not in {"random", "fi"}:
        return
    selection = manifest.get("partition_selection") or {}
    from sregnas.data.proxy_partition_mode import partition_mode
    if cfg.graph.partition_strategy == "fi" and selection.get("partition_mode", "legacy") != partition_mode():
        raise ValueError("graph cache does not match the requested partition protocol mode")
    if (
        manifest.get("dataset") != cfg.graph.dataset
        or manifest.get("num_shards") != cfg.graph.num_gpus
        or selection.get("requested_strategy") != cfg.graph.partition_strategy
        or selection.get("random_seed") != cfg.graph.partition_seed
        or (
            cfg.graph.partition_strategy == "fi"
            and selection.get("label_scope") != cfg.graph.partition_label_scope
        )
    ):
        raise ValueError("graph cache does not match the requested partition protocol")


def _ensure_graph(
    cfg: ExperimentConfig, run_dir: Path, db_path: Path
) -> dict[str, Any]:
    from .startup import partition_receipt

    from sregnas.icde27.x18.cold_prep import enabled as memory_prep_enabled
    if memory_prep_enabled() and cfg.runtime.graph_cache_reference:
        raise ValueError('X18 cold preparation cannot reuse a graph cache reference')

    if cfg.runtime.graph_cache_reference:
        reference = Path(cfg.runtime.graph_cache_reference)
        if not reference.is_absolute():
            reference = ROOT / reference
        if not reference.exists():
            raise FileNotFoundError(reference)
        manifest = json.loads(reference.read_text())
        _validate_partition_cache(cfg, manifest)
        queue = TaskQueue(db_path)
        queue.add_event(
            cfg.name,
            "GRAPH_STAGE_REFERENCE_REUSED",
            EvidenceKind.RAW,
            {
                "manifest": str(reference.resolve()),
                "policy": "read-only reuse of persisted validated graph stage",
            },
        )
        queue.close()
        cfg._partition_receipt = partition_receipt(reference)
        return manifest
    graph_dir = run_dir / "graph_cache" / _graph_cache_key(cfg)
    manifest_path = graph_dir / "graph" / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        _validate_partition_cache(cfg, manifest)
        cfg._partition_receipt = partition_receipt(manifest_path)
        return manifest
    started = time.perf_counter()
    manifest = build_graph_stage(cfg, graph_dir)
    queue = TaskQueue(db_path)
    queue.add_event(
        cfg.name,
        "GRAPH_STAGE_COMPLETE",
        EvidenceKind.RAW,
        {"wall_s": time.perf_counter() - started, "manifest": str(manifest_path)},
    )
    queue.close()
    cfg._partition_receipt = partition_receipt(manifest_path)
    return manifest


def _assert_products_verification_gate(
    cfg: ExperimentConfig, db_path: Path
) -> dict[str, Any] | None:
    if cfg.graph.dataset != "ogbn-products":
        return None
    from sregnas.comparison.products_approval import assert_products_method_approval

    receipt = assert_products_method_approval(cfg)
    queue = TaskQueue(db_path)
    queue.add_event(
        cfg.name, "PRODUCTS_METHOD_APPROVAL_PASS", EvidenceKind.RAW, receipt
    )
    queue.close()
    return receipt


def run_profile(cfg: ExperimentConfig, run_dir: Path, db_path: Path) -> dict[str, Any]:
    from sregnas.paths import require_gpu_permission

    require_gpu_permission()
    if cfg.runtime.comparison_method:
        raise RuntimeError("standalone comparison configs require sregnas.comparison")
    profile_dir = run_dir / "profiles" / cfg.name
    profile_dir.mkdir(parents=True, exist_ok=True)
    summary_path = profile_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    (profile_dir / "resolved_config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, sort_keys=True)
    )
    from .campaign_timing import record

    preparation_started = time.perf_counter()
    from sregnas.icde27.ledger import start_clock
    start_clock("orchestrator before _ensure_graph")
    graph_manifest = _ensure_graph(cfg, run_dir, db_path)
    record(profile_dir, cpu_preparation_s=time.perf_counter() - preparation_started)
    _assert_products_verification_gate(cfg, db_path)
    if cfg.runtime.execution_scope == "system":
        return _run_system_only(cfg, run_dir, profile_dir, db_path, graph_manifest)
    shared_pool = (
        _build_shared_random_pool(
            cfg=cfg,
            profile_dir=profile_dir,
            db_path=db_path,
        )
        if cfg.search.pool_mode == "shared_random"
        else None
    )
    balanced_shared_search = (
        shared_pool is not None
        and cfg.runtime.shared_search_scheduler
        == "cost_balanced_system_dynamic_baseline"
    )
    independent_bound_search = (
        shared_pool is None
        and cfg.runtime.shared_search_scheduler
        == "independent_bound_system_dynamic_baseline"
    )
    if independent_bound_search:
        system_search_runner = _run_independent_bound_system_search
        baseline_search_runner = _run_independent_dynamic_baseline_search
    elif balanced_shared_search:
        system_search_runner = _run_cost_balanced_shared_system_search
        baseline_search_runner = _run_dynamic_shared_baseline_search
    else:
        search_runner = (
            _run_shared_search_side if shared_pool is not None else _run_search_side
        )
        system_search_runner = search_runner
        baseline_search_runner = search_runner
    shared_kwargs = {"shared_pool": shared_pool} if shared_pool is not None else {}
    persistent = cfg.runtime.execution_mode == "persistent_worker"
    system_workers = (
        PersistentWorkerPool(profile_dir, cfg.runtime.system_gpus, f"{cfg.name}_system")
        if persistent
        else None
    )
    shared_physical_workers = _share_physical_worker_pool(cfg)
    contract_queue = TaskQueue(db_path)
    strict_continuous_sides = (
        contract_queue.get_state("search_seed_campaign") is not None
        or contract_queue.get_state("continuous_side_campaign") is not None
    )
    contract_queue.close()
    direct_side_walls: dict[str, float] = {}
    direct_side_events: dict[str, dict[str, Any]] = {}

    def start_direct_side(side: str) -> float:
        started = time.perf_counter()
        payload = {
            "side": side,
            "pid": os.getpid(),
            "start_epoch_s": time.time(),
            "contract": "single continuous same-process side execution",
        }
        direct_side_events[side] = payload
        queue = TaskQueue(db_path)
        queue.add_event(
            f"{cfg.name}:{side}",
            "DIRECT_SIDE_E2E_START",
            EvidenceKind.RAW,
            payload,
        )
        queue.close()
        return started

    def stop_direct_side(side: str, started: float) -> None:
        wall_s = time.perf_counter() - started
        direct_side_walls[side] = wall_s
        payload = {
            **direct_side_events[side],
            "stop_epoch_s": time.time(),
            "wall_s": wall_s,
        }
        direct_side_events[side] = payload
        queue = TaskQueue(db_path)
        queue.add_event(
            f"{cfg.name}:{side}",
            "DIRECT_SIDE_E2E_STOP",
            EvidenceKind.MEASURED,
            payload,
        )
        queue.close()

    baseline_workers = (
        system_workers
        if shared_physical_workers
        else (
            PersistentWorkerPool(
                profile_dir,
                cfg.runtime.baseline_gpus,
                f"{cfg.name}_baseline",
            )
            if persistent
            else None
        )
    )
    if shared_physical_workers:
        queue = TaskQueue(db_path)
        queue.add_event(
            cfg.name,
            "SHARED_PHYSICAL_WORKER_POOL_ACTIVE",
            EvidenceKind.RAW,
            {
                "gpus": cfg.runtime.system_gpus,
                "parallel_sides": False,
                "enforcement": (
                    "baseline_workers is system_workers; system post-search "
                    "does not close the pool; baseline closes it after use"
                ),
            },
        )
        queue.close()
    try:
        if cfg.runtime.parallel_sides:
            with ThreadPoolExecutor(max_workers=2) as pool:
                system_future = pool.submit(
                    system_search_runner,
                    cfg=cfg,
                    profile_dir=profile_dir,
                    db_path=db_path,
                    graph_manifest=graph_manifest,
                    side="system",
                    worker_pool=system_workers,
                    **shared_kwargs,
                )
                baseline_future = pool.submit(
                    baseline_search_runner,
                    cfg=cfg,
                    profile_dir=profile_dir,
                    db_path=db_path,
                    graph_manifest=graph_manifest,
                    side="baseline",
                    worker_pool=baseline_workers,
                    **shared_kwargs,
                )
                system_search = system_future.result()
                baseline_search = baseline_future.result()
            with ThreadPoolExecutor(max_workers=2) as pool:
                system_future = pool.submit(
                    _post_search_side,
                    cfg=cfg,
                    profile_dir=profile_dir,
                    db_path=db_path,
                    graph_manifest=graph_manifest,
                    side="system",
                    search_info=system_search,
                    worker_pool=system_workers,
                    close_worker_pool=persistent,
                )
                baseline_future = pool.submit(
                    _post_search_side,
                    cfg=cfg,
                    profile_dir=profile_dir,
                    db_path=db_path,
                    graph_manifest=graph_manifest,
                    side="baseline",
                    search_info=baseline_search,
                    worker_pool=baseline_workers,
                    close_worker_pool=persistent,
                )
                system_post = system_future.result()
                baseline_post = baseline_future.result()
        else:
            system_side_started = start_direct_side("system")
            system_search = system_search_runner(
                cfg=cfg,
                profile_dir=profile_dir,
                db_path=db_path,
                graph_manifest=graph_manifest,
                side="system",
                worker_pool=system_workers,
                **shared_kwargs,
            )
            system_post = _post_search_side(
                cfg=cfg,
                profile_dir=profile_dir,
                db_path=db_path,
                graph_manifest=graph_manifest,
                side="system",
                search_info=system_search,
                worker_pool=system_workers,
                close_worker_pool=persistent and not shared_physical_workers,
            )
            stop_direct_side("system", system_side_started)
            baseline_side_started = start_direct_side("baseline")
            baseline_search = baseline_search_runner(
                cfg=cfg,
                profile_dir=profile_dir,
                db_path=db_path,
                graph_manifest=graph_manifest,
                side="baseline",
                worker_pool=baseline_workers,
                **shared_kwargs,
            )
            baseline_post = _post_search_side(
                cfg=cfg,
                profile_dir=profile_dir,
                db_path=db_path,
                graph_manifest=graph_manifest,
                side="baseline",
                search_info=baseline_search,
                worker_pool=baseline_workers,
                close_worker_pool=persistent,
            )
            stop_direct_side("baseline", baseline_side_started)
    finally:
        if system_workers is not None:
            system_workers.close()
        if baseline_workers is not None and baseline_workers is not system_workers:
            baseline_workers.close()
    system_phase_sum = (
        system_search["wall_s"]
        + system_post["probe_wall_s"]
        + system_post["retrain_wall_s"]
        + system_post["test_wall_s"]
    )
    baseline_phase_sum = (
        baseline_search["wall_s"]
        + baseline_post["probe_wall_s"]
        + baseline_post["retrain_wall_s"]
        + baseline_post["test_wall_s"]
    )
    system_e2e = direct_side_walls.get("system", system_phase_sum)
    baseline_e2e = direct_side_walls.get("baseline", baseline_phase_sum)
    if strict_continuous_sides:
        if cfg.runtime.parallel_sides:
            raise AssertionError(
                "search-seed campaign forbids parallel/spliced side execution"
            )
        if set(direct_side_walls) != {"system", "baseline"}:
            raise AssertionError(f"{cfg.name}: missing direct continuous side timer")
        pids = {int(direct_side_events[side]["pid"]) for side in ("system", "baseline")}
        system_stop = direct_side_events["system"]["stop_epoch_s"]
        baseline_start = direct_side_events["baseline"]["start_epoch_s"]
        continuous_assertion = {
            "profile": cfg.name,
            "pids": sorted(pids),
            "current_pid": os.getpid(),
            "system_wall_s": direct_side_walls["system"],
            "baseline_wall_s": direct_side_walls["baseline"],
            "system_stop_before_baseline_start": system_stop <= baseline_start,
            "phase_order": (
                "system search->probe->retrain->winner-test; baseline "
                "search->probe->retrain->winner-test"
            ),
            "splicing_allowed": False,
        }
        if pids != {os.getpid()} or system_stop > baseline_start:
            raise AssertionError(
                f"{cfg.name}: continuous side assertion failed: "
                f"{continuous_assertion}"
            )
        queue = TaskQueue(db_path)
        queue.add_event(
            cfg.name,
            "CONTINUOUS_SIDE_EXECUTION_ASSERTION_PASS",
            EvidenceKind.RAW,
            continuous_assertion,
        )
        queue.close()
    system_scheduler = system_search.get(
        "scheduler", "round_synchronous_static_semantic_arch_to_shard"
    )
    baseline_scheduler = baseline_search.get(
        "scheduler", "round_synchronous_dynamic_work_stealing"
    )
    system_sync_contract = system_search.get("sync_contract")
    baseline_sync_contract = baseline_search.get("sync_contract")
    fixed_symmetric_barrier_free = (
        system_scheduler == baseline_scheduler == "barrier_free_per_gpu_streams"
        and bool(system_sync_contract)
        and bool(baseline_sync_contract)
        and not system_sync_contract["cross_gpu_sync_inside_search"]
        and not baseline_sync_contract["cross_gpu_sync_inside_search"]
    )
    balanced_scheduling_pair = (
        system_scheduler == "barrier_free_system_cost_balanced"
        and baseline_scheduler == "barrier_free_baseline_dynamic_work_stealing"
        and bool(system_sync_contract)
        and bool(baseline_sync_contract)
        and not system_sync_contract["cross_gpu_round_barrier"]
        and not baseline_sync_contract["cross_gpu_round_barrier"]
        and system_sync_contract.get("runtime_scheduler_mode_audit", {}).get(
            "observed_scheduler_modes"
        )
        == ["cost_balanced_assignment"]
        and baseline_sync_contract.get("runtime_scheduler_mode_audit", {}).get(
            "observed_scheduler_modes"
        )
        == ["dynamic_work_stealing"]
    )
    independent_scheduling_pair = (
        system_scheduler == "barrier_free_system_independent_controller_bound"
        and baseline_scheduler == "barrier_free_baseline_dynamic_work_stealing"
        and bool(system_sync_contract)
        and bool(baseline_sync_contract)
        and not system_sync_contract["cross_gpu_round_barrier"]
        and not baseline_sync_contract["cross_gpu_round_barrier"]
        and system_sync_contract.get("runtime_scheduler_mode_audit", {}).get(
            "observed_scheduler_modes"
        )
        == ["independent_controller_bound"]
        and baseline_sync_contract.get("runtime_scheduler_mode_audit", {}).get(
            "observed_scheduler_modes"
        )
        == ["dynamic_work_stealing"]
    )
    symmetric_barrier_free_search = (
        fixed_symmetric_barrier_free
        or balanced_scheduling_pair
        or independent_scheduling_pair
    )
    from .provenance import commit_hash
    from .campaign_timing import record

    summary = {
        "commit_hash": commit_hash(),
        "campaign_timing": record(profile_dir),
        "profile": cfg.name,
        "n": cfg.graph.num_gpus,
        "generator": cfg.search.generator,
        "pool_mode": cfg.search.pool_mode,
        "q": cfg.search.q,
        "m": cfg.search.m,
        "t": cfg.rounds,
        "selection_k": cfg.selection.k,
        "nominee_count_x": cfg.graph.num_gpus * cfg.selection.k,
        "system_physical_gpus": cfg.runtime.system_gpus,
        "baseline_physical_gpus": cfg.runtime.baseline_gpus,
        "topology_mode": (
            cfg.runtime.topology_mode
            or (
                "native"
                if len(cfg.runtime.system_gpus) == cfg.graph.num_gpus
                else f"oversubscribed_{cfg.graph.num_gpus // len(cfg.runtime.system_gpus)}x"
            )
        ),
        "parallel_sides": cfg.runtime.parallel_sides,
        "execution_mode": cfg.runtime.execution_mode,
        "system_search_s": system_search["wall_s"],
        "baseline_search_s": baseline_search["wall_s"],
        "system_probe_s": system_post["probe_wall_s"],
        "baseline_probe_s": baseline_post["probe_wall_s"],
        "system_retrain_s": system_post["retrain_wall_s"],
        "baseline_retrain_s": baseline_post["retrain_wall_s"],
        "system_post_selection_test_s": system_post["test_wall_s"],
        "baseline_post_selection_test_s": baseline_post["test_wall_s"],
        "system_e2e_s": system_e2e,
        "baseline_e2e_s": baseline_e2e,
        "system_phase_sum_s": system_phase_sum,
        "baseline_phase_sum_s": baseline_phase_sum,
        "direct_continuous_side_e2e": bool(strict_continuous_sides),
        "speedup": baseline_e2e / system_e2e,
        "system_winner": system_post["winner"],
        "baseline_winner": baseline_post["winner"],
        "system_val": system_post["val_acc"],
        "baseline_val": baseline_post["val_acc"],
        "system_test": system_post["test_acc"],
        "baseline_test": baseline_post["test_acc"],
        "system_coverages": system_post.get("coverages", {}),
        "baseline_coverages": baseline_post.get("coverages", {}),
        "winner_selection_metric": "validation_accuracy",
        "kpi2_report_metric": "test_accuracy",
        "test_isolation": "POST_SELECTION_ENFORCED",
        "system_worker_audit": system_post["worker_audit"],
        "baseline_worker_audit": baseline_post["worker_audit"],
        "shared_physical_worker_pool": shared_physical_workers,
        "system_scheduler": system_scheduler,
        "baseline_scheduler": baseline_scheduler,
        "baseline_search_graph": cfg.runtime.baseline_search_graph,
        "system_search_sync_contract": system_sync_contract,
        "baseline_search_sync_contract": baseline_sync_contract,
        "symmetric_barrier_free_search": symmetric_barrier_free_search,
        "balanced_scheduling_pair": balanced_scheduling_pair,
        "independent_scheduling_pair": independent_scheduling_pair,
        "scheduler_family": cfg.runtime.scheduler_family,
        "kpi_eligible": symmetric_barrier_free_search,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    generate_report(run_dir)
    return summary


def _run_system_only(cfg, run_dir, profile_dir, db_path, graph_manifest):
    workers = (
        PersistentWorkerPool(profile_dir, cfg.runtime.system_gpus, f"{cfg.name}_system")
        if cfg.runtime.execution_mode == "persistent_worker"
        else None
    )
    started = time.perf_counter()
    queue = TaskQueue(db_path)
    queue.add_event(
        f"{cfg.name}:system",
        "DIRECT_SIDE_E2E_START",
        EvidenceKind.RAW,
        {
            "side": "system",
            "pid": os.getpid(),
            "start_epoch_s": time.time(),
            "execution_scope": "system",
        },
    )
    queue.close()
    from .campaign_timing import mark

    mark(profile_dir, "search_started")
    try:
        search = _run_independent_bound_system_search(
            cfg=cfg,
            profile_dir=profile_dir,
            db_path=db_path,
            graph_manifest=graph_manifest,
            side="system",
            worker_pool=workers,
        )
        mark(profile_dir, "search_finished")
        post = _post_search_side(
            cfg=cfg,
            profile_dir=profile_dir,
            db_path=db_path,
            graph_manifest=graph_manifest,
            side="system",
            search_info=search,
            worker_pool=workers,
            close_worker_pool=False,
        )
        wall_s = time.perf_counter() - started
        queue = TaskQueue(db_path)
        try:
            if queue.conn.execute(
                "SELECT count(*) FROM tasks WHERE phase LIKE ?",
                (f"{cfg.name}:baseline:%",),
            ).fetchone()[0]:
                raise AssertionError("system-only execution dispatched baseline tasks")
            queue.add_event(
                f"{cfg.name}:system",
                "DIRECT_SIDE_E2E_STOP",
                EvidenceKind.MEASURED,
                {
                    "side": "system",
                    "pid": os.getpid(),
                    "wall_s": wall_s,
                    "stop_epoch_s": time.time(),
                    "execution_scope": "system",
                },
            )
        finally:
            queue.close()
    finally:
        if workers is not None:
            workers.close()
    from .provenance import commit_hash
    from .campaign_timing import record

    summary = {
        "commit_hash": commit_hash(),
        "campaign_timing": record(profile_dir),
        "profile": cfg.name,
        "execution_scope": "system",
        "completed_sides": ["system"],
        "n": cfg.graph.num_gpus,
        "generator": cfg.search.generator,
        "pool_mode": cfg.search.pool_mode,
        "q": cfg.search.q,
        "m": cfg.search.m,
        "t": cfg.rounds,
        "selection_k": cfg.selection.k,
        "nominee_count_x": cfg.graph.num_gpus * cfg.selection.k,
        "system_physical_gpus": cfg.runtime.system_gpus,
        "baseline_physical_gpus": [],
        "topology_mode": cfg.runtime.topology_mode,
        "parallel_sides": False,
        "execution_mode": cfg.runtime.execution_mode,
        "system_search_s": search["wall_s"],
        "system_probe_s": post["probe_wall_s"],
        "system_retrain_s": post["retrain_wall_s"],
        "system_post_selection_test_s": post["test_wall_s"],
        "system_e2e_s": wall_s,
        "system_phase_sum_s": sum(
            [
                search["wall_s"],
                post["probe_wall_s"],
                post["retrain_wall_s"],
                post["test_wall_s"],
            ]
        ),
        "e2e_timing_basis": "continuous_side_wall_excluding_graph_and_worker_shutdown",
        "direct_continuous_side_e2e": True,
        "baseline_e2e_s": None,
        "baseline_search_s": None,
        "baseline_val": None,
        "baseline_test": None,
        "speedup": None,
        "speedup_unavailable_reason": "baseline was not run",
        "system_winner": post["winner"],
        "system_val": post["val_acc"],
        "system_test": post["test_acc"],
        "system_coverages": post.get("coverages", {}),
        "winner_selection_metric": "validation_accuracy",
        "kpi2_report_metric": "test_accuracy",
        "test_isolation": post["test_isolation"],
        "system_worker_audit": workers.audit() if workers is not None else None,
        "shared_physical_worker_pool": False,
        "system_scheduler": search.get("scheduler"),
        "system_search_sync_contract": search.get("sync_contract"),
        "scheduler_family": cfg.runtime.scheduler_family,
        "kpi_eligible": False,
    }
    (profile_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    generate_report(run_dir)
    return summary
