

from __future__ import annotations

import ast


import inspect

import json


import statistics


import textwrap

import time

from concurrent.futures import ThreadPoolExecutor, as_completed


from pathlib import Path

from typing import Any

from sregnas.core import Architecture

from sregnas.config import ExperimentConfig

from sregnas.core.contracts import EvidenceKind, Status, TaskKey

from sregnas.search.controller import Proposal, STATIC_GATE_PREDICATE, ShardController

from sregnas.data.policy import dataset_policy, system_static_gate_enabled


from sregnas.execution.pool import PersistentWorkerPool


from sregnas.execution.queue import TaskQueue


from .tasks import (
    _assert_cuda_phase_boundary,
    _baseline_search_graph,
    _cell_spec,
    _completed_search_result,
    _execute_batches,
    _execute_dynamic_batches,
    _execute_gpu_batch,
    _finish_timed_phase_segment,
    _products_cost_gate_callback,
    _runtime_scheduler_mode_audit,
    _start_timed_phase_segment,
)


def _assert_barrier_free_shared_search_route() -> dict[str, Any]:
    pass

    def calls(function: Any) -> set[str]:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        names = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
        return names

    shared_calls = calls(_run_shared_search_side)
    stream_calls = calls(_execute_gpu_batch)
    dispatch_calls = calls(_execute_batches)
    forbidden = {
        "_execute_dynamic_batches",
        "Barrier",
        "Condition",
        "Event",
        "Queue",
        "join",
        "wait",
    }
    violations = sorted((shared_calls | stream_calls | dispatch_calls) & forbidden)
    if violations:
        raise AssertionError(
            "cross-GPU synchronization reachable inside shared search: " f"{violations}"
        )
    if "_execute_batches" not in shared_calls:
        raise AssertionError("shared search does not use fixed stream dispatch")
    if "as_completed" not in dispatch_calls:
        raise AssertionError("shared search lacks the allowed final join")
    return {
        "result": "PASS",
        "forbidden_reachable_calls": violations,
        "allowed_final_join": "as_completed after complete stream dispatch",
        "shared_route_calls": sorted(shared_calls),
        "stream_route_calls": sorted(stream_calls),
    }


def _assert_barrier_free_balanced_search_routes() -> dict[str, Any]:
    pass

    def calls(function: Any) -> set[str]:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        names = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
        return names

    system_calls = calls(_run_cost_balanced_shared_system_search)
    baseline_calls = calls(_run_dynamic_shared_baseline_search)
    static_dispatch_calls = calls(_execute_batches)
    dynamic_dispatch_calls = calls(_execute_dynamic_batches)
    forbidden = {"Barrier", "Condition", "Event", "join", "wait"}
    violations = sorted(
        (system_calls | baseline_calls | static_dispatch_calls | dynamic_dispatch_calls)
        & forbidden
    )
    if violations:
        raise AssertionError(
            "cross-GPU round synchronization reachable inside final-口径 "
            f"search: {violations}"
        )
    if "_execute_batches" not in system_calls:
        raise AssertionError("system route lacks fixed stream dispatch")
    if "_execute_dynamic_batches" not in baseline_calls:
        raise AssertionError("baseline route lacks dynamic work stealing")
    if "as_completed" not in static_dispatch_calls:
        raise AssertionError("system route lacks allowed final join")
    if "as_completed" not in dynamic_dispatch_calls:
        raise AssertionError("baseline route lacks allowed final join")
    return {
        "result": "PASS",
        "forbidden_reachable_calls": violations,
        "system_route": ("_run_cost_balanced_shared_system_search->_execute_batches"),
        "baseline_route": (
            "_run_dynamic_shared_baseline_search->_execute_dynamic_batches"
        ),
        "allowed_final_join": (
            "as_completed only after all per-GPU streams/workers are dispatched"
        ),
    }


def _assert_independent_bound_search_routes() -> dict[str, Any]:
    pass

    def calls(function: Any) -> set[str]:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        names = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
        return names

    system_calls = calls(_run_independent_bound_system_search)
    stream_calls = calls(_execute_gpu_batch)
    baseline_calls = calls(_run_independent_dynamic_baseline_search)
    dynamic_calls = calls(_execute_dynamic_batches)
    forbidden_system = {


        "_execute_dynamic_batches",
        "Barrier",
        "Condition",
        "Event",
        "Queue",
        "join",
        "wait",
        "all_reduce",
        "broadcast",
    }
    system_violations = sorted((system_calls | stream_calls) & forbidden_system)
    forbidden_baseline = {
        "Barrier",
        "Condition",
        "Event",
        "join",
        "wait",
        "all_reduce",
        "broadcast",
    }
    baseline_violations = sorted((baseline_calls | dynamic_calls) & forbidden_baseline)
    if system_violations or baseline_violations:
        raise AssertionError(
            "independent final-口径 route exposes forbidden synchronization: "
            f"system={system_violations} baseline={baseline_violations}"
        )
    if "_execute_gpu_batch" not in system_calls:
        raise AssertionError(
            "independent system route lacks controller-bound local dispatch"
        )
    if "_execute_dynamic_batches" not in baseline_calls:
        raise AssertionError("independent baseline route lacks dynamic work stealing")
    if "as_completed" not in system_calls:
        raise AssertionError("independent system route lacks final-only join")
    return {
        "result": "PASS",
        "system_forbidden_reachable_calls": system_violations,
        "baseline_forbidden_reachable_calls": baseline_violations,
        "system_route": (
            "_run_independent_bound_system_search->per_controller_stream->"
            "_execute_gpu_batch"
        ),
        "baseline_route": (
            "_run_independent_dynamic_baseline_search->" "_execute_dynamic_batches"
        ),
        "system_allowed_final_join": (
            "final join across controllers; replicated_2x additionally joins the two workers of each controller before its observe"
        ),
        "baseline_controller_boundary": (
            "central controller observes after each M-result batch; dynamic "
            "workers have no fixed assignment or equal-count barrier"
        ),
    }


def _propose_launch(
    *,
    queue: TaskQueue,
    controller: ShardController,
    phase: str,
    round_idx: int,
    gpu: int,
    shard: int,
    static_gate: bool,
    floor_reward: float,
    max_resamples: int,
) -> tuple[Proposal, list[tuple[Proposal, float]]]:
    rejected: list[tuple[Proposal, float]] = []
    for retry in range(max_resamples + 1):
        proposal = controller.propose()
        fp = proposal.architecture.fingerprint()
        status = controller.classify(proposal, static_gate=static_gate)
        if status == "ACCEPT":
            queue.add_proposal(
                phase=phase,
                round_idx=round_idx,
                gpu=gpu,
                shard=shard,
                fingerprint=fp,
                architecture=proposal.architecture.arch_description,
                status="ADMITTED",
                reward=None,
                consumes_q=False,
                retry_index=retry,
            )
            return proposal, rejected
        queue.add_proposal(
            phase=phase,
            round_idx=round_idx,
            gpu=gpu,
            shard=shard,
            fingerprint=fp,
            architecture=proposal.architecture.arch_description,
            status=status,
            reward=floor_reward,
            consumes_q=False,
            retry_index=retry,
        )
        rejected.append((proposal, floor_reward))
    queue.add_event(
        phase,
        "RESAMPLE_EXHAUSTED",
        EvidenceKind.RAW,
        {"round": round_idx, "gpu": gpu, "shard": shard},
    )
    raise RuntimeError(
        f"{phase} round={round_idx} gpu={gpu}: exhausted {max_resamples} resamples"
    )


def _run_search_side(
    *,
    cfg: ExperimentConfig,
    profile_dir: Path,
    db_path: Path,
    graph_manifest: dict[str, Any],
    side: str,
    worker_pool: PersistentWorkerPool | None = None,
    gpu_hard_gate: float | None = None,
    initial_gpu_s: float = 0.0,
    per_cell_guard_s: float | None = None,
) -> dict[str, Any]:
    queue = TaskQueue(db_path)
    phase = f"{cfg.name}:{side}:{cfg.search.generator}:proxy"
    queue.set_state("current_phase", phase)
    is_system = side == "system"
    physical_gpus = cfg.runtime.system_gpus if is_system else cfg.runtime.baseline_gpus
    controller_count = cfg.graph.num_gpus if is_system else 1
    controllers = [
        ShardController(
            cfg.search,
            cfg.search.base_seed + (index if is_system else 10_000),
        )
        for index in range(controller_count)
    ]
    checkpoint_dir = profile_dir / "controllers" / side
    started, segment_epoch_s, carry_s = _start_timed_phase_segment(
        db_path=db_path, phase=phase
    )
    guard_s = per_cell_guard_s

    def measured_gpu_s() -> float:
        row = queue.conn.execute(
            "SELECT COALESCE(SUM(wall_s),0) total FROM tasks WHERE phase=?",
            (phase,),
        ).fetchone()
        return float(initial_gpu_s) + float(row["total"])

    for round_idx in range(cfg.rounds):
        checkpoint_paths = [
            checkpoint_dir / f"controller_{index}_round_{round_idx}.pt"
            for index in range(controller_count)
        ]
        round_state = queue.conn.execute(
            """
            SELECT COUNT(*) n,
                   SUM(CASE WHEN status IN ('PENDING','RUNNING') THEN 1 ELSE 0 END) incomplete
            FROM tasks WHERE phase=? AND round=?
            """,
            (phase, round_idx),
        ).fetchone()
        round_complete = (
            int(round_state["n"]) == cfg.search.m
            and int(round_state["incomplete"] or 0) == 0
        )
        if all(path.exists() for path in checkpoint_paths) and round_complete:
            for controller, path in zip(controllers, checkpoint_paths):
                controller.load(path)
            continue
        current_gpu_s = measured_gpu_s()
        if gpu_hard_gate is not None:
            gate_s = float(gpu_hard_gate) * 3600.0
            next_round_guard_s = (
                cfg.search.m * float(guard_s) if guard_s is not None else 0.0
            )
            if current_gpu_s + next_round_guard_s > gate_s:
                queue.add_event(
                    phase,
                    "COST_GATE_STOP",
                    EvidenceKind.MEASURED,
                    {
                        "round": round_idx,
                        "measured_gpu_s": current_gpu_s,
                        "next_round_guard_s": next_round_guard_s,
                        "hard_gate_gpu_h": gpu_hard_gate,
                    },
                )
                queue.close()
                raise RuntimeError(
                    f"{phase}: hard GPU gate would be exceeded before round "
                    f"{round_idx}"
                )
        proposals_by_controller: list[list[tuple[Proposal, float]]] = [
            [] for _ in controllers
        ]
        accepted_by_key: dict[tuple[int, str], tuple[int, Proposal]] = {}
        batches: dict[int, list[tuple[TaskKey, int, list[dict], dict[str, Any]]]] = {
            gpu: [] for gpu in physical_gpus
        }
        dynamic_items: list[tuple[TaskKey, int, list[dict], dict[str, Any]]] = []
        if is_system:
            for shard, controller in enumerate(controllers):
                gpu = physical_gpus[shard % len(physical_gpus)]
                for _ in range(cfg.launches_per_shard_round):
                    proposal, rejected = _propose_launch(
                        queue=queue,
                        controller=controller,
                        phase=phase,
                        round_idx=round_idx,
                        gpu=gpu,
                        shard=shard,
                        static_gate=system_static_gate_enabled(cfg.graph.dataset),
                        floor_reward=cfg.search.floor_reward,
                        max_resamples=cfg.search.max_resamples_per_slot,
                    )
                    proposals_by_controller[shard].extend(rejected)
                    fp = proposal.architecture.fingerprint()
                    accepted_by_key[(shard, fp)] = (shard, proposal)


                    key = TaskKey(phase, round_idx, shard, fp)
                    graph_path = graph_manifest["shards"][shard]["path"]
                    spec = _cell_spec(
                        key=key,
                        mode="proxy",
                        shard=shard,
                        architecture=proposal.architecture.arch_description,
                        graph_path=graph_path,
                        cfg=cfg,
                        physical_gpu=gpu,
                    )
                    batches[gpu].append(
                        (key, shard, proposal.architecture.arch_description, spec)
                    )
        else:
            controller = controllers[0]
            for slot in range(cfg.search.m):
                shard = slot % cfg.graph.num_gpus
                proposal, rejected = _propose_launch(
                    queue=queue,
                    controller=controller,
                    phase=phase,
                    round_idx=round_idx,
                    gpu=-1,
                    shard=shard,
                    static_gate=False,
                    floor_reward=cfg.search.floor_reward,
                    max_resamples=cfg.search.max_resamples_per_slot,
                )
                proposals_by_controller[0].extend(rejected)
                fp = proposal.architecture.fingerprint()
                accepted_by_key[(shard, fp)] = (0, proposal)
                key = TaskKey(phase, round_idx, shard, fp)
                spec = _cell_spec(
                    key=key,
                    mode="full",
                    shard=shard,
                    architecture=proposal.architecture.arch_description,
                    graph_path=graph_manifest["full_graph_path"],
                    cfg=cfg,
                    physical_gpu=physical_gpus[0],
                )
                dynamic_items.append(
                    (key, shard, proposal.architecture.arch_description, spec)
                )
        results = (
            _execute_batches(db_path, profile_dir, batches, worker_pool=worker_pool)
            if is_system
            else _execute_dynamic_batches(
                db_path,
                profile_dir,
                physical_gpus,
                dynamic_items,
                worker_pool=worker_pool,
            )
        )
        if results:




            observed_guard_s = 1.25 * statistics.mean(
                float(result.wall_s) for result in results
            )
            guard_s = max(float(guard_s or 0.0), observed_guard_s)
        current_gpu_s = measured_gpu_s()
        if gpu_hard_gate is not None and current_gpu_s > float(gpu_hard_gate) * 3600.0:
            queue.add_event(
                phase,
                "COST_GATE_EXCEEDED_STOP",
                EvidenceKind.MEASURED,
                {
                    "round": round_idx,
                    "measured_gpu_s": current_gpu_s,
                    "hard_gate_gpu_h": gpu_hard_gate,
                },
            )
            queue.close()
            raise RuntimeError(f"{phase}: hard GPU gate exceeded")
        for result in results:
            ctrl_idx, proposal = accepted_by_key[(result.shard, result.fingerprint)]
            reward = (
                float(result.value)
                if result.status == Status.OK and result.value is not None
                else cfg.search.floor_reward
            )
            proposals_by_controller[ctrl_idx].append((proposal, reward))
        for index, (controller, observations) in enumerate(
            zip(controllers, proposals_by_controller)
        ):
            update = controller.observe(
                observations,
                launches_per_shard_round=(
                    cfg.launches_per_shard_round if is_system else cfg.search.m
                ),
                flush=round_idx == cfg.rounds - 1,
            )
            queue.add_event(
                phase,
                "CONTROLLER_OBSERVE",
                EvidenceKind.MEASURED,
                {"round": round_idx, "controller": index, **update},
            )
            controller.save(checkpoint_paths[index])
    wall_s, active_segment_wall_s = _finish_timed_phase_segment(
        db_path=db_path,
        phase=phase,
        perf_start_s=started,
        epoch_start_s=segment_epoch_s,
        carry_s=carry_s,
    )
    rows = queue.conn.execute(
        "SELECT COUNT(*) n FROM tasks WHERE phase=?", (phase,)
    ).fetchone()
    launches = int(rows["n"])
    if launches != cfg.search.q:
        raise AssertionError(
            f"{phase}: trained launches {launches} != Q={cfg.search.q}"
        )
    queue.add_event(
        phase,
        "SEARCH_COMPLETE",
        EvidenceKind.MEASURED,
        {"wall_s": wall_s, "trained_launches": launches},
    )
    final_measured_gpu_h = measured_gpu_s() / 3600.0
    queue.close()
    return {
        "phase": phase,
        "wall_s": wall_s,
        "launches": launches,
        "measured_gpu_h": final_measured_gpu_h,
        "hard_gate_gpu_h": gpu_hard_gate,
    }


def _run_independent_bound_system_search(
    *,
    cfg: ExperimentConfig,
    profile_dir: Path,
    db_path: Path,
    graph_manifest: dict[str, Any],
    side: str,
    worker_pool: PersistentWorkerPool | None = None,
) -> dict[str, Any]:
    pass
    if side != "system":
        raise AssertionError("controller-bound independent search is system-only")
    phase = f"{cfg.name}:system:{cfg.search.generator}:proxy"
    persisted = _completed_search_result(
        db_path=db_path,
        phase=phase,
        expected_tasks=cfg.search.q,
        scheduler="barrier_free_system_independent_controller_bound",
    )
    if persisted is not None:
        return persisted
    physical_gpus = cfg.runtime.system_gpus
    replicated = cfg.runtime.topology_mode == "replicated_2x"
    if not replicated and cfg.graph.num_gpus % len(physical_gpus):
        raise AssertionError(
            "controller-bound search requires an even logical-shard mapping"
        )
    oversubscription_factor = cfg.graph.num_gpus / len(physical_gpus)
    controller_seeds = [
        cfg.search.base_seed + shard for shard in range(cfg.graph.num_gpus)
    ]
    controllers = [ShardController(cfg.search, seed) for seed in controller_seeds]
    reachability = _assert_independent_bound_search_routes()
    sync_contract = {
        "route": "barrier_free_independent_controller_bound_streams",
        "cross_gpu_communication_inside_search": False,
        "cross_gpu_round_barrier": False,
        "per_controller_launches_per_round": (cfg.launches_per_shard_round),
        "controller_seeds": controller_seeds,
        "physical_gpus": physical_gpus,
        "logical_shard_to_physical_gpu": {
            str(shard): physical_gpus[shard % len(physical_gpus)]
            for shard in range(cfg.graph.num_gpus)
        },
        "oversubscription_factor": oversubscription_factor,
        "logical_shard_to_physical_workers": {
            str(shard): (
                physical_gpus[shard :: cfg.graph.num_gpus]
                if replicated
                else [physical_gpus[shard % len(physical_gpus)]]
            )
            for shard in range(cfg.graph.num_gpus)
        },
        "same_gpu_controller_serialization": (
            "independent per-shard process/lock; CUDA concurrency managed by MPS"
            if worker_pool is not None and hasattr(worker_pool, "shard_workers")
            else "PersistentGpuWorker lock; no cross-GPU communication"
        ),
        "gpu_worker_binding": (
            "one_process_per_shard"
            if worker_pool is not None and hasattr(worker_pool, "shard_workers")
            else (
                "one_process_per_physical_gpu"
                if worker_pool is not None
                else "subprocess_per_cell"
            )
        ),
        "system_architecture_bound_to_proposer_shard": True,
        "static_gate_at_proposal": system_static_gate_enabled(cfg.graph.dataset),
        "static_gate_predicate": dataset_policy(
            cfg.graph.dataset
        ).static_gate_predicate,
        "allowed_final_rendezvous": "final_join_before_nomination",
        "static_reachability_assertion": reachability,
    }
    queue = TaskQueue(db_path)
    queue.set_state("current_phase", phase)
    queue.add_event(
        phase,
        "INDEPENDENT_CONTROLLER_BOUND_SEARCH_ACTIVE",
        EvidenceKind.RAW,
        sync_contract,
    )
    queue.close()
    checkpoint_dir = profile_dir / "controllers" / side

    def controller_stream(shard: int, controller: ShardController) -> int:
        gpu = physical_gpus[shard % len(physical_gpus)]
        completed = 0
        local_queue = TaskQueue(db_path)
        for round_idx in range(cfg.rounds):
            checkpoint_path = (
                checkpoint_dir / f"controller_{shard}_round_{round_idx}.pt"
            )
            round_state = local_queue.conn.execute(
                """
                SELECT COUNT(*) n,
                       SUM(CASE WHEN status IN ('PENDING','RUNNING')
                                THEN 1 ELSE 0 END) incomplete
                FROM tasks WHERE phase=? AND round=? AND shard=?
                """,
                (phase, round_idx, shard),
            ).fetchone()
            round_complete = (
                int(round_state["n"]) == cfg.launches_per_shard_round
                and int(round_state["incomplete"] or 0) == 0
            )
            if checkpoint_path.exists() and round_complete:
                controller.load(checkpoint_path)
                completed += cfg.launches_per_shard_round
                continue
            replayed_proposals = local_queue.mark_controller_round_for_replay(
                phase=phase, round_idx=round_idx, shard=shard
            )
            if replayed_proposals:
                local_queue.add_event(
                    phase,
                    "CONTROLLER_ROUND_DETERMINISTIC_REPLAY",
                    EvidenceKind.RAW,
                    {
                        "round": round_idx,
                        "controller": shard,
                        "seed": controller.seed,
                        "historical_proposal_rows": replayed_proposals,
                        "policy": (
                            "reload prior checkpoint and replay identical RNG "
                            "stream; completed cell outputs are reused"
                        ),
                    },
                )
            observations: list[tuple[Proposal, float]] = []
            accepted: dict[str, Proposal] = {}
            batch = []
            for local_slot in range(cfg.launches_per_shard_round):
                if replicated:
                    gpu = physical_gpus[shard + (local_slot % 2) * cfg.graph.num_gpus]
                proposal, rejected = _propose_launch(
                    queue=local_queue,
                    controller=controller,
                    phase=phase,
                    round_idx=round_idx,
                    gpu=gpu,
                    shard=shard,
                    static_gate=system_static_gate_enabled(cfg.graph.dataset),
                    floor_reward=cfg.search.floor_reward,
                    max_resamples=cfg.search.max_resamples_per_slot,
                )
                observations.extend(rejected)
                fingerprint = proposal.architecture.fingerprint()
                accepted[fingerprint] = proposal
                key = TaskKey(phase, round_idx, shard, fingerprint)
                spec = _cell_spec(
                    key=key,
                    mode="proxy",
                    shard=shard,
                    architecture=proposal.architecture.arch_description,
                    graph_path=graph_manifest["shards"][shard]["path"],
                    cfg=cfg,
                    physical_gpu=gpu,
                )
                task_index = (
                    round_idx * cfg.search.m + local_slot * cfg.graph.num_gpus + shard
                )
                spec["dispatch_task_index"] = task_index
                spec["static_worker_id"] = physical_gpus[
                    task_index % len(physical_gpus)
                ]
                batch.append(
                    (
                        key,
                        shard,
                        proposal.architecture.arch_description,
                        spec,
                    )
                )
            if replicated:
                batches = {}
                for item in batch:
                    batches.setdefault(item[3]["gpu"], []).append(item)
                unordered = _execute_batches(
                    db_path,
                    profile_dir,
                    batches,
                    worker_pool=worker_pool,
                    scheduler_mode="independent_controller_bound",
                    side="system",
                )
                by_fingerprint = {r.fingerprint: r for r in unordered}
                results = [by_fingerprint[item[0].fingerprint] for item in batch]
            else:
                results = _execute_gpu_batch(
                    db_path,
                    profile_dir,
                    batch,
                    worker_pool=worker_pool,
                    scheduler_mode="independent_controller_bound",
                    side="system",
                )
            for result in results:
                proposal = accepted[result.fingerprint]
                reward = (
                    float(result.value)
                    if result.status == Status.OK and result.value is not None
                    else cfg.search.floor_reward
                )
                observations.append((proposal, reward))
            update = controller.observe(
                observations,
                launches_per_shard_round=cfg.launches_per_shard_round,
                flush=round_idx == cfg.rounds - 1,
            )
            local_queue.add_event(
                phase,
                "CONTROLLER_OBSERVE",
                EvidenceKind.MEASURED,
                {"round": round_idx, "controller": shard, **update},
            )
            controller.save(checkpoint_path)
            completed += len(results)
        local_queue.close()
        return completed

    started, segment_epoch_s, carry_s = _start_timed_phase_segment(
        db_path=db_path, phase=phase
    )
    with ThreadPoolExecutor(max_workers=cfg.graph.num_gpus) as pool:
        futures = [
            pool.submit(controller_stream, shard, controller)
            for shard, controller in enumerate(controllers)
        ]
        completed_by_stream = [future.result() for future in as_completed(futures)]
    wall_s, active_segment_wall_s = _finish_timed_phase_segment(
        db_path=db_path,
        phase=phase,
        perf_start_s=started,
        epoch_start_s=segment_epoch_s,
        carry_s=carry_s,
    )
    cuda_boundary_assertion = _assert_cuda_phase_boundary(
        db_path=db_path, phase=phase, expected_tasks=cfg.search.q
    )
    runtime_mode_audit = _runtime_scheduler_mode_audit(
        db_path=db_path,
        phase=phase,
        expected_mode="independent_controller_bound",
        expected_tasks=cfg.search.q,
    )
    sync_contract["runtime_scheduler_mode_audit"] = runtime_mode_audit
    sync_contract["completed_by_stream"] = completed_by_stream
    queue = TaskQueue(db_path)
    launches = int(
        queue.conn.execute(
            "SELECT COUNT(*) n FROM tasks WHERE phase=?", (phase,)
        ).fetchone()["n"]
    )
    if launches != cfg.search.q:
        queue.close()
        raise AssertionError(
            f"{phase}: trained launches {launches} != Q={cfg.search.q}"
        )
    queue.add_event(
        phase,
        "SEARCH_COMPLETE",
        EvidenceKind.MEASURED,
        {
            "wall_s": wall_s,
            "active_segment_wall_s": active_segment_wall_s,
            "recovered_active_wall_s": carry_s,
            "trained_launches": launches,
            "pool_mode": "independent",
            "sync_contract": sync_contract,
            "cuda_boundary_assertion": cuda_boundary_assertion,
        },
    )
    queue.close()
    return {
        "phase": phase,
        "wall_s": wall_s,
        "launches": launches,
        "scheduler": "barrier_free_system_independent_controller_bound",
        "sync_contract": sync_contract,
    }


def _run_independent_dynamic_baseline_search(
    *,
    cfg: ExperimentConfig,
    profile_dir: Path,
    db_path: Path,
    graph_manifest: dict[str, Any],
    side: str,
    worker_pool: PersistentWorkerPool | None = None,
) -> dict[str, Any]:
    pass
    if side != "baseline":
        raise AssertionError("independent dynamic search is baseline-only")
    phase = f"{cfg.name}:baseline:{cfg.search.generator}:proxy"
    persisted = _completed_search_result(
        db_path=db_path,
        phase=phase,
        expected_tasks=cfg.search.q,
        scheduler="barrier_free_baseline_dynamic_work_stealing",
    )
    if persisted is not None:
        return persisted
    physical_gpus = cfg.runtime.baseline_gpus
    controller_seed = cfg.search.base_seed + 10_000
    controller = ShardController(cfg.search, controller_seed)
    baseline_static_gate = bool(cfg.runtime.baseline_static_gate)
    baseline_mode, sampled_graph_path = _baseline_search_graph(cfg)
    cost_gate_callback = _products_cost_gate_callback(cfg, db_path, phase)
    checkpoint_dir = profile_dir / "controllers" / side
    reachability = _assert_independent_bound_search_routes()
    sync_contract = {
        "route": "barrier_free_baseline_dynamic_work_stealing",
        "cross_gpu_round_barrier": False,
        "work_queue_scope": "M_semantically_interchangeable_cells_per_controller_round",
        "central_controller_seed": controller_seed,
        "central_controller_batch_boundary": True,
        "static_gate_at_proposal": baseline_static_gate,
        "baseline_search_graph": cfg.runtime.baseline_search_graph,
        "static_gate_predicate": (
            STATIC_GATE_PREDICATE if baseline_static_gate else None
        ),
        "static_gate_reason": (
            "approved direct ablation: baseline uses the same narrowed "
            "memory-model proposal gate as system"
            if baseline_static_gate
            else "baseline is ungated and pays every proposed full-graph "
            "cell, including measured-boundary OOMs"
        ),
        "allowed_final_rendezvous": "controller_observe_after_each_M_result_batch",
        "static_reachability_assertion": reachability,
    }
    queue = TaskQueue(db_path)
    queue.set_state("current_phase", phase)
    queue.add_event(
        phase,
        "BASELINE_DYNAMIC_WORK_STEALING_ACTIVE",
        EvidenceKind.RAW,
        sync_contract,
    )
    started, segment_epoch_s, carry_s = _start_timed_phase_segment(
        db_path=db_path, phase=phase
    )
    for round_idx in range(cfg.rounds):
        checkpoint_path = checkpoint_dir / f"controller_0_round_{round_idx}.pt"
        round_state = queue.conn.execute(
            """
            SELECT COUNT(*) n,
                   SUM(CASE WHEN status IN ('PENDING','RUNNING')
                            THEN 1 ELSE 0 END) incomplete
            FROM tasks WHERE phase=? AND round=?
            """,
            (phase, round_idx),
        ).fetchone()
        round_complete = (
            int(round_state["n"]) == cfg.search.m
            and int(round_state["incomplete"] or 0) == 0
        )
        if checkpoint_path.exists() and round_complete:
            controller.load(checkpoint_path)
            continue
        replayed_proposals = sum(
            queue.mark_controller_round_for_replay(
                phase=phase, round_idx=round_idx, shard=shard
            )
            for shard in range(cfg.graph.num_gpus)
        )
        if replayed_proposals:
            queue.add_event(
                phase,
                "CONTROLLER_ROUND_DETERMINISTIC_REPLAY",
                EvidenceKind.RAW,
                {
                    "round": round_idx,
                    "controller": 0,
                    "seed": controller.seed,
                    "historical_proposal_rows": replayed_proposals,
                    "policy": (
                        "reload prior checkpoint and replay identical RNG "
                        "stream; completed cell outputs are reused"
                    ),
                },
            )
        observations: list[tuple[Proposal, float]] = []
        accepted: dict[tuple[int, str], Proposal] = {}
        items = []
        for slot in range(cfg.search.m):
            shard = slot % cfg.graph.num_gpus
            proposal, rejected = _propose_launch(
                queue=queue,
                controller=controller,
                phase=phase,
                round_idx=round_idx,
                gpu=-1,
                shard=shard,
                static_gate=baseline_static_gate,
                floor_reward=cfg.search.floor_reward,
                max_resamples=cfg.search.max_resamples_per_slot,
            )
            observations.extend(rejected)
            fingerprint = proposal.architecture.fingerprint()
            accepted[(shard, fingerprint)] = proposal
            key = TaskKey(phase, round_idx, shard, fingerprint)
            spec = _cell_spec(
                key=key,
                mode=baseline_mode,
                shard=shard,
                architecture=proposal.architecture.arch_description,
                graph_path=(
                    sampled_graph_path
                    if baseline_mode == "proxy"
                    else graph_manifest["full_graph_path"]
                ),
                cfg=cfg,
                physical_gpu=physical_gpus[0],
            )



            spec["controller"] = 0
            items.append(
                (
                    key,
                    shard,
                    proposal.architecture.arch_description,
                    spec,
                )
            )
        results = _execute_dynamic_batches(
            db_path,
            profile_dir,
            physical_gpus,
            items,
            worker_pool=worker_pool,
            completion_callback=cost_gate_callback,
        )
        for result in results:
            proposal = accepted[(result.shard, result.fingerprint)]
            reward = (
                float(result.value)
                if result.status == Status.OK and result.value is not None
                else cfg.search.floor_reward
            )
            observations.append((proposal, reward))
        update = controller.observe(
            observations,
            launches_per_shard_round=cfg.search.m,
            flush=round_idx == cfg.rounds - 1,
        )
        queue.add_event(
            phase,
            "CONTROLLER_OBSERVE",
            EvidenceKind.MEASURED,
            {"round": round_idx, "controller": 0, **update},
        )
        controller.save(checkpoint_path)
    wall_s, active_segment_wall_s = _finish_timed_phase_segment(
        db_path=db_path,
        phase=phase,
        perf_start_s=started,
        epoch_start_s=segment_epoch_s,
        carry_s=carry_s,
    )
    cuda_boundary_assertion = _assert_cuda_phase_boundary(
        db_path=db_path, phase=phase, expected_tasks=cfg.search.q
    )
    runtime_mode_audit = _runtime_scheduler_mode_audit(
        db_path=db_path,
        phase=phase,
        expected_mode="dynamic_work_stealing",
        expected_tasks=cfg.search.q,
    )
    sync_contract["runtime_scheduler_mode_audit"] = runtime_mode_audit
    launch_rows = queue.conn.execute(
        "SELECT gpu, COUNT(*) n FROM tasks WHERE phase=? " "GROUP BY gpu ORDER BY gpu",
        (phase,),
    ).fetchall()
    launch_counts = {int(row["gpu"]): int(row["n"]) for row in launch_rows}
    launches = sum(launch_counts.values())
    if launches != cfg.search.q:
        queue.close()
        raise AssertionError(
            f"{phase}: trained launches {launches} != Q={cfg.search.q}"
        )
    sync_contract["per_gpu_launch_counts"] = launch_counts
    sync_contract["launch_counts_all_equal"] = len(set(launch_counts.values())) == 1
    sync_contract["launch_count_gate_retired"] = True
    queue.add_event(
        phase,
        "BASELINE_DYNAMIC_RUNTIME_ASSERTION_PASS",
        EvidenceKind.RAW,
        sync_contract,
    )
    queue.add_event(
        phase,
        "SEARCH_COMPLETE",
        EvidenceKind.MEASURED,
        {
            "wall_s": wall_s,
            "active_segment_wall_s": active_segment_wall_s,
            "recovered_active_wall_s": carry_s,
            "trained_launches": launches,
            "pool_mode": "independent",
            "sync_contract": sync_contract,
            "cuda_boundary_assertion": cuda_boundary_assertion,
        },
    )
    queue.close()
    return {
        "phase": phase,
        "wall_s": wall_s,
        "launches": launches,
        "scheduler": "barrier_free_baseline_dynamic_work_stealing",
        "sync_contract": sync_contract,
    }


def _shared_pool_path(profile_dir: Path) -> Path:
    return profile_dir / "shared_random_pool.json"


def _assert_shared_pool_reference(
    *,
    cfg: ExperimentConfig,
    entries: list[dict[str, Any]],
    db_path: Path,
) -> None:
    reference_value = cfg.runtime.shared_pool_reference
    if not reference_value:
        return
    reference_path = Path(reference_value).resolve()
    reference_payload = json.loads(reference_path.read_text())
    reference_entries = reference_payload["entries"]
    target = [entry["fingerprint"] for entry in entries]
    reference = [entry["fingerprint"] for entry in reference_entries]
    n_equal = sum(a == b for a, b in zip(target, reference))
    ordered_exact = target == reference and len(target) == cfg.search.q
    queue = TaskQueue(db_path)
    queue.add_event(
        f"{cfg.name}:shared:random:proposal",
        "SHARED_POOL_REFERENCE_AUDIT",
        EvidenceKind.RAW,
        {
            "reference_path": str(reference_path),
            "n_equal": n_equal,
            "denominator": cfg.search.q,
            "ordered_exact": ordered_exact,
        },
    )
    queue.close()
    print(
        f"PREFLIGHT shared_pool ordered_equal={n_equal}/{cfg.search.q} "
        f"exact={ordered_exact}",
        flush=True,
    )
    if not ordered_exact:
        raise AssertionError(
            f"{cfg.name}: shared pool ordered identity " f"{n_equal}/{cfg.search.q}"
        )


def _build_shared_random_pool(
    *,
    cfg: ExperimentConfig,
    profile_dir: Path,
    db_path: Path,
) -> list[dict[str, Any]]:
    pass
    path = _shared_pool_path(profile_dir)
    if path.exists():
        payload = json.loads(path.read_text())
        entries = payload["entries"]
        if (
            payload.get("pool_mode") != "shared_random"
            or int(payload.get("q", -1)) != cfg.search.q
            or len(entries) != cfg.search.q
        ):
            raise AssertionError(f"invalid persisted shared pool: {path}")
        for entry in entries:
            rebuilt = Architecture(entry["architecture"]).fingerprint()
            if rebuilt != entry["fingerprint"]:
                raise AssertionError(
                    f"shared pool fingerprint drift: {rebuilt} != "
                    f"{entry['fingerprint']}"
                )
        _assert_shared_pool_reference(cfg=cfg, entries=entries, db_path=db_path)
        return entries

    queue = TaskQueue(db_path)
    phase = f"{cfg.name}:shared:random:proposal"
    controller = ShardController(cfg.search, cfg.search.base_seed + 20_000)
    entries: list[dict[str, Any]] = []
    for launch_slot in range(cfg.search.q):
        round_idx = launch_slot // cfg.search.m
        slot_in_round = launch_slot % cfg.search.m
        shard = slot_in_round % cfg.graph.num_gpus
        for retry in range(cfg.search.max_resamples_per_slot + 1):
            proposal = controller.propose()
            fingerprint = proposal.architecture.fingerprint()
            status = controller.classify(
                proposal,
                static_gate=system_static_gate_enabled(cfg.graph.dataset),
            )
            if status == "ACCEPT":
                queue.add_proposal(
                    phase=phase,
                    round_idx=round_idx,
                    gpu=shard,
                    shard=shard,
                    fingerprint=fingerprint,
                    architecture=proposal.architecture.arch_description,
                    status="POOL_ADMITTED",
                    reward=None,
                    consumes_q=False,
                    retry_index=retry,
                )
                entries.append(
                    {
                        "launch_slot": launch_slot,
                        "round": round_idx,
                        "slot_in_round": slot_in_round,
                        "shard": shard,
                        "fingerprint": fingerprint,
                        "architecture": proposal.architecture.arch_description,
                    }
                )
                break
            queue.add_proposal(
                phase=phase,
                round_idx=round_idx,
                gpu=shard,
                shard=shard,
                fingerprint=fingerprint,
                architecture=proposal.architecture.arch_description,
                status=status,
                reward=cfg.search.floor_reward,
                consumes_q=False,
                retry_index=retry,
            )
        else:
            queue.add_event(
                phase,
                "RESAMPLE_EXHAUSTED",
                EvidenceKind.RAW,
                {
                    "round": round_idx,
                    "shard": shard,
                    "launch_slot": launch_slot,
                },
            )
            queue.close()
            raise RuntimeError(
                f"{phase}: exhausted resamples at launch slot {launch_slot}"
            )
    payload = {
        "pool_mode": "shared_random",
        "generator": "random",
        "base_seed": cfg.search.base_seed,
        "stream_seed": cfg.search.base_seed + 20_000,
        "static_gate": system_static_gate_enabled(cfg.graph.dataset),
        "static_gate_predicate": dataset_policy(
            cfg.graph.dataset
        ).static_gate_predicate,
        "q": cfg.search.q,
        "m": cfg.search.m,
        "n": cfg.graph.num_gpus,
        "entries": entries,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)
    queue.add_event(
        phase,
        "SHARED_POOL_COMPLETE",
        EvidenceKind.RAW,
        {
            "path": str(path),
            "trained_launches_per_side": len(entries),
            "free_static_or_dedup_rejections": sum(
                row["status"] in {"STATIC_REJECT", "DEDUP_REJECT"}
                for row in queue.rows("proposal_events")
                if row["phase"] == phase
            ),
        },
    )
    queue.close()
    _assert_shared_pool_reference(cfg=cfg, entries=entries, db_path=db_path)
    return entries


def _run_shared_search_side(
    *,
    cfg: ExperimentConfig,
    profile_dir: Path,
    db_path: Path,
    graph_manifest: dict[str, Any],
    side: str,
    shared_pool: list[dict[str, Any]],
    worker_pool: PersistentWorkerPool | None = None,
) -> dict[str, Any]:
    pass
    queue = TaskQueue(db_path)
    phase = f"{cfg.name}:{side}:{cfg.search.generator}:proxy"
    queue.set_state("current_phase", phase)
    is_system = side == "system"
    physical_gpus = cfg.runtime.system_gpus if is_system else cfg.runtime.baseline_gpus
    batches: dict[int, list[tuple[TaskKey, int, list[dict], dict[str, Any]]]] = {
        gpu: [] for gpu in physical_gpus
    }
    per_round_gpu_counts: dict[tuple[int, int], int] = {}
    for entry in shared_pool:
        round_idx = int(entry["round"])
        shard = int(entry["shard"])
        fingerprint = entry["fingerprint"]
        architecture = entry["architecture"]
        physical_gpu = physical_gpus[shard % len(physical_gpus)]
        queue.add_proposal(
            phase=phase,
            round_idx=round_idx,
            gpu=physical_gpu,
            shard=shard,
            fingerprint=fingerprint,
            architecture=architecture,
            status="ADMITTED",
            reward=None,
            consumes_q=False,
            retry_index=0,
        )
        key = TaskKey(phase, round_idx, shard, fingerprint)
        spec = _cell_spec(
            key=key,
            mode="proxy" if is_system else "full",
            shard=shard,
            architecture=architecture,
            graph_path=(
                graph_manifest["shards"][shard]["path"]
                if is_system
                else graph_manifest["full_graph_path"]
            ),
            cfg=cfg,
            physical_gpu=physical_gpu,
        )
        task_index = int(entry["launch_slot"])
        spec["dispatch_task_index"] = task_index
        spec["static_worker_id"] = physical_gpus[task_index % len(physical_gpus)]
        batches[physical_gpu].append((key, shard, architecture, spec))
        count_key = (round_idx, physical_gpu)
        per_round_gpu_counts[count_key] = per_round_gpu_counts.get(count_key, 0) + 1
    expected_per_gpu_round = cfg.search.m // len(physical_gpus)
    count_violations = [
        {
            "round": round_idx,
            "gpu": physical_gpu,
            "actual": per_round_gpu_counts.get((round_idx, physical_gpu), 0),
            "expected": expected_per_gpu_round,
        }
        for round_idx in range(cfg.rounds)
        for physical_gpu in physical_gpus
        if per_round_gpu_counts.get((round_idx, physical_gpu), 0)
        != expected_per_gpu_round
    ]
    if count_violations:
        queue.close()
        raise AssertionError(
            f"{phase}: invalid independent worker streams " f"{count_violations[:4]}"
        )
    if sum(len(batch) for batch in batches.values()) != cfg.search.q:
        queue.close()
        raise AssertionError(
            f"{phase}: shared pool does not contain Q={cfg.search.q} entries"
        )
    reachability_assertion = _assert_barrier_free_shared_search_route()
    sync_contract = {
        "route": "barrier_free_per_gpu_streams",
        "cross_gpu_sync_inside_search": False,
        "allowed_final_rendezvous": "final_join_before_nomination",
        "per_gpu_stream_lengths": {
            str(gpu): len(batch) for gpu, batch in batches.items()
        },
        "per_round_cells_per_gpu": expected_per_gpu_round,
        "static_reachability_assertion": reachability_assertion,
        "enforcement": (
            "all Q cells are assigned before worker launch; each physical-GPU "
            "thread executes its complete ordered stream serially; no shared "
            "work queue, condition, barrier, or per-round future wait is "
            "reachable; futures are joined only after every stream exits"
        ),
    }
    queue.add_event(
        phase,
        "BARRIER_FREE_SEARCH_ACTIVE",
        EvidenceKind.RAW,
        sync_contract,
    )
    queue.close()
    started = time.perf_counter()
    _execute_batches(
        db_path,
        profile_dir,
        batches,
        worker_pool=worker_pool,
    )
    wall_s = time.perf_counter() - started
    queue = TaskQueue(db_path)
    launches = int(
        queue.conn.execute(
            "SELECT COUNT(*) n FROM tasks WHERE phase=?", (phase,)
        ).fetchone()["n"]
    )
    if launches != cfg.search.q:
        queue.close()
        raise AssertionError(
            f"{phase}: trained launches {launches} != Q={cfg.search.q}"
        )
    queue.add_event(
        phase,
        "SEARCH_COMPLETE",
        EvidenceKind.MEASURED,
        {
            "wall_s": wall_s,
            "trained_launches": launches,
            "pool_mode": "shared_random",
            "sync_contract": sync_contract,
        },
    )
    queue.close()
    return {
        "phase": phase,
        "wall_s": wall_s,
        "launches": launches,
        "scheduler": "barrier_free_per_gpu_streams",
        "sync_contract": sync_contract,
    }


def _architecture_assignment_cost(
    architecture: list[dict[str, Any]],
) -> int:
    return sum(
        int(layer["num_heads"]) * int(layer["hidden_dim"]) for layer in architecture
    )


def _cost_balanced_semantic_pool(
    *,
    cfg: ExperimentConfig,
    shared_pool: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pass
    n = cfg.graph.num_gpus
    per_shard_round = cfg.search.m // n
    totals = [0] * n
    launch_counts = [0] * n
    d256_layers = [0] * n
    round_counts = [[0] * n for _ in range(cfg.rounds)]
    balanced_by_slot: dict[int, dict[str, Any]] = {}
    assignment_rows = []
    entries_by_round = [
        [entry for entry in shared_pool if int(entry["round"]) == round_idx]
        for round_idx in range(cfg.rounds)
    ]
    for round_idx, entries in enumerate(entries_by_round):
        if len(entries) != cfg.search.m:
            raise AssertionError(
                f"round {round_idx}: shared entries "
                f"{len(entries)} != M={cfg.search.m}"
            )
    for entry in sorted(
        shared_pool,
        key=lambda item: (
            -_architecture_assignment_cost(item["architecture"]),
            int(item["round"]),
            int(item["launch_slot"]),
        ),
    ):
        round_idx = int(entry["round"])
        candidates = [
            shard
            for shard in range(n)
            if round_counts[round_idx][shard] < per_shard_round
        ]
        shard = min(candidates, key=lambda item: (totals[item], item))
        cost = _architecture_assignment_cost(entry["architecture"])
        rewritten = dict(entry)
        rewritten["original_shard"] = int(entry["shard"])
        rewritten["shard"] = shard
        rewritten["predicted_assignment_cost"] = cost
        balanced_by_slot[int(entry["launch_slot"])] = rewritten
        totals[shard] += cost
        launch_counts[shard] += 1
        round_counts[round_idx][shard] += 1
        d256_layers[shard] += sum(
            int(layer["hidden_dim"]) == 256 for layer in entry["architecture"]
        )
        assignment_rows.append(
            {
                "launch_slot": int(entry["launch_slot"]),
                "round": round_idx,
                "fingerprint": entry["fingerprint"],
                "original_shard": int(entry["shard"]),
                "balanced_shard": shard,
                "predicted_assignment_cost": cost,
            }
        )
    for round_idx, counts in enumerate(round_counts):
        if counts != [per_shard_round] * n:
            raise AssertionError(f"round {round_idx}: cost-balance quota {counts}")
    balanced = [balanced_by_slot[int(entry["launch_slot"])] for entry in shared_pool]
    mean_cost = statistics.fmean(totals)
    plan = {
        "method": (
            "deterministic longest-predicted-cost first across the full pool; "
            "assign to the lowest global cumulative shard cost subject to "
            "M/N cells per shard per round"
        ),
        "cost_formula": "sum(layer.num_heads * layer.hidden_dim)",
        "semantic_change": True,
        "ordered_fingerprint_stream_preserved": [
            entry["fingerprint"] for entry in balanced
        ]
        == [entry["fingerprint"] for entry in shared_pool],
        "per_shard_launch_counts": launch_counts,
        "per_shard_predicted_cost": totals,
        "per_shard_d256_layers": d256_layers,
        "predicted_cost_mean": mean_cost,
        "predicted_cost_max": max(totals),
        "predicted_cost_max_over_mean": max(totals) / mean_cost,
        "assignments": assignment_rows,
    }
    if launch_counts != [cfg.search.q // n] * n:
        raise AssertionError(f"cost-balanced launch quotas failed: {launch_counts}")
    if not plan["ordered_fingerprint_stream_preserved"]:
        raise AssertionError("cost balancing changed ordered architecture pool")
    if float(plan["predicted_cost_max_over_mean"]) > 1.01:
        raise AssertionError(
            "cost-balanced predicted max/mean exceeds 1.01: "
            f"{plan['predicted_cost_max_over_mean']}"
        )
    return balanced, plan


def _run_cost_balanced_shared_system_search(
    *,
    cfg: ExperimentConfig,
    profile_dir: Path,
    db_path: Path,
    graph_manifest: dict[str, Any],
    side: str,
    shared_pool: list[dict[str, Any]],
    worker_pool: PersistentWorkerPool | None = None,
) -> dict[str, Any]:
    if side != "system":
        raise AssertionError("cost-balanced semantic search is system-only")
    phase = f"{cfg.name}:system:{cfg.search.generator}:proxy"
    physical_gpus = cfg.runtime.system_gpus
    if len(physical_gpus) != cfg.graph.num_gpus:
        raise AssertionError("cost-balanced validation requires native topology")
    balanced_pool, plan = _cost_balanced_semantic_pool(cfg=cfg, shared_pool=shared_pool)
    reachability_assertion = _assert_barrier_free_balanced_search_routes()
    plan["per_shard_capped_edges"] = [
        int(shard["E_msg_sub"]) for shard in graph_manifest["shards"]
    ]
    plan_path = profile_dir / "system_cost_balance_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True))
    batches: dict[int, list[tuple[TaskKey, int, list[dict], dict[str, Any]]]] = {
        gpu: [] for gpu in physical_gpus
    }
    queue = TaskQueue(db_path)
    queue.set_state("current_phase", phase)
    for entry in balanced_pool:
        shard = int(entry["shard"])
        physical_gpu = physical_gpus[shard]
        architecture = entry["architecture"]
        fingerprint = entry["fingerprint"]
        round_idx = int(entry["round"])
        queue.add_proposal(
            phase=phase,
            round_idx=round_idx,
            gpu=physical_gpu,
            shard=shard,
            fingerprint=fingerprint,
            architecture=architecture,
            status="COST_BALANCED_ADMITTED",
            reward=None,
            consumes_q=False,
            retry_index=0,
        )
        key = TaskKey(phase, round_idx, shard, fingerprint)
        spec = _cell_spec(
            key=key,
            mode="proxy",
            shard=shard,
            architecture=architecture,
            graph_path=graph_manifest["shards"][shard]["path"],
            cfg=cfg,
            physical_gpu=physical_gpu,
        )
        task_index = int(entry["launch_slot"])
        spec["dispatch_task_index"] = task_index
        spec["static_worker_id"] = physical_gpus[task_index % len(physical_gpus)]
        batches[physical_gpu].append((key, shard, architecture, spec))
    sync_contract = {
        "route": "barrier_free_cost_balanced_semantic_streams",
        "cross_gpu_round_barrier": False,
        "allowed_final_rendezvous": "final_join_before_nomination",
        "per_gpu_stream_lengths": {
            str(gpu): len(batch) for gpu, batch in batches.items()
        },
        "system_semantic_shard_binding_preserved": True,
        "system_architecture_assignment_semantically_changed": True,
        "predicted_cost_max_over_mean": (plan["predicted_cost_max_over_mean"]),
        "static_reachability_assertion": reachability_assertion,
    }
    queue.add_event(
        phase,
        "SYSTEM_COST_BALANCE_ACTIVE",
        EvidenceKind.RAW,
        {"plan_path": str(plan_path), "sync_contract": sync_contract},
    )
    queue.close()
    started = time.perf_counter()
    _execute_batches(
        db_path,
        profile_dir,
        batches,
        worker_pool=worker_pool,
        scheduler_mode="cost_balanced_assignment",
        side="system",
    )
    wall_s = time.perf_counter() - started
    runtime_mode_audit = _runtime_scheduler_mode_audit(
        db_path=db_path,
        phase=phase,
        expected_mode="cost_balanced_assignment",
        expected_tasks=cfg.search.q,
    )
    sync_contract["runtime_scheduler_mode_audit"] = runtime_mode_audit
    queue = TaskQueue(db_path)
    launches = int(
        queue.conn.execute(
            "SELECT COUNT(*) n FROM tasks WHERE phase=?", (phase,)
        ).fetchone()["n"]
    )
    if launches != cfg.search.q:
        queue.close()
        raise AssertionError(
            f"{phase}: trained launches {launches} != Q={cfg.search.q}"
        )
    queue.add_event(
        phase,
        "SEARCH_COMPLETE",
        EvidenceKind.MEASURED,
        {
            "wall_s": wall_s,
            "trained_launches": launches,
            "pool_mode": "shared_random",
            "sync_contract": sync_contract,
        },
    )
    queue.close()
    return {
        "phase": phase,
        "wall_s": wall_s,
        "launches": launches,
        "scheduler": "barrier_free_system_cost_balanced",
        "sync_contract": sync_contract,
    }


def _run_dynamic_shared_baseline_search(
    *,
    cfg: ExperimentConfig,
    profile_dir: Path,
    db_path: Path,
    graph_manifest: dict[str, Any],
    side: str,
    shared_pool: list[dict[str, Any]],
    worker_pool: PersistentWorkerPool | None = None,
) -> dict[str, Any]:
    if side != "baseline":
        raise AssertionError("dynamic shared search is baseline-only")
    phase = f"{cfg.name}:baseline:{cfg.search.generator}:proxy"
    physical_gpus = cfg.runtime.baseline_gpus
    queue = TaskQueue(db_path)
    queue.set_state("current_phase", phase)
    queue.close()
    items = []
    for entry in shared_pool:
        shard = int(entry["shard"])
        architecture = entry["architecture"]
        fingerprint = entry["fingerprint"]
        key = TaskKey(phase, int(entry["round"]), shard, fingerprint)
        spec = _cell_spec(
            key=key,
            mode="full",
            shard=shard,
            architecture=architecture,
            graph_path=graph_manifest["full_graph_path"],
            cfg=cfg,
            physical_gpu=physical_gpus[0],
        )
        items.append((key, shard, architecture, spec))
    sync_contract = {
        "route": "barrier_free_baseline_dynamic_work_stealing",
        "cross_gpu_round_barrier": False,
        "work_queue_scope": "all_Q_semantically_interchangeable_cells",
        "allowed_final_rendezvous": "final_join_before_nomination",
        "static_reachability_assertion": (
            _assert_barrier_free_balanced_search_routes()
        ),
    }
    queue = TaskQueue(db_path)
    queue.add_event(
        phase,
        "BASELINE_DYNAMIC_WORK_STEALING_ACTIVE",
        EvidenceKind.RAW,
        sync_contract,
    )
    queue.close()
    started = time.perf_counter()
    _execute_dynamic_batches(
        db_path,
        profile_dir,
        physical_gpus,
        items,
        worker_pool=worker_pool,
    )
    wall_s = time.perf_counter() - started
    queue = TaskQueue(db_path)
    launch_rows = queue.conn.execute(
        "SELECT gpu, COUNT(*) n FROM tasks WHERE phase=? " "GROUP BY gpu ORDER BY gpu",
        (phase,),
    ).fetchall()
    launch_counts = {int(row["gpu"]): int(row["n"]) for row in launch_rows}
    launches = sum(launch_counts.values())
    runtime_mode_audit = _runtime_scheduler_mode_audit(
        db_path=db_path,
        phase=phase,
        expected_mode="dynamic_work_stealing",
        expected_tasks=cfg.search.q,
    )
    if launches != cfg.search.q:
        queue.close()
        raise AssertionError(
            f"{phase}: trained launches {launches} != Q={cfg.search.q}"
        )
    sync_contract["per_gpu_launch_counts"] = launch_counts
    sync_contract["launch_counts_all_equal"] = len(set(launch_counts.values())) == 1
    sync_contract["launch_count_gate_retired"] = True
    sync_contract["runtime_scheduler_mode_audit"] = runtime_mode_audit
    queue.add_event(
        phase,
        "BASELINE_DYNAMIC_RUNTIME_ASSERTION_PASS",
        EvidenceKind.RAW,
        sync_contract,
    )
    queue.add_event(
        phase,
        "SEARCH_COMPLETE",
        EvidenceKind.MEASURED,
        {
            "wall_s": wall_s,
            "trained_launches": launches,
            "pool_mode": "shared_random",
            "sync_contract": sync_contract,
        },
    )
    queue.close()
    return {
        "phase": phase,
        "wall_s": wall_s,
        "launches": launches,
        "scheduler": "barrier_free_baseline_dynamic_work_stealing",
        "sync_contract": sync_contract,
    }
