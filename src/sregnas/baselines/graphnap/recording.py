

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from sregnas.icde27 import recorder as shared
from sregnas.paths import PROJECT_ROOT as ROOT
from sregnas.training.minibatch import _hash_seed

BUDGET = ROOT / "results/icde27/gpu_budget_p0b.json"
CAMPAIGN_DEADLINE_ENV = "SREGNAS_ICDE27_CAMPAIGN_DEADLINE_EPOCH"



snapshot = shared.snapshot


def acquire_lock(lock, deadline=None):
    pass
    if deadline is None:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return True
    while time.time() < deadline:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            time.sleep(min(.5, max(0., deadline - time.time())))
    return False


def record_unstarted_budget_end(out, args, cfg, command, waited):
    pass
    out.mkdir(parents=True)
    info = shared.metadata(cfg, args.method, out, args.experiment_id, args.record_type, command)
    info.update(mode=args.mode, physical_gpus=cfg.runtime.baseline_gpus,
                P=len(cfg.runtime.baseline_gpus), K=args.shortlist_size or cfg.fixed_x,
                K_actual=0, status="incomplete", stop_reason="campaign_budget_exhausted",
                blocking_reason="campaign deadline expired while waiting for GPU flock",
                pipeline_started=False, process_wall_s=0., reserved_gpu_hours=0.,
                lock_wait_s=waited, timing_valid=False, returncode=None,
                assertions=dict(campaign_budget_available=False))
    shared.dump(out / "run.json", info)
    (out / "raw.log").write_text(info["blocking_reason"] + "; no pipeline or GPU allocation started.\n")
    for name in ("candidates", "shortlist", "retrain"):
        shared.write_table(out / f"{name}.tsv", [], ["run_id", "fingerprint", "status"])
    (out / "summary.md").write_text(
        f"run_id: {out.name}\n\nrecord_type: {args.record_type}\n\nstatus: incomplete\n\n"
        "ASSERTION_PASS: false\n\n- campaign_budget_available: false\n")
    return 2


def finalize(out, info):
    rows_path = out / "pipeline/port_candidates.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    events = shared.read_rows(out / "events.jsonl")
    launches = shared.read_rows(out / "launch_ledger.jsonl")
    source = out/"pipeline/source_selection.json"
    budget_path = out/"pipeline/search_budget.json"
    expected = None
    if source.exists():
        selection = json.loads(source.read_text())
        expected = 0
        info.update(measurement_scope="post_search_only", search_expected_count=0,
                    search_pool_count=len(selection["selected"]), reused_search_candidates=len(selection["selected"]),
                    source_search_run_id=selection["source_run_id"], Q_trunc=len(selection["selected"]))
        info["assertions"]["source_selection_count"] = len(selection["selected"]) == info["search_limit"]
        info["assertions"]["source_graph_split_space_match"] = all(selection["source_"+key] == info[key]
                                                                     for key in ("graph_hash","split_hash","space_hash"))
    elif budget_path.exists():
        bound = json.loads(budget_path.read_text())
        expected = bound["actual_attempts"]
        count_ok = bound["max_search_candidates"] is None or bound["completed"] <= bound["max_search_candidates"]
        reason_ok = ((bound["stop_reason"] == "max_search_candidates" and bound["completed"] == bound["max_search_candidates"])
                     or (bound["stop_reason"] == "max_search_seconds" and
                         bound["search_return_perf_s"]-bound["search_start_perf_s"] >= bound["max_search_seconds"]) or expected == info["Q_requested"])
        info["assertions"]["explicit_search_bound"] = count_ok and reason_ok and bound["completed"] == expected
        info.update(measurement_scope="bounded_pipeline", search_expected_count=expected, search_pool_count=expected,
                    Q_trunc=expected, search_stop_reason=bound["stop_reason"], search_budget=bound)
    if info.get("paspl_full_nas") and info.get("stage") == "search":
        from sregnas.icde27.x15.observations import finalize_search_only
        finalize_search_only(out, info, rows, launches)
    elif info["mode"] == "full_nas":
        shared.finalize_record(out, info, expected_search_count=expected, post_search_only=source.exists())
    else:
        attempts = [r for r in launches if r["kind"] in {"candidate", "retry", "resume"}]
        info.update(Qbench=len(attempts), Q_search=0, Q_attempt=len(attempts),
                    Q_probe=sum(r["kind"] == "probe" for r in launches))
        info["assertions"]["fixed_pool_attempt_count"] = len(attempts) == info["Q_requested"]
        info["assertions"]["fixed_pool_complete"] = len(rows) == info["Q_requested"] and all(r["status"] == "OK" for r in rows)
        try:
            start = next(e["perf_s"] for e in events if e["event"] == "e2e_start")
            end = max(e["perf_s"] for e in events if e["event"] == "e2e_end")
            ss = min(r["perf_s"] for r in attempts)
            se = max(e["perf_s"] for e in events if e["event"] == "cell_end")
            bounds = [("prep", start, ss), ("search", ss, se),
                      ("nomination+probe+refill", se, se), ("retrain+validation", se, se),
                      ("winner-test", se, se), ("other", se, end)]
            info["phases"] = [dict(phase=p, start_perf_s=a, end_perf_s=b, wall_s=b-a,
                                    physical_gpus=info["physical_gpus"]) for p,a,b in bounds]
            info["cold_e2e_wall_s"] = end-start
            info["phase_sum_s"] = sum(p["wall_s"] for p in info["phases"])
            info["direct_e2e_wall_s"] = next(e["wall_s"] for e in events if e["event"] == "direct_e2e_wall")
            info["phase_wall_discrepancy_fraction"] = abs(info["phase_sum_s"]-info["direct_e2e_wall_s"])/info["direct_e2e_wall_s"]
            info["assertions"]["six_phase_wall_agreement"] = info["phase_wall_discrepancy_fraction"] <= .02
        except (ValueError, StopIteration):
            info["assertions"]["six_phase_wall_agreement"] = False
        shared.write_table(out / "launch_ledger.tsv", launches,
                           ["launch_id", "kind", "stage", "identity", "fingerprint", "backend", "perf_s"])
    good = [r for r in rows if r["status"] == "OK"]
    info["assertions"]["proxy_50_steps"] = bool(good) and all(r["optimizer_steps"] == 50 for r in good)
    info["completed_candidates"] = len(good)
    if info.get("paspl_full_nas"):
        info.update(Q_success=len(good), search_feature_residency="whole_graph_preload")
    if info["method"] == "GraphNAP":
        info["parameter_average_per_candidate"] = [
            {k: r.get(k) for k in ("fingerprint", "status", "parameter_average_count", "parameter_average_bytes")}
            for r in rows]
        info["parameter_average_bytes_definition"] = "sum input state tensor bytes, all replicas, all means; not network wire bytes"
        rng_events = (shared.read_rows(Path(selection["source_dir"])/"events.jsonl") if source.exists() else events)
        selected_fps = {row["fingerprint"] for row in rows}
        resets = [e for e in rng_events if e["event"] == "graphnap_rng" and e["fingerprint"] in selected_fps]
        info["assertions"]["graphnap_rank_initialization_equal"] = bool(rows) and all(
            len({e["partition_id"] for e in resets if e["fingerprint"] == row["fingerprint"]}) == info["P"]
            and len({e["initialization_hash"] for e in resets if e["fingerprint"] == row["fingerprint"]}) == 1
            for row in rows)
        info["assertions"]["graphnap_partition_rng"] = bool(resets) and all(
            e["initialization_seed"] == _hash_seed(*e["init_seed_inputs"])
            and e["training_seed"] == _hash_seed(e["fingerprint"], "search", 0, e["partition_id"])
            for e in resets)
        info["assertions"]["graphnap_50_parameter_means"] = bool(good) and all(
            r["parameter_average_count"] == 50 and r["parameter_average_bytes"] > 0 for r in good)
    if info["method"] == "AP":
        info["assertions"]["ap_search_features_cpu"] = bool(good) and all(
            r["search_features_on_gpu"] is False and r["feature_device"] == "cpu" for r in good)
        retrain = out / "pipeline/retrained.json"
        trained = json.loads(retrain.read_text()) if retrain.exists() else []
        info["feature_residency_observed"] = dict(
            search=sorted({r["feature_device"] for r in good}),
            retrain=sorted({r["metrics"]["cell_start"]["feature_residency"] for r in trained}),
            retrain_inference=sorted({r["metrics"]["cell_start"]["inference_residency"] for r in trained}))
        info["assertions"]["ap_retrain_residency"] = bool(trained) and all(
            r["metrics"]["cell_start"]["feature_residency"] == "gpu"
            and r["metrics"]["cell_start"]["inference_residency"] == "gpu" for r in trained)
    failure = out / "pipeline/port_failure.json"
    if failure.exists():
        info["failure"] = json.loads(failure.read_text())
        if info["status"] != "timeout" and not info.get("stop_reason"):
            info["status"] = info["failure"]["status"]
    required = "run_id experiment_id method dataset environment backend commit config_hash graph_hash split_hash space_hash P S resource_class seed_rule status record_type timing_valid".split()
    info["assertions"]["required_run_fields"] = all(info.get(k) is not None for k in required)
    if info["status"] == "OK" and not all(info["assertions"].values()):
        info["status"] = "incomplete"
    shared.write_table(out / "candidates.tsv", [dict(run_id=out.name, **r) for r in rows],
        ["run_id", "attempt", "fingerprint", "status", "value", "wall_s", "optimizer_steps",
         "optimizer_steps_total", "peak_allocated_bytes", "peak_reserved_bytes", "search_features_on_gpu", "feature_device",
         "parameter_average_count", "parameter_average_bytes", "parameter_average_bytes_definition", "failure_stage", "error"])
    for name in ("shortlist", "retrain"):
        if not (out / f"{name}.tsv").exists():
            shared.write_table(out / f"{name}.tsv", [], ["run_id", "fingerprint", "status"])


def terminate_tree(child):
    from sregnas.execution.telemetry import _owned
    pids = [int(p) for p in subprocess.check_output(["ps", "-eo", "pid="], text=True).split()]
    owned = [p for p in pids if p != os.getpid() and _owned(p, child.pid)]
    for pid in [*owned, child.pid]:
        try: os.kill(pid, signal.SIGTERM)
        except ProcessLookupError: pass
    try: child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill(); child.wait()
    for pid in owned:
        try: os.kill(pid, signal.SIGKILL)
        except ProcessLookupError: pass


def record_comparison(args, cfg, original_argv):
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    argv = list(sys.argv[1:] if original_argv is None else original_argv)
    argv.remove("--icde27-record")
    argv[argv.index("--out")+1] = str(out / "pipeline")
    command = [sys.executable, "-m", "sregnas.comparison", *argv]
    gpus = cfg.runtime.baseline_gpus
    quality_only = getattr(args, "quality_only", False)
    gpu_lock = getattr(args, "gpu_lock", None) or shared.LOCK

    print(f"{args.experiment_id} waiting for flock {shared.LOCK}", flush=True)
    waiting_started = time.perf_counter()
    deadline = os.environ.get(CAMPAIGN_DEADLINE_ENV)
    from sregnas.icde27.x16.locking import record_lock
    with record_lock(gpu_lock, enabled=getattr(args, "x16_paspl_fanout5", False)) as lock:
        if not acquire_lock(lock, float(deadline) if deadline is not None else None):
            return record_unstarted_budget_end(out, args, cfg, command, time.perf_counter() - waiting_started)
        print(f"{args.experiment_id} acquired flock: {out.name}", flush=True)
        reserved_start = time.perf_counter()


        p0b = args.experiment_id == "P0B"
        budget_path = BUDGET if p0b else ROOT / f"results/icde27/gpu_budget_{args.experiment_id.lower()}_baselines{'_gpu'+'_'.join(map(str,gpus)) if quality_only else ''}.json"
        if getattr(args,'x18_experiment',False):
            budget_path=out.parent/'x18_gpu_budget_baselines.json'
        budget = json.loads(budget_path.read_text()) if budget_path.exists() else dict(used_gpu_s=0, runs=[])
        timeout = args.max_wall_seconds
        if p0b:
            budget.setdefault("limit_gpu_s", 21600)
            remaining = (budget["limit_gpu_s"]-budget["used_gpu_s"])/len(gpus) - 20
            if remaining <= 0:
                raise RuntimeError("P0-B 6 GPU-hour budget exhausted")
            timeout = min(timeout, remaining)
        out.mkdir(parents=True)
        info = shared.metadata(cfg, args.method, out, args.experiment_id, args.record_type, command)
        if getattr(args, 'cached_sampler', False):
            info.update(cached_sampler=True, sampler_cache_scope='one CSC per worker, shared by train and validation')
        info.update(timing_scope="quality_only" if quality_only else "exclusive_primary", gpu_lock=str(gpu_lock))
        info.update(mode=args.mode, physical_gpus=gpus, P=len(gpus), K=args.shortlist_size or cfg.fixed_x,
                    max_wall_seconds=timeout, lock_wait_s=time.perf_counter() - waiting_started,
                    rebase_commit="12184cb260348a4a4c2fc4f79febdbcbf3f212cb",
                    seed_rule=dict(initialization="H(fingerprint,search,0)",
                                   controller=f'CentralController(base_seed={cfg.search.base_seed}+10000)',
                                   graphnap_training="H(fingerprint,search,0,partition_id)",
                                   ap_paspl_training="initialization RNG continues", retrain="H(fingerprint,retrain,0)"))
        if getattr(args, "paspl_full_nas", False):
            info.update(paspl_full_nas=True, stage=args.stage,
                        search_feature_residency="whole_graph_preload")
        if getattr(args, "x16_paspl_fanout5", False):
            info.update(x16_paspl_fanout5=True, proxy_fanout=[5, 5])
            budget_path = out.parent / "x15_gpu_budget.json"
            budget = json.loads(budget_path.read_text()) if budget_path.exists() else dict(used_gpu_s=0, runs=[])
        if args.mode == "fixed_pool":
            from .search import load_pool
            info.update(Q_requested=len(load_pool(args.pool, args.pool_limit)), K=0,
                        pool_hash=shared.sha(args.pool), capacity_probe=args.capacity_probe)
        info.update(from_search_record=str(getattr(args, "from_search_record", None).resolve()) if getattr(args, "from_search_record", None) else None,
                    search_limit=getattr(args, "search_limit", None), max_search_candidates=getattr(args, "max_search_candidates", None),
                    max_search_seconds=getattr(args, "max_search_seconds", None),
                    collect_search_garbage=bool(getattr(args, "collect_search_garbage", False) or
                                               getattr(args, "paspl_collect_search_garbage", False)))
        known_owned = {}
        def observe():
            observation = snapshot(gpus, os.getpid(), known_owned)
            if getattr(args, "paspl_full_nas", False) or getattr(args, 'cached_sampler', False):
                from sregnas.icde27.x15.observations import augment_snapshot
                augment_snapshot(observation, os.getpid(), gpus)
            if getattr(args, "max_search_candidates", None) is not None or getattr(args, "from_search_record", None):
                from .truncation import host_rss_bytes
                observation["owned_process_rss_bytes"] = {str(p["pid"]):host_rss_bytes(p["pid"])
                                                          for p in observation["processes"] if p["owned"]}
            for row in observation["ownership_events"]:
                with (out / "ownership_events.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(row, epoch_s=observation["epoch_s"],
                                                 perf_s=observation["perf_s"])) + "\n")
            return observation
        before = observe()
        shared.dump(out / "resources_before.json", before)
        shared.dump(out / "run.json", info)
        if before["foreign_processes"] or (before["cpu_contention_observed"] and not quality_only):
            info.update(status="contaminated", blocking_reason="contention after acquiring flock")
            shared.dump(out / "run.json", info)
            return 2
        env = dict(os.environ, SREGNAS_ICDE27_RECORD_DIR=str(out), SREGNAS_ALLOW_GPU="1",
                   SREGNAS_APPROVED_GPUS=",".join(map(str, gpus)), PYTHONPATH=str(ROOT / "src"),
                   OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", PYTHONUNBUFFERED="1",
                   SREGNAS_ICDE27_TIE_RULE="1")
        if os.environ.get('SREGNAS_CPU_THREADS'):
            env.update({key:os.environ['SREGNAS_CPU_THREADS'] for key in
                        ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
        env.pop("CUDA_VISIBLE_DEVICES", None)
        observations = []
        hit_timeout = False
        started = time.perf_counter()
        with (out / "raw.log").open("w") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            from sregnas.execution.telemetry import _start_ticks
            info["pipeline_started_perf_s"] = started
            info["pipeline_process"] = dict(pid=child.pid, pgid=child.pid,
                start_ticks=_start_ticks(child.pid), recorder_pid=os.getpid(), recorder_pgid=os.getpgrp())
            shared.dump(out / "run.json", info)
            try:
                while child.poll() is None:
                    observation = observe()
                    observations.append(observation)
                    with (out / "resources_during.jsonl").open("a") as stream:
                        stream.write(json.dumps(observation)+"\n")
                    if getattr(args, "paspl_full_nas", False):
                        from sregnas.icde27.x15.observations import terminal_actor_failure
                        candidate_file = out / "pipeline/port_candidates.json"
                        try:
                            candidate_rows = json.loads(candidate_file.read_text())
                        except (FileNotFoundError, json.JSONDecodeError):
                            candidate_rows = []
                        terminal_failure = terminal_actor_failure(candidate_rows)
                        if terminal_failure:
                            shared.dump(out / "pipeline/port_failure.json", terminal_failure)
                            shared.dump(out / "method_failure_decision.json", dict(
                                terminal_failure, run_id=out.name, pipeline_process=info["pipeline_process"],
                                termination_requested=True, observed_epoch_s=time.time()))
                            terminate_tree(child)
                            break
                    if getattr(args, "paspl_full_nas", False) and observation["foreign_processes"]:

                        terminate_tree(child)
                        break
                    if shared.recording_limit_reached(out,started,timeout):
                        hit_timeout = True
                        terminate_tree(child)
                        break
                    time.sleep(.5)
            finally:
                if child.poll() is None:
                    terminate_tree(child)



        from sregnas.execution.telemetry import gpu_idle_preflight_settled
        settled = gpu_idle_preflight_settled(gpus, owner_pid=os.getpid(),
                    settle_delay_s=2, poll_interval_s=.5, timeout_s=8)
        shared.dump(out / "resources_settled.json", settled)
        after = observe()
        shared.dump(out / "resources_after.json", after)
        wall = time.perf_counter()-reserved_start
        charged = wall*len(gpus)
        budget["used_gpu_s"] += charged
        budget["runs"].append(dict(run_id=out.name, P=len(gpus), wall_s=wall, gpu_s=charged))
        budget_path.parent.mkdir(parents=True, exist_ok=True)
        shared.dump(budget_path, budget)
        info.update(process_wall_s=time.perf_counter()-started, reserved_gpu_hours=charged/3600,
                    gpu_budget_used_hours=budget["used_gpu_s"]/3600, returncode=child.returncode,
                    timing_valid=not any(o["foreign_processes"] or (o["cpu_contention_observed"] and not quality_only) for o in [before,*observations,after]),
                    status="timeout" if hit_timeout else "OK" if child.returncode == 0 else "incomplete")
        preflight_path = out / "gpu_idle_preflight.json"
        if preflight_path.exists():
            preflight = json.loads(preflight_path.read_text())
            info.update(gpu_idle_wait_seconds=preflight["wait_seconds"],
                        gpu_idle_preflight_attempts=preflight["attempts"], gpu_idle_preflight_passed=preflight["idle"])
        decision_path = out / "watchdog_decision.json"
        if decision_path.exists():
            decision = json.loads(decision_path.read_text())
            if (decision.get("decision") == "terminate" and decision.get("termination_requested")
                and decision.get("run_id") == info["run_id"]
                and decision.get("pipeline_process") == info["pipeline_process"] and child.returncode != 0):
                info.update(status="incomplete", stop_reason="projected_over_threshold")
        finalize(out, info)
        if getattr(args, 'cached_sampler', False) and not getattr(args, 'paspl_full_nas', False):
            from sregnas.icde27.x15.observations import finalize_resources
            finalize_resources(out, info, [before, *observations, after])
        if getattr(args, "paspl_full_nas", False):
            from sregnas.icde27.x15.observations import finalize_resources
            finalize_resources(out, info, [before, *observations, after])
            info["external_gpu_contention"] = any(o["foreign_processes"] for o in [before, *observations, after])
            if info["external_gpu_contention"]:
                info.update(status="contaminated", timing_valid=False)
        info["assertions"]["gpu_released_before_unlock"] = settled["idle"]
        if not settled["idle"] and info["status"] == "OK":
            info["status"] = "incomplete"
        if info["status"] == "OK" and not info["timing_valid"]:
            info["status"] = "contaminated"
        if quality_only:
            if info.get('Q_attempt',0)<info.get('Q_requested',0):
                info.update(Q_trunc=info['Q_attempt'],quality_completeness='truncated_Q400')
            if info['status']=='timeout': info.update(status='incomplete',stop_reason='over_twice_estimate')
            info.update(timing_valid=False, resource_class="quality_gpu_group")
        if any(os.environ.get(k)=='1' for k in ('SREGNAS_X22_INCREMENTAL_AUDIT','SREGNAS_X22_DEFER_RECORDS')):
            from sregnas.icde27.x22.checks import assertions as framework_assertions
            info['assertions'].update(framework_assertions(out))
            if info['status']=='OK' and not all(info['assertions'].values()):info['status']='incomplete'
        shared.dump(out / "run.json", info)
        passed = info["status"] == "OK" and all(info["assertions"].values())
        (out / "summary.md").write_text(f"run_id: {out.name}\n\nrecord_type: {info['record_type']}\n\nstatus: {info['status']}\n\nASSERTION_PASS: {str(passed).lower()}\n\n" +
            "\n".join(f"- {k}: {str(v).lower()}" for k,v in info["assertions"].items()) + "\n")
        print(json.dumps(dict(run_id=out.name, status=info["status"], assertion_pass=passed,
                              gpu_budget_used_hours=budget["used_gpu_s"]/3600)), flush=True)
        return 0 if passed else 2
