
from sregnas.paths import PROJECT_ROOT as _artifact_root
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from sregnas.paths import PROJECT_ROOT as ROOT
from sregnas.icde27 import recorder as shared
from .plan import DATASETS, EXECUTORS, selected_rows, logical_shards, candidate_mapping, execution_key, digest, apply_amendments
from .resources import mark_stage, stage_at_failure, rss_tree, is_oom, failure_evidence, pool_estimate
from .exp3_hooks import requested as fine_profile_requested


def configuration(cell):
    from sregnas.config import load_config
    cfg = load_config(ROOT / "configs/icde27" / (DATASETS[cell["dataset"]]+"_q400_n8.yaml"))
    p = cell["P"]
    s = logical_shards(cell) or p
    cfg.graph.num_gpus = s
    cfg.runtime.system_gpus = cfg.runtime.baseline_gpus = list(range(p))
    cfg.runtime.require_native_topology = p == s
    cfg.runtime.topology_mode = "native" if p == s else f"oversubscribed_{s//p}x"
    cfg.runtime.require_fi_gpu = cell["kind"] == "replay" and cell["method"] == "SREGNAS" and cell["domain"] == "local"
    for key, value in zip(("sampler_impl", "feature_residency", "inference_residency"), EXECUTORS[cell["executor"]]):
        setattr(cfg.runtime, key, value)
    cfg.fixed_x = s * cfg.selection.k
    cfg.validate()
    return cfg


def terminate(child):
    pass
    from sregnas.execution.telemetry import _owned, _start_ticks
    pids = [int(p) for p in subprocess.check_output(["ps", "-eo", "pid="], text=True).split()]
    identities = {p: _start_ticks(p) for p in pids if p != os.getpid() and _owned(p, child.pid)}
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid, ticks in identities.items():
            if ticks is not None and _start_ticks(pid) == ticks:
                try: os.kill(pid, sig)
                except ProcessLookupError: pass
        if child.poll() is None:
            try: os.killpg(child.pid, sig)
            except ProcessLookupError: pass
        if sig == signal.SIGTERM:
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired: pass
    child.wait(timeout=10)


def final_record(out, info, cell):
    events = shared.read_rows(out / "events.jsonl")
    ledger = shared.read_rows(out / "launch_ledger.jsonl")
    result = json.loads((out / "measurement.json").read_text()) if (out / "measurement.json").exists() else {}
    if not result:
        for name in ("partial_candidates.json", "port_candidates.json"):
            if (out/name).exists(): result["rows"] = json.loads((out/name).read_text()); break
        if "rows" not in result:
            result["rows"] = sorted([r for path in out.glob("worker_[0-9]*.json") if not path.name.endswith(".job.json")
                for r in json.loads(path.read_text()).get("rows", [])], key=lambda r: r.get("index", 0))
    info.update(result)
    attempts = [r for r in ledger if r["kind"] in ("candidate", "retry", "resume")]
    info.update(Q_search=0, Q_trunc=None, Qbench=len(attempts) if cell["kind"] == "replay" else None,
        Q_attempt=len(attempts), warmup_launches=sum(r["kind"] == "warmup" for r in ledger),
        probe_launches=sum(r["kind"] == "probe" for r in ledger))
    def event_time(name):
        return next((r["perf_s"] for r in events if r["event"] == name), None)
    start, pool_start, end = (event_time(k) for k in ("e2e_start", "pool_start", "pool_end"))
    if all(v is not None for v in (start, pool_start, end)):
        info.update(prep_s=pool_start-start, pool_s=end-pool_start, e2e_s=end-start,
                    reserved_gpu_hours=cell["P"]*(end-start)/3600)
        info["phases"] = [dict(phase=name, wall_s=b-a, start_perf_s=a, end_perf_s=b, P=cell["P"], gpus=list(range(cell["P"])))
            for name, a, b in (("prep", start, pool_start), ("pool", pool_start, end))]
    rows = result.get("rows", [])
    info["completed_candidates"] = sum(r.get("status") == "OK" for r in rows)
    assertions = dict(candidate_list_exact=[r.get("fingerprint") for r in rows] ==
        [r["fingerprint"] for r in selected_rows(cell)],
        all_cells_complete=len(rows) == cell["count"] and all(r.get("status") == "OK" for r in rows),
        two_phase_boundary=all(v is not None for v in (start, pool_start, end)),
        no_RL_updates=result.get("controller_updates") == 0)
    if cell["kind"] == "replay": assertions["launch_count"] = len(attempts) == cell["count"]
    if cell["method"] == "FastGNAS" and cell["kind"] == "replay":
        all_segments=shared.read_rows(out/"native_segments.jsonl")
        segments = [r for r in all_segments if r["event"] == "segment_end" and r["phase"] == "search"]
        assertions["native_N_segments"] = all(sum(s["parent_candidate"] == a["launch_id"] for s in segments)==logical_shards(cell) for a in attempts) and bool(attempts)
        assertions["native_segment_steps"] = bool(segments) and all(s["optimizer_steps"] == s["expected_optimizer_steps"] for s in segments)
        from .validation import native_rng_assertions
        checks, checked=native_rng_assertions(all_segments,events)
        assertions.update(checks); shared.dump(out/"rng_segment_assertions.json",checked)
        warmup=[s for s in all_segments if s["event"]=="segment_end" and s["phase"]=="warmup"]
        assertions["warmup_in_prep"] = bool(warmup) and pool_start is not None and all(s["perf_s"]<pool_start for s in warmup)
        shared.write_table(out/"ledger_segments.tsv", segments, sorted(set().union(*(r.keys() for r in segments))) if segments else ["segment_id"])
    assertions.update(result.get("assertions", {}))
    if any(os.environ.get(k)=='1' for k in ('SREGNAS_X22_INCREMENTAL_AUDIT','SREGNAS_X22_DEFER_RECORDS')):
        from sregnas.icde27.x22.checks import assertions as framework_assertions
        assertions.update(framework_assertions(out))
    if "gpu_released_before_unlock" in info:
        assertions["gpu_released_before_unlock"] = info["gpu_released_before_unlock"]
    info["assertions"] = assertions
    if info["status"] == "OK" and not all(assertions.values()): info["status"] = "incomplete"
    if info["status"] == "OK" and not info["timing_valid"] and info.get("timing_scope") != "quality_only": info["status"] = "contaminated"
    failed = [r.get("status") for r in rows if r.get("status") != "OK"]
    if info["status"] == "incomplete" and failed and all(s == "OOM" for s in failed): info["status"] = "OOM"
    failure = out / "failure.json"
    if failure.exists():
        info["failure"] = json.loads(failure.read_text())
        if info["status"] not in ("timeout", "contaminated"): info["status"] = info["failure"]["status"]
    failure_evidence(out, rows, info)
    pool_estimate(cell, info, rows)
    if info.get("timing_scope") == "quality_only":
        info["timing_valid"] = False
        if info["status"] == "timeout": info.update(status="incomplete",stop_reason="over_twice_estimate")
    info["checkpoint_paths"] = [r.get("checkpoint_path") or r.get("metrics", {}).get("checkpoint_path") for r in rows]
    shared.write_table(out/"launch_ledger.tsv", ledger, sorted(set().union(*(r.keys() for r in ledger))) if ledger else ["launch_id"])
    fields = ["run_id", "fingerprint", "status", "value", "wall_s", "worker", "gpu", "metrics", "checkpoint_path"]
    for name in ("candidates", "shortlist", "retrain"):
        selected = rows if name == ("candidates" if cell["kind"] == "replay" else "retrain") else []
        shared.write_table(out/(name+".tsv"), [dict(run_id=out.name, **r) for r in selected], fields)
    if fine_profile_requested(cell):
        from .exp3_summary import publish
        try:
            publish(out, info, cell)
        except Exception as exc:
            traceback.print_exc()
            assertions["exp3_component_accounting"] = False
            info["exp3_profile_error"] = repr(exc)
            if info["status"] == "OK": info["status"] = "incomplete"
    passed = info["status"] == "OK" and all(assertions.values())
    shared.dump(out/"run.json", info)
    (out/"summary.md").write_text(f"run_id: {out.name}\n\nrecord_type: {info['record_type']}\n\nstatus: {info['status']}\n\nASSERTION_PASS: {str(passed).lower()}\n\n"+
        "\n".join(f"- {k}: {str(v).lower()}" for k,v in assertions.items())+"\n")
    return 0 if passed else 2


def record(cell, out, timeout, lock_nonblocking=False, *, gpu_lock=None, quality_only=False):
    cell = apply_amendments(cell)
    monitor_gpus = cell.get('exclusive_gpus', list(range(cell['P'])))
    assert set(range(cell['P'])).issubset(monitor_gpus)
    cfg = configuration(cell)
    rows = selected_rows(cell)
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with Path(gpu_lock or shared.LOCK).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | (fcntl.LOCK_NB if lock_nonblocking else 0))


        from .live import remaining_budget
        timeout = remaining_budget(cell["experiment_id"], out.parent.parent, timeout)
        out.mkdir()
        shared.dump(out/"cell.json", cell)
        command = [sys.executable, "-m", "sregnas.icde27.p2p6.entry", "--job", str(out/"cell.json"), "--out", str(out), "--worker"]
        if cell.get("collect_search_garbage"):
            command.append("--collect-search-garbage")
        info = shared.metadata(cfg, cell["method"], out, cell["experiment_id"], cell["record_type"], command)
        info.update(timing_scope="quality_only" if quality_only else "exclusive_primary",gpu_lock=str(gpu_lock or shared.LOCK))
        if quality_only: info["timing_valid"] = False
        info.update(cell=cell, Q_requested=cell["count"] if cell["kind"] == "replay" else 0,
            native_rng_reset=True, controller_updates=0,
            K=0 if cell["kind"] == "replay" else cell["count"], S=logical_shards(cell) if logical_shards(cell) is not None else "N/A",
            S_reason="architecture parallelism has no logical graph shards" if logical_shards(cell) is None else None,
            execution_key=execution_key(cell, cfg.to_dict(), rows),
            pool_semantics_hash=digest([json.loads(r["architecture_json"]) for r in rows]),
            pool_id=DATASETS[cell["dataset"]]+"_pool32", pool_file_sha256=shared.sha(ROOT/"pools"/(DATASETS[cell["dataset"]]+"_pool32.tsv")),
            candidate_mapping_id=candidate_mapping(cell),
            scaling_mode=cell.get("scaling_mode", "fixed_workload"), max_wall_seconds=timeout,
            execution_profile_id=digest(dict(method=cell["method"], kind=cell["kind"], executor=cell["executor"],
                proxy_epochs=cfg.train.proxy_epochs, retrain_epochs=cfg.train.retrain_epochs, seed_rule=info["seed_rule"])),
            fixed_step_schedule_id=f"native_rotation_{logical_shards(cell)}" if cell["method"] == "FastGNAS" else "50_steps_per_replica" if cell["method"] in ("GraphNAP", "AP", "PAS-PL") else f"{cfg.train.proxy_epochs}_proxy_epochs")
        info["seed_rule"]["controller"] = "none: fixed candidate list"
        if cell["method"] in ("GraphNAP", "AP", "PAS-PL") and cell["kind"] == "replay":
            info["seed_rule"] = dict(controller="none: fixed candidate list", initialization="training.minibatch._hash_seed(fingerprint,search,0)",
                training="training.minibatch._hash_seed(fingerprint,search,0,partition_id)" if cell["method"]=="GraphNAP" else "initialization RNG continues")
        if cell["kind"] == "cell" and cell["method"] != "FastGNAS":


            configs = [configuration(dict(cell, method=method)).to_dict() for method in ("SREGNAS", "GraphNAP", "AP")]
            assert configs[0] == configs[1] == configs[2]
            info["verified_canonical_methods"] = ["SREGNAS", "GraphNAP", "AP"]
            info["canonical_match_key"] = digest(dict(config=configs[0], architecture=json.loads(rows[0]["architecture_json"]),
                backend="canonical", graph_hash=info["graph_hash"], split_hash=info["split_hash"],
                initialization="H(fingerprint,retrain,0)", checkpoint="final_epoch", executor=EXECUTORS[cell["executor"]],
                boundary="single_cell_spawn_to_validation_return", P=cell["P"], S=logical_shards(cell),
                implementation=info["implementation_sha256"]))
        shared.dump(out/"run.json", info)
        if timeout <= 0:
            info.update(status="incomplete", process_wall_s=0., timing_valid=False,
                        blocking_reason="experiment exceeded estimated upper bound ×1.5 including authorized supplements")
            return final_record(out, info, cell)
        identities = {}
        before = shared.snapshot(monitor_gpus, os.getpid(), identities)
        shared.dump(out/"resources_before.json", before)
        if before["gpu"]["returncode"] or before["compute"]["returncode"]:
            info.update(status="unsupported", blocking_reason="GPU telemetry unavailable", timing_valid=False)
            return final_record(out, info, cell)
        if before["foreign_processes"] or (not quality_only and before["cpu_contention_observed"]):
            info.update(status="contaminated", blocking_reason="contention after flock", timing_valid=False)
            return final_record(out, info, cell)
        from sregnas.execution.telemetry import gpu_idle_preflight
        release_baseline = gpu_idle_preflight(monitor_gpus, owner_pid=os.getpid(), verified_owned=identities)
        shared.dump(out/"release_baseline.json", release_baseline)
        env = dict(os.environ, SREGNAS_ICDE27_RECORD_DIR=str(out), SREGNAS_ALLOW_GPU="1",
            SREGNAS_APPROVED_GPUS=",".join(map(str, range(cell["P"]))),
            CUDA_VISIBLE_DEVICES=",".join(map(str, range(cell["P"]))),
            SREGNAS_ICDE27_NATIVE_RNG="1", SREGNAS_P2P6_PROFILE="1" if cell.get("profile") else "0",
            SREGNAS_EXP3_FINE_PROFILE="1" if fine_profile_requested(cell) else "0",
            SREGNAS_ICDE27_PROFILE="1" if cell["kind"] != "replay" else "0",
            SREGNAS_DATA_DIR=os.environ.get('SREGNAS_DATA_DIR', str(_artifact_root / 'data')), PYTHONPATH=str(ROOT/"src"),
            OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", PYTHONUNBUFFERED="1")
        observations = []
        device_memory = []
        host_memory = []
        started = time.perf_counter(); timed_out = False
        info["process_started_epoch_s"] = time.time()
        with (out/"raw.log").open("w") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            from sregnas.execution.telemetry import _start_ticks
            info["pipeline_process"] = dict(pid=child.pid, pgid=child.pid, start_ticks=_start_ticks(child.pid), recorder_pid=os.getpid())
            shared.dump(out/"run.json", info)
            try:
                while child.poll() is None:
                    observation = shared.snapshot(monitor_gpus, os.getpid(), identities)
                    observations.append(observation)
                    host = rss_tree(child.pid, [r["pid"] for r in observation["processes"] if r["owned"]])
                    host_memory.append(host)
                    with (out/"host_memory.jsonl").open("a") as stream: stream.write(json.dumps(host)+"\n")
                    with (out/"resources_during.jsonl").open("a") as stream: stream.write(json.dumps(observation)+"\n")
                    memory = subprocess.run(["nvidia-smi", "-i", ",".join(map(str, range(cell["P"]))),
                        "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
                    for line in csv.reader(memory.stdout.splitlines()):
                        if len(line)==2:
                            try: device_memory.append(dict(gpu=int(line[0]), used_GiB=float(line[1])/1024, perf_s=time.perf_counter()))
                            except ValueError: pass
                    if time.perf_counter()-started >= timeout:
                        timed_out = True; terminate(child); break
                    time.sleep(.5)
            finally:
                if child.poll() is None: terminate(child)
        from .cleanup import wait_for_gpu_release
        settled = wait_for_gpu_release(monitor_gpus, owner_pid=os.getpid(),
            verified_owned=identities, baseline=release_baseline, out=out,
            max_idle_drift_wait_s=120 if cell.get('x567_queue') else None)
        shared.dump(out/"resources_settled.json", settled)
        info["gpu_released_before_unlock"] = settled["idle"]
        after = shared.snapshot(monitor_gpus, os.getpid(), identities)
        shared.dump(out/"resources_after.json", after)
        info.update(process_wall_s=time.perf_counter()-started, allocation_wall_gpu_hours=cell["P"]*(time.perf_counter()-started)/3600,
            status="timeout" if timed_out else "OK" if child.returncode == 0 else "incomplete", returncode=child.returncode,
            timing_valid=not settled["foreign_process_observed"] and not any(o["foreign_processes"] or o["cpu_contention_observed"] or o["gpu"]["returncode"] or o["compute"]["returncode"] for o in [before,*observations,after]))
        info.update(process_ended_epoch_s=time.time(),
            peak_rss_bytes=max((s["rss_bytes"] for s in host_memory if s["rss_bytes"] is not None), default=None),
            peak_rss_definition="sampled sum of owned process-tree RSS; shared pages may be counted repeatedly; not continuous high-water mark")


        peaks = [r.get("memory_mib", 0)/1024 for o in observations for r in o["processes"] if r["owned"]]
        info["peak_single_process_gib"] = max(peaks) if peaks else None
        shared.dump(out/"device_memory.json", device_memory)
        info["worst_card_peak_gib"] = max((r["used_GiB"] for r in device_memory), default=None)
        info["peak_memory_definition"] = "sampled whole-card used memory including recorded desktop baseline; max over reserved GPUs"
        return final_record(out, info, cell)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-wall-seconds", type=float, default=3600)
    parser.add_argument("--gpu-lock",type=Path)
    parser.add_argument("--quality-only",action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--try-lock", action="store_true")
    parser.add_argument("--collect-search-garbage", action="store_true",
                        help="AP/GraphNAP replay: collect unreachable loader cycles after each candidate")
    args = parser.parse_args()
    cell = apply_amendments(json.loads(args.job.read_text()))
    if args.collect_search_garbage:
        if cell["method"] not in ("AP", "GraphNAP") or cell["kind"] != "replay":
            parser.error("--collect-search-garbage requires AP or GraphNAP replay")
        cell["collect_search_garbage"] = True
    out = args.out.resolve()
    if not args.worker:
        try:
            if args.gpu_lock is not None or args.quality_only:
                return record(cell, out, args.max_wall_seconds, args.try_lock,
                              gpu_lock=args.gpu_lock, quality_only=args.quality_only)
            return record(cell, out, args.max_wall_seconds, args.try_lock)
        except BlockingIOError:
            if args.try_lock: return 75
            raise
    if not os.environ.get("SREGNAS_ICDE27_RECORD_DIR"): raise RuntimeError("recorder required")
    from sregnas.comparison.launch import wait_gpu_idle_preflight
    try:
        mark_stage(out, "prep/idle_preflight")
        if not wait_gpu_idle_preflight(list(range(cell["P"])), timeout_s=min(300, args.max_wall_seconds))["idle"]:
            raise RuntimeError("GPU idle preflight failed")
        if cell["kind"] != "replay" or cell["method"] == "SREGNAS":
            from .canonical import execute
        elif cell["method"] == "FastGNAS":
            from .native import execute
        else:
            from .ports import execute
        mark_stage(out, "prep/entry")
        result = execute(cell, configuration(cell), out)
        shared.dump(out/"measurement.json", result)
    except BaseException as exc:
        traceback.print_exc()
        message = str(exc).lower()
        status = "OOM" if is_oom(type(exc).__name__+message) else "unsupported" if isinstance(exc, (ImportError, NotImplementedError)) else "incomplete"
        shared.dump(out/"failure.json", dict(status=status, exception=type(exc).__name__, error=str(exc), failure_stage=stage_at_failure(out)))
        return 2
    return 0 if all(r["status"] == "OK" for r in result["rows"]) else 2


if __name__ == "__main__": raise SystemExit(main())
