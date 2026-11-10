
import csv
import json
import os
from pathlib import Path
import resource
import subprocess

from sregnas.baselines.paspl.adapter import Worker


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


class ObservedPreloadWorker(Worker):
    pass

    def run_candidate(self, spec):
        row = super().run_candidate(spec)
        if self.config.get("collect_search_garbage", False):

            import gc
            import time
            from sregnas.baselines.graphnap.truncation import host_rss_bytes
            started = time.perf_counter()
            row["host_rss_before_collect_bytes"] = host_rss_bytes()
            row["unreachable_objects_collected"] = gc.collect()
            row["host_rss_bytes"] = host_rss_bytes()
            row["garbage_collection_wall_s"] = time.perf_counter() - started
            row["wall_s"] += row["garbage_collection_wall_s"]
        return row

    def metrics(self):
        row = super().metrics()
        row.update(label_device=str(self.full_data.y.device),
                   adjacency_device=str(self.full_data.edge_index.device),
                   cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                   host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        return row


def record_preloads(workers, out):
    import ray
    futures = [w.metrics.remote() for w in workers]
    rows = []
    for index, future in enumerate(futures):
        try:
            rows.append(dict(ray.get(future), status="OK", requested_worker=index))
        except Exception as exc:
            rows.append(dict(requested_worker=index, status="failed",
                             error=f"{type(exc).__name__}: {exc}"))
        write_json(Path(out) / "preload_workers.json", rows)
    if any(r["status"] != "OK" for r in rows):

        raise RuntimeError("X15 preload failed: out of memory (preregistered preload criterion)")
    return rows


def augment_snapshot(observation, owner, gpus):
    from sregnas.execution.telemetry import _owned
    process = subprocess.run(["ps", "-eo", "pid=,rss="], text=True, capture_output=True)
    rss = {}
    for line in process.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and _owned(int(parts[0]), owner):
            rss[parts[0]] = int(parts[1]) * 1024
    observation["owned_process_rss_bytes"] = rss
    observation["host_rss_bytes"] = sum(rss.values()) if process.returncode == 0 else None
    query = subprocess.run(["nvidia-smi", "-i", ",".join(map(str, gpus)),
                            "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                           text=True, capture_output=True)
    observation["gpu_memory_query_returncode"] = query.returncode
    observation["gpu_memory_mib"] = {}
    for row in csv.reader(query.stdout.splitlines()):
        if len(row) == 2:
            try:
                observation["gpu_memory_mib"][str(int(row[0]))] = float(row[1])
            except ValueError:
                pass


def memory_termination(text):
    lower = text.lower()
    return any(s in lower for s in (
        "outofmemoryerror", "out of memory", "oom-kill", "oom_kill",
        "killed due to memory pressure", "memory usage threshold", "memoryerror",
        "node running low on memory", "cannot allocate memory", "can't allocate memory"))


def terminal_actor_failure(candidates):
    pass
    row = next((r for r in candidates if r.get("status") == "OOM" and (
        r.get("failure_stage") == "actor" or
        "Task was killed due to the node running low on memory" in r.get("error", ""))), None)
    if row is None:
        return None
    return dict(status="OOM", failure_stage="search/actor", error=row.get("error", ""),
                source="pipeline/port_candidates.json", fingerprint=row["fingerprint"],
                attempt=row["attempt"], policy="persistent actor lost; no dataset retry")


def precheck_decision(preloads, candidates, *, failure=None, returncode=0, killed_for_memory=False):
    pass
    failure = failure or {}
    preload_failed = any(r.get("status") != "OK" for r in preloads) or (
        failure.get("failure_stage") == "graph_preload")
    killed_for_memory = bool(killed_for_memory or memory_termination(json.dumps(failure)) or any(
        r.get("failure_stage") == "actor" and memory_termination(r.get("error", ""))
        for r in candidates))
    oom_count = sum(r.get("status") == "OOM" or memory_termination(r.get("error", ""))
                    for r in candidates)
    reasons = []
    if preload_failed:
        reasons.append("at least one GPU preload failed")
    if killed_for_memory:
        reasons.append("process terminated for GPU/host memory")
    if oom_count >= 4:
        reasons.append("at least 4 of 16 candidates OOM")
    complete = len(preloads) == 8 and len(candidates) == 16 and all(
        r.get("status") in ("OK", "OOM") for r in candidates) and returncode == 0
    status = "OOM" if reasons else "OK" if complete else "incomplete"
    return dict(status=status, preload_workers=len(preloads), candidate_count=len(candidates),
                candidate_oom_count=oom_count, process_memory_termination=killed_for_memory,
                reasons=reasons, full_nas_eligible=status == "OK", candidates=candidates,
                workers=preloads)


def finalize_search_only(out, info, rows, launches):
    from sregnas.icde27.recorder import write_table
    search = [r for r in launches if r.get("stage") == "search" and
              r["kind"] in {"candidate", "retry", "resume"}]
    info.update(Q_search=len(search), Q_attempt=len(search), Q_probe=0, Q_retry=0, Q_resume=0,
                K=0, K_actual=0, measurement_scope="precheck_search_only")
    info["assertions"].update(candidate_launch_count=len(search) == info["Q_requested"],
                              candidate_result_count=len(rows) == len(search))
    write_table(out / "launch_ledger.tsv", search,
                ["launch_id", "kind", "stage", "identity", "fingerprint", "perf_s"])


def finalize_resources(out, info, observations):
    def read(path, default):
        return json.loads(path.read_text()) if path.exists() else default

    preloads = read(out / "pipeline/preload_workers.json", [])
    candidates = read(out / "pipeline/port_candidates.json", [])
    host = [r["host_rss_bytes"] for r in observations if r.get("host_rss_bytes") is not None]
    info.update(per_gpu_peak_memory_GiB={str(g): max(
        (r["gpu_memory_mib"][str(g)] / 1024 for r in observations
         if str(g) in r.get("gpu_memory_mib", {})), default=None) for g in info["physical_gpus"]},
        host_peak_rss_GiB=max(host) / 2**30 if host else None,
        host_memory_definition="peak sampled sum of owned process RSS; shared pages may be counted multiple times",
        gpu_memory_definition="peak sampled device memory.used, including the recorded desktop baseline",
        preload_workers=preloads, Q_success=sum(r["status"] == "OK" for r in candidates))
    from sregnas.icde27.recorder import read_rows
    events = read_rows(out / "events.jsonl")
    for metric in ("allocated", "reserved"):
        peaks = {str(g): [] for g in info["physical_gpus"]}
        for row in preloads + candidates:
            gpu = row.get("cuda_visible_devices", "")
            value = row.get("peak_" + metric + "_bytes")
            if gpu in peaks and value is not None:
                peaks[gpu].append(value / 2**30)
        for event in events:
            if event.get("event") != "cell_result":
                continue
            result = event["result"]
            gpu, value = str(result.get("gpu")), result.get("peak_" + metric + "_gb")
            if gpu in peaks and value is not None:
                peaks[gpu].append(value)
        info["per_gpu_peak_" + metric + "_GiB"] = {g: max(v) if v else None for g, v in peaks.items()}
    if info.get("stage") == "search":
        check = precheck_decision(preloads, candidates, failure=info.get("failure"),
                                  returncode=info.get("returncode"))
        check.update(per_gpu_peak_memory_GiB=info["per_gpu_peak_memory_GiB"],
                     host_peak_rss_GiB=info["host_peak_rss_GiB"])
        write_json(out / "precheck.json", check)
        info["precheck"] = check
        if check["status"] != "OK":
            info["status"] = check["status"]
    elif info["status"] != "OK" and memory_termination(json.dumps(info.get("failure", {}))):
        info["status"] = "OOM"
    elif info["status"] != "OK" and info.get("failure", {}).get("failure_stage") == "retrain":


        if "ranking exhausted" in info["failure"].get("error", "") and any(
                memory_termination(r.get("error", "")) for r in candidates):
            info["status"] = "OOM"
    if info["status"] == "OOM":
        info["stop_reason"] = f"OOM（第 {info.get('Q_attempt', 0)} 次 launch 后）"
