
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import time
import uuid


def summarize_trace(path):
    trace = json.loads(Path(path).read_text())
    categories = {}
    for event in trace.get("traceEvents", []):
        if event.get("ph") != "X": continue
        name = event.get("name", "")
        lower = name.lower()
        category = ("H2D" if "htod" in lower or "host to device" in lower else
                    "D2H" if "dtoh" in lower or "device to host" in lower else
                    "D2D" if "dtod" in lower or "device to device" in lower else
                    "wait" if any(k in lower for k in ("synchronize", "waitevent")) else
                    "collective" if any(k in lower for k in ("nccl", "allreduce", "allgather")) else None)
        if category is None: continue
        item = categories.setdefault(category, dict(events=0, duration_s=0., bytes=0, missing_byte_events=0))
        item["events"] += 1
        item["duration_s"] += event.get("dur", 0)/1e6
        if category != "wait":
            args = event.get("args", {})
            size = next((v for k,v in args.items() if k.lower() in ("bytes", "byte size", "size (bytes)")), None)
            if isinstance(size, (int, float)): item["bytes"] += size
            else: item["missing_byte_events"] += 1
    for key, item in categories.items():
        if item["missing_byte_events"] or key == "wait": item["bytes"] = None
    return categories


@contextmanager
def capture(label, enabled=None):
    if enabled is None: enabled = os.environ.get("SREGNAS_P2P6_PROFILE") == "1"
    if not enabled:
        yield
        return
    import torch
    root = Path(os.environ["SREGNAS_ICDE27_RECORD_DIR"]) / "profiles"
    root.mkdir(exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", label) + f"_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    start = time.perf_counter()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available(): activities.append(torch.profiler.ProfilerActivity.CUDA)
    try:
        with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=False,
                                    with_stack=False) as prof:
            yield
    finally:
        end = time.perf_counter()
        target = root / (name + ".json")
        prof.export_chrome_trace(str(target))
        summary = dict(label=label, pid=os.getpid(), start_perf_s=start, end_perf_s=end,
                       wall_s=end-start, trace=str(target), categories=summarize_trace(target),
                       timing_scope="profile_only; overlapping event durations are not exclusive phases",
                       wait_definition="observed runtime synchronization API duration; worker task gaps are recorded separately")
        target.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2)+"\n")


def worker_tail(spans, pool_start, pool_end):
    result = []
    for worker in sorted({r["worker"] for r in spans}, key=str):
        rows = [r for r in spans if r["worker"] == worker]
        first = min(r["start_perf_s"] for r in rows)
        last = max(r["end_perf_s"] for r in rows)
        busy = sum(r["end_perf_s"]-r["start_perf_s"] for r in rows)
        result.append(dict(worker=worker, tasks=len(rows), first_start_perf_s=first,
            last_end_perf_s=last, busy_s=busy, completion_offset_s=last-pool_start,
            tail_idle_s=max(0., pool_end-last),
            between_tasks_wait_s=max(0., last-first-busy)))
    return result
