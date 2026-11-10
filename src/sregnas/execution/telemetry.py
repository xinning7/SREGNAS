

from __future__ import annotations

import csv
import json
import os
import pwd
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from sregnas.paths import PROJECT_ROOT as ROOT


def write_snapshot(path: Path, gpus: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.used,memory.total,"
            "utilization.gpu,clocks.sm,temperature.gpu",
            "--format=csv,noheader",
        ],
        text=True,
    )
    wanted = set(gpus)
    gpu = (
        "\n".join(
            line
            for line in all_gpu.splitlines()
            if int(line.split(",", 1)[0].strip()) in wanted
        )
        + "\n"
    )
    procs = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
    )
    path.write_text("[gpu]\n" + gpu + "[compute_processes]\n" + procs)


def _parent(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[1])
    except Exception:
        return None


def _start_ticks(pid: int) -> int | None:
    pass
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[19])
    except Exception:
        return None


def _owned(pid: int, owner: int) -> bool:
    pass
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == owner:
            return True
        seen.add(pid)
        parent = _parent(pid)
        if parent is None:
            return False
        pid = parent
    return False


def classify_gpu_process(pid, name, owner, verified_owned, *, retain_live_cache=False):
    pass
    ticks = _start_ticks(pid)
    previous = verified_owned.get(pid)
    if previous is not None and not isinstance(previous, dict):
        previous = dict(start_ticks=previous, name=None, exited_event_recorded=False)
    real_name = name != "[No data]"
    changed = previous is not None and (
        (ticks is not None and previous.get("start_ticks") is not None
         and ticks != previous["start_ticks"])
        or (real_name and previous.get("name") is not None and name != previous["name"])
    )
    if changed:
        verified_owned.pop(pid, None)
        previous = None
    exiting = previous is not None and (ticks is None or not real_name)
    cached_live = retain_live_cache and previous is not None
    owned = exiting or cached_live or _owned(pid, owner)
    event = None
    if owned:
        identity = dict(previous or {})
        if ticks is not None:
            identity["start_ticks"] = ticks
        if real_name:
            identity["name"] = name
        if exiting and not identity.get("exited_event_recorded", False):
            event = dict(event="exited_owned", pid=pid, observed_name=name,
                         confirmed_name=identity.get("name"),
                         confirmed_start_ticks=identity.get("start_ticks"),
                         process_missing=ticks is None)
            identity["exited_event_recorded"] = True
        verified_owned[pid] = identity
    return dict(owned=bool(owned), start_ticks=ticks, identity_changed=changed,
                exited_owned=exiting, ownership_event=event)


def _process_owner(pid: int) -> str:
    try:
        return pwd.getpwuid(Path(f"/proc/{pid}").stat().st_uid).pw_name
    except Exception:
        return "UNKNOWN"


def gpu_idle_preflight(
    gpus: list[int],
    *,
    owner_pid: int | None = None,
    verified_owned: dict | None = None,
) -> dict[str, Any]:
    pass
    all_rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu,"
            "clocks.sm,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    wanted = set(gpus)
    rows = "\n".join(
        line
        for line in all_rows.splitlines()
        if int(line.split(",", 1)[0].strip()) in wanted
    )
    snapshots = []
    for fields in csv.reader(rows.splitlines()):
        snapshots.append(
            {
                "gpu": int(fields[0].strip()),
                "uuid": fields[1].strip(),
                "memory_used_mib": int(fields[2].strip()),
                "utilization_pct": int(fields[3].strip()),
                "sm_clock_mhz": int(fields[4].strip()),
                "temperature_c": int(fields[5].strip()),
            }
        )
    processes = []
    for gpu in gpus:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        for fields in csv.reader(output.splitlines()):
            if not fields:
                continue
            processes.append(
                {
                    "gpu": gpu,
                    "pid": int(fields[0].strip()),
                    "process_name": fields[1].strip(),
                    "used_memory_mib": int(fields[2].strip()),
                }
            )
    verified_owned = verified_owned if verified_owned is not None else {}
    for process in processes:
        pid = int(process["pid"])
        process["owner"] = _process_owner(pid)
        if owner_pid is None:
            process["owned_by_run"] = None
            continue
        classification = classify_gpu_process(
            pid, process["process_name"], owner_pid, verified_owned,
            retain_live_cache=True,
        )
        process.update(classification)
        process["owned_by_run"] = classification["owned"]
    nonzero_utilization = [row for row in snapshots if int(row["utilization_pct"]) != 0]
    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpus": list(gpus),
        "snapshots": snapshots,
        "processes": processes,
        "ownership_events": [p["ownership_event"] for p in processes if p.get("ownership_event")],
        "owned_processes": [
            process for process in processes if process.get("owned_by_run") is True
        ],
        "foreign_processes": [
            process for process in processes if process.get("owned_by_run") is not True
        ],
        "nonzero_utilization": nonzero_utilization,
        "idle": not processes and not nonzero_utilization,
    }

    json.dumps(result, sort_keys=True)
    return result


def gpu_idle_preflight_settled(
    gpus: list[int],
    *,
    owner_pid: int,
    settle_delay_s: float = 4.0,
    poll_interval_s: float = 2.0,
    timeout_s: float = 30.0,
    required_consecutive_idle: int = 2,
) -> dict[str, Any]:
    pass
    started_at = datetime.now(timezone.utc).isoformat()
    if settle_delay_s > 0:
        time.sleep(settle_delay_s)
    deadline = time.monotonic() + timeout_s
    verified_owned: dict = {}
    samples: list[dict[str, Any]] = []
    consecutive_idle = 0
    decision = "SETTLE_TIMEOUT"
    while True:
        snapshot = gpu_idle_preflight(
            gpus,
            owner_pid=owner_pid,
            verified_owned=verified_owned,
        )
        samples.append(snapshot)
        if snapshot["foreign_processes"]:
            decision = "FOREIGN_PROCESS_ABORT"
            break
        if snapshot["idle"]:
            consecutive_idle += 1
            if consecutive_idle >= required_consecutive_idle:
                decision = "PASS_AFTER_SETTLE"
                break
        else:
            consecutive_idle = 0
        if time.monotonic() >= deadline:
            break
        time.sleep(poll_interval_s)

    result = dict(samples[-1])
    result.update(
        {
            "settling_started_at_utc": started_at,
            "settling_finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "settle_delay_s": settle_delay_s,
            "poll_interval_s": poll_interval_s,
            "timeout_s": timeout_s,
            "required_consecutive_idle": required_consecutive_idle,
            "consecutive_idle_samples": consecutive_idle,
            "settling_samples": samples,
            "settling_decision": decision,
            "idle": decision == "PASS_AFTER_SETTLE",
        }
    )
    json.dumps(result, sort_keys=True)
    return result


def start_monitors(
    out_dir: Path, gpus: list[int], interval_s: float
) -> list[subprocess.Popen]:
    monitor = Path(__file__).with_name("monitor.py")
    processes = []
    for gpu in gpus:
        log = (out_dir / f"gpu{gpu}_monitor.log").open("w")
        proc = subprocess.Popen(
            [
                sys.executable,
                str(monitor),
                "--gpu",
                str(gpu),
                "--owner-pid",
                str(os.getpid()),
                "--out",
                str(out_dir / f"gpu{gpu}_monitor.tsv"),
                "--interval",
                str(interval_s),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
        )
        processes.append(proc)
    return processes


def stop_monitors(processes: list[subprocess.Popen]) -> None:
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    for proc in processes:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def start_machine_load_monitor(
    out_dir: Path, interval_s: float = 10.0
) -> list[subprocess.Popen]:
    pass
    monitor = Path(__file__).with_name("host_monitor.py")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (out_dir / "machine_load.log").open("w")
    proc = subprocess.Popen(
        [
            sys.executable,
            str(monitor),
            "--out",
            str(out_dir / "machine_load.tsv"),
            "--interval",
            str(interval_s),
            "--owner-user",
            pwd.getpwuid(os.getuid()).pw_name,
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=ROOT,
    )
    return [proc]
