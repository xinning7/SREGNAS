
import json
import subprocess
import time
from pathlib import Path


def mark_stage(out, stage):
    path = Path(out)/"failure_stage.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(stage=stage, epoch_s=time.time()))+"\n")
    temporary.replace(path)


def stage_at_failure(out):
    try: return json.loads((Path(out)/"failure_stage.json").read_text())["stage"]
    except (FileNotFoundError, json.JSONDecodeError): return "before_worker"


def is_oom(value):
    message = str(value).lower()
    return any(s in message for s in ("out of memory", "outofmemoryerror", "memoryerror",
        "node running low on memory", "cannot allocate memory", "can't allocate memory"))


def rss_tree(root_pid, known_owned=()):
    pass
    value = subprocess.run(["ps", "-eo", "pid=,ppid=,rss="], text=True, capture_output=True)
    if value.returncode: return dict(epoch_s=time.time(), rss_bytes=None, processes=[])
    rows = []
    for line in value.stdout.splitlines():
        try: pid, ppid, rss = map(int, line.split())
        except ValueError: continue
        rows.append(dict(pid=pid, ppid=ppid, rss_bytes=rss*1024))
    owned = {root_pid, *known_owned}
    while True:
        more = {r["pid"] for r in rows if r["ppid"] in owned}
        if more <= owned: break
        owned |= more
    selected = [r for r in rows if r["pid"] in owned]
    return dict(epoch_s=time.time(), rss_bytes=sum(r["rss_bytes"] for r in selected), processes=selected)


def failure_evidence(out, rows, info):
    failed = next((r for r in rows if r.get("status") == "OOM"), None)
    failure = info.get("failure", {})
    raw = Path(out)/"raw.log"

    messages = []
    if raw.exists():
        with raw.open(errors="replace") as stream:
            messages = [line.strip()[-3000:] for line in stream if is_oom(line)][-4:]
    if failed or is_oom(failure) or messages:
        if info.get("status") not in ("timeout", "contaminated"):
            info["status"] = "OOM"
        source = failed or failure
        info["failure"] = dict(failure, status="OOM",
            failure_stage=source.get("failure_stage") or stage_at_failure(out),
            error=source.get("error") or "\n".join(messages) or "candidate reported OOM",
            evidence="candidate/failure record and raw.log", peak_rss_bytes=info.get("peak_rss_bytes"),
            worst_card_peak_gib=info.get("worst_card_peak_gib"),
            memory_kind="host" if any(s in str(source).lower()+str(messages).lower()
                for s in ("node running low on memory", "memoryerror", "cannot allocate memory"))
                and "cuda out of memory" not in str(source).lower()+str(messages).lower() else "device_or_unspecified")
    if info.get("status") != "OK":
        info.setdefault("failure", dict(status=info["status"], error="see raw.log and candidate records"))
        info["failure"].setdefault("failure_stage", stage_at_failure(out))
        info["failure"].setdefault("peak_rss_bytes", info.get("peak_rss_bytes"))
        info["failure"].setdefault("worst_card_peak_gib", info.get("worst_card_peak_gib"))


def pool_estimate(cell, info, rows):
    pass
    if not cell.get("estimate_target_count"): return
    info.update(measured_candidate_count=len(rows), estimate_target_count=cell["estimate_target_count"],
                estimate_formula="prep_s + mean(measured_candidate_wall_s) * target_count")
    if (info.get("status") == "OK" and info.get("timing_valid")
            and len(rows) == cell["count"] and all(r.get("status") == "OK" and r.get("wall_s") is not None for r in rows)
            and info.get("prep_s") is not None):
        mean = sum(r["wall_s"] for r in rows)/len(rows)
        info.update(measured_candidate_mean_s=mean,
                    estimated_pool32_s=info["prep_s"]+mean*cell["estimate_target_count"])


def capacity_reference(result_root):
    pass
    path = Path(result_root)/"p2p6_smoke_status.json"
    if not path.exists(): return {}
    rows = json.loads(path.read_text())
    row = next((r for r in rows if r["smoke_key"] == "core_6" and r["status"] == "OOM"), None)
    if not row: return {}
    out = Path(row["path"])
    info = json.loads((out/"run.json").read_text())
    return dict(source_run_id=info["run_id"], source_path=str(out), source_dataset="Flickr",
        source_failure_stage="pool/native_training_and_evaluation",
        source_peak_single_process_gib=info.get("peak_single_process_gib"),
        source_worst_card_peak_gib=info.get("worst_card_peak_gib"),
        target_dataset_measured=False, decision="user_confirmed_fixed_S8_P2_capacity_OOM")
