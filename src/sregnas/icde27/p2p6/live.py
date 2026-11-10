
from copy import deepcopy
from datetime import datetime
import csv
import hashlib
import json
import math
from pathlib import Path

from .plan import LIMITS, selected_rows

LIVE_NAME = "p2p6_live_amendments.json"
ACTIVE = ("pending", "running", "waiting_lock")


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def run_info(attempt):
    path = attempt.get("result_path")
    return read_json(Path(path)/"run.json", {}) if path else {}


def gc_setting(attempt):
    info = run_info(attempt)
    if not info: return None
    setup = read_json(Path(attempt["result_path"])/"port_setup.json", {})

    return bool(setup.get("config", {}).get("collect_search_garbage",
                info.get("cell", {}).get("collect_search_garbage", False)))


def measured_seconds(attempts, experiment_id):
    seen, seconds = set(), 0.
    for attempt in attempts:
        path = attempt.get("result_path")
        if not path or path in seen: continue
        seen.add(path)
        info = run_info(attempt)

        if str(info.get("experiment_id")) == str(experiment_id):
            seconds += float(info.get("process_wall_s", 0.))
    return seconds


def remaining_budget(experiment_id, results, supplied):
    journal = read_json(Path(results)/LIVE_NAME, {})
    if not journal.get("enabled") or str(experiment_id) not in ("6", "7"):
        return supplied
    base = read_json(Path(results)/"p2p6_queue_state.json", {})
    attempts = base.get("attempts", []) + journal.get("attempts", [])
    return min(supplied, LIMITS[str(experiment_id)] - measured_seconds(attempts, experiment_id))


def combine(base, journal):
    pass
    state = deepcopy(base)
    known = {a.get("authorization_id") for a in state["attempts"] if a.get("authorization_id")}
    for original in journal.get("attempts", []):
        if original["authorization_id"] in known: continue
        attempt = deepcopy(original)
        attempt["queue_index"] = len(state["attempts"])
        state["attempts"].append(attempt)
        known.add(attempt["authorization_id"])
    cells = {c["cell_id"]: c for c in state["cells"]}
    for a in state["attempts"]:
        if cells[a["cell_id"]]["method"] == "AP":
            a["collect_search_garbage"] = gc_setting(a)
    for exp in ("6", "7"):
        state["used_seconds"][exp] = measured_seconds(state["attempts"], exp)
        extras = [a for a in journal.get("attempts", []) if a["experiment_id"] == exp]
        if not extras: continue
        timing = state.setdefault("experiment_timing", {}).setdefault(exp, {})
        times = [a["started_epoch_s"] for a in extras if a.get("started_epoch_s")]
        if times:
            start = min(times + ([timing["started_epoch_s"]] if timing.get("started_epoch_s") else []))
            timing.update(started_epoch_s=start, started_at=datetime.fromtimestamp(start).isoformat())
        if any(a["status"] in ACTIVE for a in state["attempts"] if a["experiment_id"] == exp):
            for k in ("ended_epoch_s", "ended_at", "wall_s"): timing.pop(k, None)
        else:
            ends = [a["ended_epoch_s"] for a in extras if a.get("ended_epoch_s")]
            if ends and timing.get("started_epoch_s"):
                end = max(ends + [timing.get("ended_epoch_s", 0)])
                timing.update(ended_epoch_s=end, ended_at=datetime.fromtimestamp(end).isoformat(), wall_s=end-timing["started_epoch_s"])
    return state


def requests(state):
    pass
    latest = {a["cell_id"]: a for a in state["attempts"]}
    output, audit = [], []
    for cell in state["cells"]:
        a = latest[cell["cell_id"]]
        if (cell["experiment_id"] == "7" and cell.get("reuse_cell_id")
                and a["status"] == "incomplete" and not a.get("result_path")
                and ("reuse" in a.get("reason", "") or "execution key" in a.get("reason", ""))):
            job = dict(cell); job.pop("reuse_cell_id")
            output.append(dict(authorization_id="exp7_fresh:"+cell["cell_id"], cell=job,
                source_attempt=a["attempt"], reason="用户授权：Exp7 完整 key 复用失败后直接新跑；"+a.get("reason", "")))
    for dataset in ("arXiv", "Products"):
        cells = [c for c in state["cells"] if c["experiment_id"] == "6" and c["method"] == "AP" and c["dataset"] == dataset]
        settings = {c["P"]: gc_setting(latest[c["cell_id"]]) for c in cells}
        complete = len(settings) == 3 and all(v is not None for v in settings.values())
        majority = sum(settings.values()) >= 2 if complete else None
        audit.append(dict(dataset=dataset, settings=settings, consistent=len(set(settings.values())) == 1 if complete else None,
                          majority=majority, complete=complete))
        if not complete: continue
        for c in cells:
            a = latest[c["cell_id"]]
            if settings[c["P"]] == majority or a["status"] in ACTIVE: continue
            job = dict(c, gc_consistency_repair=True, collect_search_garbage=majority)
            output.append(dict(authorization_id=f"ap_gc_majority:{dataset}:P{c['P']}:{majority}", cell=job,
                source_attempt=a["attempt"], reason=f"用户授权：AP 同数据集 P2/P4/P8 GC 一致性补跑，少数 P{c['P']} 改为多数设置 {majority}"))
    return output, audit


def score_rows(state):
    pass
    latest = {a["cell_id"]: a for a in state["attempts"]}
    output = []
    for cell in state["cells"]:
        if cell["experiment_id"] != "7" or cell["kind"] != "replay": continue
        a = latest[cell["cell_id"]]; info = run_info(a)
        path = Path(a["result_path"])/"candidates.tsv" if a.get("result_path") else None
        rows = []
        if path and path.exists():
            with path.open() as stream: rows = list(csv.DictReader(stream, delimiter="\t"))
        indexed = {r["fingerprint"]: r for r in rows}
        if len(indexed) != len(rows): raise ValueError("duplicate candidate fingerprints in "+str(path))
        expected = selected_rows(cell)
        if set(indexed) - {r["fingerprint"] for r in expected}: raise ValueError("candidate pool mismatch in "+str(path))
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest() if path and path.exists() else ""
        for index, candidate in enumerate(expected):
            row = indexed.get(candidate["fingerprint"], {})
            score = row.get("value", "")
            valid = row.get("status") == "OK" and score not in (None, "") and math.isfinite(float(score)) and 0 <= float(score) <= 1
            method = cell["method"]
            scope = "shard_validation" if method == "SREGNAS" else "native_final_validation" if method == "FastGNAS" else "validation"
            output.append(dict(cell_id=cell["cell_id"], dataset=cell["dataset"] , method=method,
                pool_index=index, fingerprint=candidate["fingerprint"],
                architecture_json=candidate["native_architecture_json" if method == "FastGNAS" else "architecture_json"],
                backend=info.get("backend", "fastgnas-native" if method == "FastGNAS" else "canonical"),
                proxy_val_accuracy=score if valid else "", score_status="measured" if valid else "unmeasured",
                candidate_status=row.get("status", ""), run_status=info.get("status", a["status"]),
                score_scope=scope, validation_split="val", cross_shard_comparable=False if method == "SREGNAS" else "",
                source_run_id=info.get("run_id", ""), source_experiment_id=info.get("experiment_id", ""),
                source_path=str(path) if path else "", source_sha256=source_hash, source_field="value",
                record_type=cell["record_type"], timing_valid=info.get("timing_valid", ""),
                collect_search_garbage=gc_setting(a) if method == "AP" else ""))
    return output


def publish_scores(state, results):
    rows = score_rows(state)
    target = Path(results)/"EXP7_proxy_scores.tsv"
    temporary = target.with_suffix(".scores.tmp")
    fields = list(rows[0]) if rows else ["cell_id", "fingerprint", "proxy_val_accuracy"]
    with temporary.open("w") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(target)
