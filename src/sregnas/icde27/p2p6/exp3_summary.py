
import csv
import fcntl
import json
from pathlib import Path

NAME = "EXP3_profile_summary.tsv"
CANDIDATE_FIELDS = ["run_id", "fingerprint", "cpu_sampling_s", "gpu_compute_host_wall_s", "aggregation", "timing_coverage"]
AP_COVERAGE = "reference training timers; forward+backward+optimizer host wall, not CUDA-event kernel time"
FIELDS = ["run_id", "dataset", "method", "record_type", "scope", "unit", "unit_index", "P", "S",
          "candidate_count", "completed_candidates", "status", "run_status", "timing_valid", "accounting_valid",
          "wall_s", "prep_s", "measured_pool_s", "segment_training_s", "segment_training_pct",
          "migration_wait_s", "migration_wait_pct", "round_validation_s", "round_validation_pct",
          "controller_update_s", "controller_update_pct", "controller_reason", "segment_local_validation_union_s",
          "segment_model_migration_worker_sum_s", "cpu_sampling_s", "cpu_sampling_pct", "gpu_compute_host_wall_s",
          "gpu_compute_pct", "validation_s", "validation_pct", "other_s", "other_pct", "tail_worker",
          "worker_sum_cpu_sampling_s", "worker_sum_gpu_compute_host_wall_s", "worker_sum_validation_s",
          "peak_mem_gib", "peak_rss_bytes", "aggregation", "definition", "reason", "source"]


def json_rows(path):
    if not path.exists(): return []
    result = []
    for line in path.read_text().splitlines():
        try: result.append(json.loads(line))
        except json.JSONDecodeError: continue
    return result


def union_intervals(intervals, start, end):
    merged = []
    for a,b in sorted((max(start,a),min(end,b)) for a,b in intervals if b >= a):
        if b <= a: continue
        if merged and a <= merged[-1][1]: merged[-1][1] = max(merged[-1][1],b)
        else: merged.append([a,b])
    return merged


def seconds(intervals, start, end):
    return sum(b-a for a,b in union_intervals(intervals,start,end))


def spans(rows):
    return [(r["start_perf_s"],r["end_perf_s"]) for r in rows if r.get("end_perf_s") is not None]


def ap_candidate_timing(run_id, candidates):
    rows = []
    for candidate in candidates:
        stats = candidate.get("training",{}).get("timing_stats",{})
        if candidate.get("status") != "OK" or not all(k in stats for k in ("sampling","forward","backward","optimizer_step")): continue
        rows.append(dict(run_id=run_id,fingerprint=candidate["fingerprint"],cpu_sampling_s=stats["sampling"],
            gpu_compute_host_wall_s=sum(stats[k] for k in ("forward","backward","optimizer_step")),
            aggregation="single assigned worker",timing_coverage=AP_COVERAGE))
    return rows


def common(info, cell, out):
    return dict(run_id=info.get("run_id",out.name), dataset=cell["dataset"], method=cell["method"],
        record_type="profile", scope="fixed_pool_no_RL" if cell["kind"]=="replay" else "full_retrain",
        P=cell["P"],S=info.get("S"),run_status=info.get("status"),timing_valid=info.get("timing_valid",False),
        prep_s=info.get("prep_s"),measured_pool_s=info.get("pool_s"),peak_mem_gib=info.get("worst_card_peak_gib"),
        peak_rss_bytes=info.get("peak_rss_bytes"),source=str(out))


def boundaries(events, kind, index):
    done={r[index]:r for r in events if r["kind"]==kind}
    starts={r[index]:r for r in events if r["kind"]==kind+"_start"}
    return [done[i] if i in done else dict(starts[i],status="incomplete") for i in sorted(starts.keys()|done.keys())]


def native_rows(out, info, cell, events, candidates):
    rows=[];base=common(info,cell,out)
    for unit in boundaries(events,"native_round","round_index"):
        fingerprints=unit["fingerprints"]; selected=[r for r in candidates if r["fingerprint"] in fingerprints]
        row=dict(base,unit="round",unit_index=unit["round_index"],candidate_count=len(fingerprints),
            completed_candidates=sum(r.get("status")=="OK" for r in selected),
            status=unit["status"],accounting_valid=False,controller_update_s="N/A",controller_update_pct="N/A",
            controller_reason="fixed candidate replay has no RL controller update",
            aggregation="union of task intervals on the same host; no sum over parallel workers",
            definition="round=submit one candidate wave through run_training return; segment training=original training start through local validation return, excluding model setup/load/export; migration/wait=residual incl model setup/load/export, scheduling, transfer, synchronization, profiler export and result overhead")
        rows.append(row)
        if unit.get("end_perf_s") is None:
            row["reason"]="round ended without a closing event";continue
        start,end=unit["start_perf_s"],unit["end_perf_s"];wall=end-start;row["wall_s"]=wall
        tasks=[r for r in events if r["kind"]=="native_segment_execution" and r["fingerprint"] in fingerprints
               and r.get("end_perf_s") is not None and r["start_perf_s"]>=start and r["end_perf_s"]<=end]
        training=[r for r in events if r["kind"]=="native_segment_training" and r["fingerprint"] in fingerprints
                  and r.get("end_perf_s") is not None and r["start_perf_s"]>=start and r["end_perf_s"]<=end]
        validation=[r for r in events if r["kind"]=="native_round_validation" and r["round_index"]==unit["round_index"]]
        schedules=[r for r in events if r["kind"]=="native_schedule" and r["round_index"]==unit["round_index"]]
        local=[r for r in events if r["kind"]=="native_local_validation" and r["fingerprint"] in fingerprints]
        counters=[r for r in events if r["kind"]=="native_segment_counters" and r["fingerprint"] in fingerprints]
        train_s=seconds(spans(training),start,end);val_s=seconds(spans(validation),start,end)
        combined=seconds(spans(training)+spans(validation),start,end)
        ok=(wall>0 and unit["status"]=="OK" and len(schedules)==1 and schedules[0]["status"]=="OK"
            and len(selected)==len(fingerprints) and all(r.get("status")=="OK" for r in selected)
            and len(tasks)==len(fingerprints)*int(info["S"]) and all(r["status"]=="OK" for r in tasks)
            and len(training)==len(tasks) and all(r["status"]=="OK" for r in training)
            and (not unit["global_validation_enabled"] or len(validation)==len(fingerprints))
            and all(r["status"]=="OK" for r in validation) and abs(train_s+val_s-combined)<1e-6)
        row.update(accounting_valid=ok,segment_training_s=train_s if training else None,
            round_validation_s=val_s if validation or not unit["global_validation_enabled"] else None,
            segment_local_validation_union_s=seconds(spans(local),start,end) if local else None,
            segment_model_migration_worker_sum_s=sum(r["counters"]["model_migration"] for r in counters) if counters else None)
        if ok:
            row["migration_wait_s"]=wall-combined
            if info.get("timing_valid"):
                row.update(segment_training_pct=100*train_s/wall,round_validation_pct=100*val_s/wall,
                           migration_wait_pct=100*(wall-combined)/wall)
        else:
            row.update(status="incomplete",reason="missing/failed task or validation evidence, or overlapping phase intervals")
    return rows


def ap_rows(out, info, cell, events, candidates):
    rows=[];base=common(info,cell,out)
    for unit in boundaries(events,"ap_wave","wave_index"):
        fingerprints=unit["fingerprints"];selected=[r for r in candidates if r["fingerprint"] in fingerprints]
        row=dict(base,unit="wave",unit_index=unit["wave_index"],candidate_count=len(fingerprints),
            completed_candidates=sum(r.get("status")=="OK" for r in selected),status=unit["status"],accounting_valid=False,
            aggregation="last-finishing worker's reference counters / measured wave wall; worker sums separately labelled",
            definition=AP_COVERAGE+"; validation=evaluation.eval_time; other=dispatch/start offset, model setup, transfers, cleanup and profiling overhead")
        rows.append(row)
        if unit.get("end_perf_s") is None:
            row["reason"]="wave ended without a closing event";continue
        wall=unit["end_perf_s"]-unit["start_perf_s"];row["wall_s"]=wall
        ok=(wall>0 and unit["status"]=="OK" and len(selected)==len(fingerprints)
            and all(r.get("status")=="OK" and r.get("exp3_end_perf_s") is not None for r in selected))
        reference=ap_candidate_timing(base["run_id"],selected)
        ok=ok and len(reference)==len(selected) and all(r.get("evaluation",{}).get("eval_time") is not None for r in selected)
        if not ok:
            row.update(status="incomplete",reason="missing/failed candidate counters or worker end events");continue
        tail=max(selected,key=lambda r:r["exp3_end_perf_s"])
        timer=next(r for r in reference if r["fingerprint"]==tail["fingerprint"])
        sampling=timer["cpu_sampling_s"];compute=timer["gpu_compute_host_wall_s"];validation=tail["evaluation"]["eval_time"]
        other=wall-sampling-compute-validation
        ok=other>=-1e-6 and all(unit["start_perf_s"]<=r["exp3_start_perf_s"]<=r["exp3_end_perf_s"]<=unit["end_perf_s"] for r in selected)
        row.update(accounting_valid=ok,tail_worker=tail["worker_id"],cpu_sampling_s=sampling,
            gpu_compute_host_wall_s=compute,validation_s=validation,other_s=other,
            worker_sum_cpu_sampling_s=sum(r["cpu_sampling_s"] for r in reference),
            worker_sum_gpu_compute_host_wall_s=sum(r["gpu_compute_host_wall_s"] for r in reference),
            worker_sum_validation_s=sum(r["evaluation"]["eval_time"] for r in selected))
        if ok and info.get("timing_valid"):
            row.update(cpu_sampling_pct=100*sampling/wall,gpu_compute_pct=100*compute/wall,
                       validation_pct=100*validation/wall,other_pct=100*other/wall)
        elif not ok:row.update(status="incomplete",reason="reference counters exceed wave wall or worker interval is outside wave")
    return rows


def write_tsv(path, rows, fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    with temporary.open("w") as stream:
        writer=csv.DictWriter(stream,fields,delimiter="\t",extrasaction="ignore",lineterminator="\n")
        writer.writeheader();writer.writerows(rows)
    temporary.replace(path)


def aggregate(result_root):
    pass
    result_root=Path(result_root);result_root.mkdir(parents=True,exist_ok=True)
    with (result_root/"EXP3_profile_summary.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        rows=[]
        for path in sorted((result_root/"EXP3").glob("*/"+NAME)):
            with path.open() as stream:rows.extend(csv.DictReader(stream,delimiter="\t"))
        write_tsv(result_root/NAME,rows,FIELDS)


def publish(out, info, cell):
    out=Path(out);events=[r for p in out.glob("exp3_events_*.jsonl") for r in json_rows(p)]
    candidates=info.get("rows",[])
    detailed=cell["kind"]=="replay" and cell["method"] in ("FastGNAS","AP")
    if detailed:
        function=native_rows if cell["method"]=="FastGNAS" else ap_rows
        rows=function(out,info,cell,events,candidates)
        divisor=int(info["S"]) if cell["method"]=="FastGNAS" else cell["P"]
        valid=len(rows)==(cell["count"]+divisor-1)//divisor and all(r["accounting_valid"] for r in rows)
        info.setdefault("assertions",{})["exp3_component_accounting"]=valid
        if info["status"]=="OK" and not valid:info["status"]="incomplete"
    else:rows=[]
    if not rows:
        rows=[dict(common(info,cell,out),unit="run",unit_index=1,candidate_count=cell["count"],
            completed_candidates=info.get("completed_candidates"),status=info["status"],
            wall_s=info.get("pool_s"),accounting_valid=False if detailed else "N/A",
            reason="no complete round/wave boundaries" if detailed else "round/wave breakdown applies to FastGNAS/AP replay; full retrain and other methods keep existing profiler artifacts")]
    for row in rows:row["run_status"]=info["status"]
    if cell["method"]=="AP" and cell["kind"]=="replay":
        write_tsv(out/"candidate_timing.tsv",ap_candidate_timing(info.get("run_id",out.name),candidates),CANDIDATE_FIELDS)
    write_tsv(out/NAME,rows,FIELDS)
    info["exp3_profile_summary"]=str(out/NAME)
    aggregate(out.parent.parent)
