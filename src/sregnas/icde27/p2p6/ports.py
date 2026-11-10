
import json
import os
import tempfile
import time

from sregnas.icde27 import ledger
from sregnas.icde27.recorder import dump
from .plan import selected_rows, logical_shards, apply_amendments
from .resources import mark_stage
from .profiling import capture, worker_tail
from .exp3_hooks import scope, enabled as fine_profile_enabled


def execute(cell, cfg, out):
    import torch
    import ray
    from sregnas.baselines.graphnap.runtime import prepare_graph, Executor
    from sregnas.baselines.graphnap.common import dataset_config, candidate
    from sregnas.baselines.graphnap.search import RecordSink
    from .port_workers import GraphWorker, APWorker, PASWorker
    defer_progress = os.environ.get('SREGNAS_X23_POOL_PROGRESS') == '1'
    ledger.start_clock("P2P6 before canonical baseline local graph read")
    mark_stage(out, "prep/graph_load")
    data, classes, _ = prepare_graph(cfg, out)
    config = dataset_config(cfg.graph.dataset, classes)
    config["dataset"] = cfg.graph.dataset
    if apply_amendments(cell).get("collect_search_garbage"):
        config["collect_search_garbage"] = True
    n = logical_shards(cell) if cell["method"] == "GraphNAP" else cell["P"]
    ray.init(num_gpus=cell["P"], num_cpus=max(n, cell["P"]), include_dashboard=False,
             _temp_dir=tempfile.mkdtemp(prefix="p2p6-ray-", dir="/tmp"))
    try:
        executor = Executor.__new__(Executor)
        executor.method, executor.dataset = cell["method"], cfg.graph.dataset
        if cell["method"] == "GraphNAP":
            mark_stage(out, "prep/partition_cache")
            from sregnas.baselines.graphnap.adapter import partitions_from_data, Backend
            parts = partitions_from_data(data, n)
            actor = ray.remote(num_gpus=cell["P"]/n, num_cpus=1, max_restarts=0, max_task_retries=0)(GraphWorker)
            executor.workers = [actor.remote(i, part, torch.device("cuda:0"), config) for i,part in enumerate(parts)]
            executor.backend = Backend(executor.workers, config)
        else:
            mark_stage(out, "prep/host_graph_setup" if cell["method"] == "AP" else "prep/graph_preload")
            cls = APWorker if cell["method"] == "AP" else PASWorker
            if cell["method"] == "AP" and fine_profile_enabled():
                from .exp3_workers import Exp3APWorker
                cls = Exp3APWorker
            actor = ray.remote(num_gpus=1, num_cpus=1, max_restarts=0, max_task_retries=0)(cls)
            graph_ref = ray.put(data)
            executor.workers = [actor.remote(i, graph_ref, True, config) for i in range(n)]
        metrics = ray.get([w.metrics.remote() for w in executor.workers])
        placements = ray.get([w.placement.remote() for w in executor.workers])
        dump(out/"worker_placement.json", placements)
        dump(out/"port_setup.json", dict(config=config, workers=metrics, placements=placements))
        assert len(placements) == n
        assert len({g for place in placements for g in place["gpu_ids"]}) == cell["P"], "Ray placement must use all reserved GPUs"
        selected = selected_rows(cell)
        specs = [candidate(json.loads(r["architecture_json"]), r["fingerprint"]) for r in selected]
        if cell.get("capacity_probe"):
            mark_stage(out, "prep/capacity_probe")

            largest = max(selected, key=lambda r: (int(r["canonical_parameters"]), r["fingerprint"]))
            probe_spec = candidate(json.loads(largest["architecture_json"]), largest["fingerprint"])
            attempt = ledger.launch("probe", stage="capacity", identity="PAS-PL:capacity", fingerprint=largest["fingerprint"])
            probe = ray.get(executor.workers[0].capacity_probe.remote(probe_spec))
            dump(out/"capacity_probe.json", probe)
            ledger.event("capacity_probe_result", result=probe, launch_id=attempt["launch_id"])
            if probe["status"] != "OK": raise RuntimeError(f"capacity probe failed: {probe}")
        sink = RecordSink(out, cell["method"], "fixed_pool")
        mark_stage(out, "pool/candidate_train_and_validation")
        ledger.event("pool_start")
        started = time.perf_counter()

        if cell["method"] == "GraphNAP":
            rows = []
            for spec in specs:
                rows.extend(executor.evaluate([spec], sink))
                if not defer_progress: dump(out/"partial_candidates.json", rows)
                if rows[-1].get("failure_stage") == "actor" and rows[-1]["status"] != "OK": break
                ray.get([w.finish_profile.remote() for w in executor.workers])
        else:
            rows = []
            for offset in range(0, len(specs), n):
                with scope("ap_wave" if cell["method"] == "AP" else "parallel_wave",
                           wave_index=offset//n+1, fingerprints=[s["fingerprint"] for s in specs[offset:offset+n]]):
                    wave = executor.evaluate(specs[offset:offset+n], sink)
                rows.extend(wave)
                if not defer_progress: dump(out/"partial_candidates.json", rows)
                if any(r.get("failure_stage") == "actor" and r["status"] != "OK" for r in wave): break
        ended = time.perf_counter()
        ledger.event("pool_end"); ledger.stop_clock()
        if defer_progress: dump(out/"partial_candidates.json", rows)
        sink.finalize_records()
        spans = []
        for worker in executor.workers:
            try: spans.extend(ray.get(worker.spans.remote()))
            except ray.exceptions.RayError: pass
        dump(out/"worker_tails.json", worker_tail(spans, started, ended))
        communication = [dict(fingerprint=r["fingerprint"], category="parameter_average_input",
            bytes=r.get("parameter_average_bytes"), count=r.get("parameter_average_count"),
            definition="logical state tensor payload; transport framing and broadcast not included") for r in rows]
        dump(out/"communication.json", communication)
        return dict(rows=rows, controller_updates=0, worker_placements=placements,
            assertions=dict(worker_count_fixed=len(executor.workers)==n,
                fixed_optimizer_steps=all(r.get("optimizer_steps")==50 for r in rows),
                graphnap_same_initialization=all(len({w["initialization_hash"] for w in r.get("workers", [])})==1 for r in rows) if cell["method"]=="GraphNAP" else True,
                all_graphnap_replicas_50_steps=all(all(w["optimizer_steps"]==50 for w in r.get("workers", [])) for r in rows)))
    finally:
        if 'sink' in locals(): sink.finalize_records()
        try:
            if os.environ.get('SREGNAS_X22_INCREMENTAL_AUDIT')=='1' or os.environ.get('SREGNAS_X22_DEFER_RECORDS')=='1':
                from sregnas.icde27.x22.ray_finish import close_workers
                close_workers(getattr(locals().get('executor'),'workers',[]))
        finally:
            ray.shutdown()
