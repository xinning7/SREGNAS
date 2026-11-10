

import argparse
import json
import logging
from pathlib import Path
import tempfile
import time

import ray
import torch
from torch_geometric.data import Data

from .common import METHODS, dataset_config, failure, result_row
from .search import RecordSink, central_search, load_pool


class Executor:
    def __init__(self, method, data, config, workers, *, setup_record_dir=None):
        self.method = method
        self.dataset = config.get("dataset")
        self.workers = []
        if method == "GraphNAP":
            from .adapter import Worker, Backend, partitions_from_data

            partitions = partitions_from_data(data, workers)
            actor = ray.remote(num_gpus=1, num_cpus=1, max_restarts=0, max_task_retries=0)(Worker)
            self.workers = [actor.remote(i, part, torch.device("cuda:0"), config)
                            for i, part in enumerate(partitions)]
            self.backend = Backend(self.workers, config)
        else:
            if method == "AP":
                from sregnas.baselines.ap.adapter import Worker
            else:
                from sregnas.baselines.paspl.adapter import Worker
                if setup_record_dir is not None or config.get('collect_search_garbage', False):
                    from sregnas.icde27.x15.observations import ObservedPreloadWorker as Worker
            actor = ray.remote(num_gpus=1, num_cpus=1, max_restarts=0, max_task_retries=0)(Worker)
            graph_ref = ray.put(data)
            self.workers = [actor.remote(i, graph_ref, True, config) for i in range(workers)]

        if setup_record_dir is None:
            self.initial_metrics = ray.get([w.metrics.remote() for w in self.workers])
        else:
            from sregnas.icde27.x15.observations import record_preloads
            self.initial_metrics = record_preloads(self.workers, setup_record_dir)

    def evaluate(self, specs, sink):
        if getattr(self, "dataset", None):
            from sregnas.icde27.space import validate_pair
            for spec in specs:
                validate_pair(spec["layers"], self.dataset)
        results = []
        if self.method == "GraphNAP":
            for spec in specs:
                budget = getattr(self, "search_budget", None)
                if budget and not budget.allowance(sink, 1):
                    break
                index = sink.start(spec)
                row = self.backend.run_candidate(spec)
                sink.finish(index, row)
                results.append(row)
        else:


            for offset in range(0, len(specs), len(self.workers)):
                budget = getattr(self, "search_budget", None)
                count = budget.allowance(sink, len(self.workers)) if budget else len(self.workers)
                if not count:
                    break
                wave = specs[offset:offset + count]
                wave_started = time.perf_counter()
                jobs = []
                for worker, spec in zip(self.workers, wave):
                    index = sink.start(spec)
                    jobs.append((index, spec, worker.run_candidate.remote(spec)))
                for index, spec, future in jobs:
                    try:
                        row = ray.get(future)
                    except Exception as exc:
                        row = result_row(spec)
                        row.update(status=failure(exc), failure_stage="actor",
                                   error=f"{type(exc).__name__}: {exc}")
                    sink.finish(index, row)
                    results.append(row)
                from sregnas.icde27 import ledger
                ledger.event("search_wave_end", first_attempt=jobs[0][0], candidate_count=len(jobs),
                             wave_start_perf_s=wave_started, worker_rss_bytes=[r.get("host_rss_bytes") for r in results[-len(jobs):]])
        return results


def prepare_graph(cfg, out):
    import os
    if os.environ.get('SREGNAS_X22_CANONICAL_INPUT')=='1':
        from sregnas.icde27.x22.canonical_input import port_graph
        return port_graph(cfg,out)
    if cfg.graph.dataset in {"reddit", "ogbn-products"}:
        import numpy as np
        from sregnas.data.products import ProductsMmapGraph
        from sregnas.paths import DATA_DIR
        relative = ("Reddit/sregnas_cache_v1/full/manifest.json" if cfg.graph.dataset == "reddit"
                    else "ogbn_products/sregnas_comparison_v1/full/manifest.json")
        manifest = DATA_DIR / relative


        graph = ProductsMmapGraph.open(manifest)
        nodes = np.arange(graph.num_nodes, dtype=np.int64)
        edge = torch.empty((2, graph.col.size), dtype=torch.int64)
        edge[0] = torch.from_numpy(np.array(graph.col, dtype=np.int64))
        edge[1] = torch.from_numpy(np.repeat(nodes, np.diff(graph.rowptr)))
        tensors = {k: torch.from_numpy(np.array(getattr(graph, k), copy=True))
                   for k in ("x", "labels", "train_mask", "val_mask", "test_mask")}
        data = Data(x=tensors.pop("x"), y=tensors.pop("labels"), edge_index=edge, **tensors)
        return data.contiguous(), graph.num_classes, str(manifest)



    from sregnas.comparison.search import prepare_central_graph

    graph = prepare_central_graph(cfg, "BaselineFull", out)
    raw = torch.load(graph["full_graph_path"], map_location="cpu", weights_only=False)
    data = Data(x=raw["x"], y=raw["labels"].reshape(-1), edge_index=raw["edge_index"],
                **{s + "_mask": raw[s + "_mask"] for s in ("train", "val", "test")})



    return data.contiguous(), raw["num_classes"], graph["retrain_graph_path"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args(argv)
    from sregnas.comparison.runtime import configure_native_client
    from sregnas.config import ExperimentConfig, _strict_dataclass
    from sregnas.execution.campaign_timing import mark, record
    from sregnas.icde27 import ledger

    configure_native_client()
    logging.basicConfig(level=logging.INFO)
    job = json.loads(args.job.read_text())
    cfg = _strict_dataclass(ExperimentConfig, job["config"], "config")
    cfg.validate()
    out = Path(job["out"])
    method = job["method"]
    paspl_full_nas = bool(job.get("paspl_full_nas"))
    if method not in METHODS or (method == "PAS-PL" and job["mode"] != "fixed_pool" and not paspl_full_nas):
        raise ValueError("unsupported port/mode")
    if paspl_full_nas:
        from sregnas.comparison.ports import validate_port_request
        validate_port_request(cfg, method, job["mode"], None, paspl_full_nas=True)
    if job.get("from_search_record"):
        from .truncation import run_from_search_record
        run_from_search_record(cfg, method, job["from_search_record"], job["search_limit"], out, job.get("shortlist_size"))
        return 0
    started = time.perf_counter()
    stage = "graph_load"
    sink = RecordSink(out, method, job["mode"])
    try:
        ledger.start_clock("P0-B before reading canonical local graph")
        data, classes, graph_path = prepare_graph(cfg, out)
        config = dataset_config(cfg.graph.dataset, classes)
        if job.get("x16_paspl_fanout5"):
            if method != "PAS-PL" or not paspl_full_nas:
                raise ValueError("X16 fanout override is only authorized for opt-in PAS-PL full_nas")
            config["neighbor_sizes"] = config["training"]["fan_out"] = [5, 5]
        config["dataset"] = cfg.graph.dataset
        if job.get('cached_sampler'):
            config['cached_sampler'] = True
        config["collect_search_garbage"] = bool(job.get("collect_search_garbage") or (
            method == 'PAS-PL' and job.get("paspl_collect_search_garbage")))
        stage = "partition_cache" if method == "GraphNAP" else "graph_preload" if method == "PAS-PL" else "cpu_graph_setup"
        n = len(cfg.runtime.baseline_gpus)
        ray.init(num_gpus=n, num_cpus=n, include_dashboard=False,
                 _temp_dir=tempfile.mkdtemp(prefix="p0b-ray-", dir="/tmp"))
        probe_attempt = None
        if job.get("capacity_probe"):
            probe_spec = load_pool(job["pool"], 1)[0]
            probe_attempt = ledger.launch("probe", stage="capacity", identity="PAS-PL:capacity",
                                          fingerprint=probe_spec["fingerprint"], backend="canonical", mode="capacity")
        executor = Executor(method, data, config, n,
                            setup_record_dir=out if paspl_full_nas else None)
        if probe_attempt:
            stage = "capacity_train_step"
            probe = ray.get(executor.workers[0].capacity_probe.remote(probe_spec))
            (out / "capacity_probe.json").write_text(json.dumps(probe, indent=2))
            ledger.event("capacity_probe_result", result=probe, launch_id=probe_attempt["launch_id"])
            if probe["status"] != "OK":
                raise RuntimeError(f"capacity probe failed: {probe}")
        (out / "port_setup.json").write_text(json.dumps(dict(
            method=method, status="OK", config=config, workers=executor.initial_metrics,
            search_feature_residency=("whole_graph_preload" if paspl_full_nas else
                                      "cpu" if method == "AP" else "gpu"),
            retrain_feature_residency="gpu", model_backend="canonical",
            input_graph_layout=dict(edge_index_contiguous=data.edge_index.is_contiguous(),
                                    feature_contiguous=data.x.is_contiguous(),
                                    edge_index_stride=list(data.edge_index.stride())),
            partition_hops=2 if method == "GraphNAP" else None,
        ), indent=2))
        record(out, cpu_preparation_s=time.perf_counter() - started)
        mark(out, "search_started")
        stage = "search"
        if job["mode"] == "fixed_pool":
            specs = load_pool(job["pool"], job.get("pool_limit"))
            executor.evaluate(specs, sink)
        else:
            central_search(cfg.search, executor, sink, max_search_candidates=job.get("max_search_candidates"),
                           max_search_seconds=job.get("max_search_seconds"), paspl_full_nas=paspl_full_nas)
        mark(out, "search_finished")
        sink.finalize_records()
        (out / "retrain_graph.json").write_text(json.dumps(dict(graph_path=graph_path)))
        (out / "search_accounting.json").write_text(json.dumps(dict(
            method=method, q=len(sink.rows),
            Q_search=len(sink.rows) if job["mode"] == "full_nas" else None,
            Qbench=len(sink.rows) if job["mode"] == "fixed_pool" else None,
            Q_trunc=None, successful=sum(r["status"] == "OK" for r in sink.rows),
            failed=sum(r["status"] != "OK" for r in sink.rows),
            failures_charged_to_budget=True, attempts_complete=True,
            status="OK" if all(r["status"] == "OK" for r in sink.rows) else "incomplete",
        ), indent=2))
        if __import__('os').environ.get('SREGNAS_X22_INCREMENTAL_AUDIT')=='1' or __import__('os').environ.get('SREGNAS_X22_DEFER_RECORDS')=='1':
            from sregnas.icde27.x22.ray_finish import close_workers
            close_workers(executor.workers)
        ray.shutdown()
        del executor, data
        from sregnas.comparison.search import ranked_candidates
        ranked = ranked_candidates([dict(r, architecture_json=json.dumps(r["architecture"])) for r in sink.rows], method)
        (out / "ranked_candidates.json").write_text(json.dumps(ranked, indent=2))
        if job["mode"] == "full_nas" and job.get("stage") == "complete":
            stage = "retrain"
            from sregnas.comparison.launch import common_retrain
            common_retrain(cfg, method, ranked, graph_path, out, shortlist_size=job.get("shortlist_size"))
        else:
            ledger.event("e2e_end", source="fixed pool or search-only returned")
            ledger.stop_clock()
    except Exception as exc:
        (out / "port_failure.json").write_text(json.dumps(dict(
            status=failure(exc), failure_stage=stage, error=f"{type(exc).__name__}: {exc}",
            attempted_candidates=len(sink.rows),
        ), indent=2))
        raise
    finally:
        sink.finalize_records()
        try:
            if ray.is_initialized() and (__import__('os').environ.get('SREGNAS_X22_INCREMENTAL_AUDIT')=='1' or __import__('os').environ.get('SREGNAS_X22_DEFER_RECORDS')=='1'):
                from sregnas.icde27.x22.ray_finish import close_workers
                close_workers(getattr(locals().get('executor'),'workers',[]))
        finally:
            ray.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
