

from copy import deepcopy
import hashlib
import math
import json
import os
from pathlib import Path

from sregnas.execution.provenance import commit_hash


def fingerprint(architecture):
    return hashlib.sha256(
        json.dumps(architecture, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def ranked_candidates(rows, method):
    from sregnas.execution.provenance import commit_hash

    if method == "FastGNAS":
        candidates = [
            dict(
                id=fingerprint(r["arch_spec"]),
                architecture=r["arch_spec"],
                value=float(r["best_accuracy"]),
            )
            for r in rows.values()
            if r.get("best_round") is not None and r.get("status", "OK") == "OK"
        ]
    else:
        candidates = [
            dict(
                id=r["fingerprint"],
                architecture=json.loads(r["architecture_json"]),
                value=float(r["value"]),
            )
            for r in rows
            if r["status"] == "OK" and r["value"] is not None
        ]
    if any(
        not math.isfinite(row["value"]) or not 0 <= row["value"] <= 1
        for row in candidates
    ):
        raise AssertionError("nonfinite or out-of-range search validation result")
    unique = {}
    for row in sorted(candidates, key=lambda r: (-r["value"], r["id"])):
        row["commit_hash"] = commit_hash()
        unique.setdefault(row["id"], row)
    return list(unique.values())


def prepare_central_graph(cfg, method, out):
    from sregnas.data import load_dataset
    from sregnas.data.graph import build_uniform_sampled_full_graph
    from sregnas.paths import DATA_DIR
    import torch

    if cfg.graph.dataset == "reddit":
        from sregnas.data.reddit import prepare_reddit_graph, prepare_baseline_payload

        prepared = prepare_reddit_graph()
        search_path = str(prepare_baseline_payload(sampled=method == "BaselineFair"))
        full_path = prepared["graph_manifest"]
    elif cfg.graph.dataset == "ogbn-products":
        from .products_protocol import confirmed_protocol
        from sregnas.data.products_comparison import (
            prepare_products_graph,
            baseline_payload,
        )

        confirmed_protocol(method)
        full_path = prepare_products_graph()
        search_path = baseline_payload(
            full_path,
            Path(full_path).parent.parent / "baselines",
            sampled=method == "BaselineFair",
        )
    else:
        dataset = load_dataset(
            cfg.graph.dataset, data_dir=str(DATA_DIR), normalize=False, preprocess=False
        )
        full_path = str(out / "full_graph.pt")
        torch.save(dataset, full_path)
        if method == "BaselineFair":
            prepared = build_uniform_sampled_full_graph(
                cfg.graph.dataset, out / "sampled"
            )
            search_path = prepared["payload_path"]
        else:
            search_path = full_path
    return dict(
        dataset=cfg.graph.dataset,
        num_shards=cfg.graph.num_gpus,
        shards=[],
        full_graph_path=search_path,
        retrain_graph_path=full_path,
        partition_strategy="none",
        baseline_method=method,
    )


def run_central_search(cfg, method, out):
    from sregnas.execution.pool import PersistentWorkerPool
    from sregnas.execution.queue import TaskQueue
    from sregnas.execution.search import _run_independent_dynamic_baseline_search

    cfg = deepcopy(cfg)
    cfg.runtime.require_fi_gpu = False

    cfg.runtime.comparison_method = method
    import time
    from sregnas.execution.campaign_timing import mark, record

    preparation_started = time.perf_counter()
    from sregnas.icde27.ledger import start_clock
    start_clock("comparison before prepare_central_graph")
    graph = prepare_central_graph(cfg, method, out)
    record(out, cpu_preparation_s=time.perf_counter() - preparation_started)
    if method == "BaselineFair":
        cfg.runtime.baseline_search_graph = "cost_aligned_uniform_5_5"
        cfg.runtime.baseline_sampled_graph_reference = graph["full_graph_path"]
    else:
        cfg.runtime.baseline_search_graph = "full_neighbour_conventional"
        cfg.runtime.baseline_sampled_graph_reference = None
    cfg.train.retrain_seed_policy = "fingerprint_retrain_replica"
    (out / "resolved_search_config.json").write_text(
        json.dumps(dict(method=method, config=cfg.to_dict(), graph=graph), indent=2)
    )
    db = out / "queue.sqlite"
    TaskQueue(db).close()
    pool = PersistentWorkerPool(out, cfg.runtime.baseline_gpus, cfg.name)
    mark(out, "search_started")
    try:
        info = _run_independent_dynamic_baseline_search(
            cfg=cfg,
            profile_dir=out,
            db_path=db,
            graph_manifest=graph,
            side="baseline",
            worker_pool=pool,
        )
        mark(out, "search_finished")
    finally:
        pool.close()
    queue = TaskQueue(db)
    rows = [
        dict(r) for r in queue.conn.execute("SELECT * FROM tasks ORDER BY task_key")
    ]
    queue.close()
    from sregnas.execution.provenance import commit_hash

    for row in rows:
        row["commit_hash"] = commit_hash()
    (out / "tasks.json").write_text(json.dumps(rows, indent=2))
    if len(rows) != cfg.search.q or any(
        r["status"] not in {"OK", "OOM", "DIVERGE"} for r in rows
    ):
        raise RuntimeError("search did not account for every budgeted attempt")
    (out / "search_accounting.json").write_text(
        json.dumps(
            dict(
                commit_hash=commit_hash(),
                method=method,
                q=cfg.search.q,
                successful=sum(r["status"] == "OK" for r in rows),
                failed=sum(r["status"] != "OK" for r in rows),
                failures_charged_to_budget=True,
                native=info,
            ),
            indent=2,
        )
    )
    return ranked_candidates(rows, method), graph["retrain_graph_path"]


def native_config(cfg, template, out):
    import yaml
    from sregnas.baselines import fastgnas

    root = Path(fastgnas.__file__).parent
    config = yaml.safe_load(Path(template).read_text())
    gpus = cfg.runtime.system_gpus
    n = cfg.graph.num_gpus
    workers = max(n, len(gpus))
    if workers % len(gpus) or workers % n or cfg.search.q % workers:
        raise ValueError("native rings require divisible workers, partitions and Q")
    name = {
        "reddit": "Reddit",
        "flickr": "Flickr",
        "cora": "Cora",
        "citeseer": "CiteSeer",
        "pubmed": "PubMed",
    }.get(cfg.graph.dataset, cfg.graph.dataset)
    config["fastgnas_backends"].update(
        num_workers=n, device_GPU=len(gpus), workers=gpus, dataset=name
    )
    config["fastgnas_backends"]["logging"]["log_dir"] = str(out / "native_logs")
    config["fastgnas_backends"]["experiment"]["random_seed"] = cfg.search.base_seed
    config["nas"].update(
        num_search_rounds=cfg.search.q // workers, samples_per_round=workers
    )
    if cfg.search.token_subset is not None:
        config["nas"]["token_subset"] = cfg.search.token_subset
    config["nas"]["balancing_strategy"]["model_path"] = str(
        root / "mload_checkpoint.pth"
    )
    config["training"].update(root_seed=cfg.search.base_seed, logical_partitions=n)
    return config, name


def run_native_search(cfg, template, out):
    import subprocess
    import sys
    import tempfile
    import yaml

    import time
    from sregnas.execution.campaign_timing import record

    preparation_started = time.perf_counter()
    from sregnas.icde27.ledger import start_clock
    start_clock("comparison before native retrain graph preparation")
    if cfg.graph.dataset == "reddit":
        from sregnas.data.reddit import prepare_reddit_graph

        graph = prepare_reddit_graph()["graph_manifest"]
    elif cfg.graph.dataset == "ogbn-products":
        from sregnas.data.products_comparison import prepare_products_graph

        graph = prepare_products_graph()
    else:
        import torch
        from sregnas.data import load_dataset
        from sregnas.paths import DATA_DIR

        graph = str(out / "full_graph.pt")
        torch.save(
            load_dataset(
                cfg.graph.dataset,
                data_dir=str(DATA_DIR),
                normalize=False,
                preprocess=False,
            ),
            graph,
        )
    record(
        out, cpu_retrain_graph_preparation_s=time.perf_counter() - preparation_started
    )
    config, dataset = native_config(cfg, template, out)
    path = out / "native_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    (out / "phase.txt").write_text("preparation")
    env = dict(
        os.environ,
        TIMING_RUN_DIR=str(out),
        TIMING_RAY_TMP=tempfile.mkdtemp(prefix="sg-ray-"),
        CUDA_VISIBLE_DEVICES=",".join(map(str, cfg.runtime.system_gpus)),
    )
    if cfg.graph.dataset == "ogbn-products":
        from .products_protocol import DIRECTIVE, assert_native_products_config
        from sregnas.baselines.fastgnas.search import merge_dataset_config

        assert_native_products_config(merge_dataset_config(deepcopy(config), dataset))
        env["SREGNAS_PRODUCTS_PROTOCOL"] = str(DIRECTIVE)
    env.pop("RAY_ADDRESS", None)
    env.pop("SIX_CAMPAIGN_HANDOFF", None)
    command = [
        sys.executable,
        "-m",
        "sregnas.baselines.fastgnas.search",
        "--config",
        str(path),
        "--dataset",
        dataset,
    ]
    (out / "native_launch.json").write_text(
        json.dumps(dict(command=command, config=config), indent=2)
    )
    with (out / "native_search.log").open("w") as log:
        subprocess.run(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    events = [
        json.loads(line)
        for file in out.glob("events_*.jsonl")
        for line in file.read_text().splitlines()
    ]
    if sum(r["event"] == "search_end" for r in events) != 1 or any(
        r["event"] in {"run_failed", "partition_task_failed"} for r in events
    ):
        raise RuntimeError("native search failed; inspect native_search.log and events")
    start = next(r for r in events if r["event"] == "search_start")
    end = next(r for r in events if r["event"] == "search_end")
    record(
        out,
        search_started_perf_s=start["perf_s"],
        search_started_epoch_s=start["epoch_s"],
        search_finished_perf_s=end["perf_s"],
        search_finished_epoch_s=end["epoch_s"],
        native_cpu_partition_s=end["partition_s"],
        native_gpu_warmup_s=end["warmup_s"],
    )
    done = [
        r
        for r in events
        if r["event"] == "partition_task_done" and r["phase"] == "search"
    ]
    terminal = [
        r
        for r in events
        if r["event"]
        in {"partition_task_done", "partition_task_oom", "partition_task_oom_skip"}
        and r["phase"] == "search"
    ]
    groups = {}
    for row in terminal:
        groups.setdefault(row["arch_id"], []).append(row)
    if len(groups) != cfg.search.q or any(
        len(rows) != cfg.graph.num_gpus for rows in groups.values()
    ):
        raise RuntimeError("native search Q/ring accounting failed")
    failed = {
        r["arch_id"]
        for r in events
        if r.get("phase") == "search"
        and r["event"] in {"partition_task_oom", "candidate_validation_oom"}
    }
    from collections import Counter

    histogram = Counter(
        sum(row.get("optimizer_steps", 0) for row in rows)
        for arch, rows in groups.items()
        if arch not in failed
    )
    (out / "search_accounting.json").write_text(
        json.dumps(
            dict(
                commit_hash=commit_hash(),
                method="FastGNAS",
                q=len(groups),
                successful=len(groups) - len(failed),
                failed=len(failed),
                oom_architectures=sorted(failed),
                failures_charged_to_budget=True,
                optimizer_steps_histogram=dict(histogram),
                warmup_tasks=sum(
                    r["event"] == "partition_task_done" and r["phase"] == "warmup"
                    for r in events
                ),
            ),
            indent=2,
        )
    )
    rows = json.loads((out / "search_candidates.json").read_text())
    return ranked_candidates(rows, "FastGNAS"), graph
