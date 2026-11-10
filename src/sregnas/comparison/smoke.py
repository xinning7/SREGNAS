

import json


def validate_smoke(cfg, stage):
    if (
        stage,
        cfg.graph.num_gpus,
        cfg.search.q,
        cfg.search.m,
        cfg.fixed_x,
        cfg.selection.k,
        cfg.train.proxy_epochs,
    ) != ("search", 2, 16, 16, 8, 4, 100):
        raise ValueError("smoke is locked to search/N2/Q16/M16/B8/k4/E100")
    if len(cfg.runtime.system_gpus) != 2 or len(cfg.runtime.baseline_gpus) != 2:
        raise ValueError("N2 smoke requires two physical GPUs")
    if (
        not cfg.runtime.require_idle_preflight
        or not cfg.runtime.require_native_topology
    ):
        raise ValueError("smoke requires idle preflight and native topology")


def run_sharded_smoke(cfg, out):
    from sregnas.execution.orchestrator import _ensure_graph
    from sregnas.execution.pool import PersistentWorkerPool
    from sregnas.execution.queue import TaskQueue
    from sregnas.execution.search import _run_independent_bound_system_search
    from .search import ranked_candidates

    validate_smoke(cfg, "search")
    db = out / "queue.sqlite"
    TaskQueue(db).close()
    import time
    from sregnas.execution.campaign_timing import mark, record

    preparation_started = time.perf_counter()
    graph = _ensure_graph(cfg, out, db)
    record(
        out,
        cpu_preparation_s=time.perf_counter() - preparation_started,
        formal_timing=False,
    )
    pool = PersistentWorkerPool(out, cfg.runtime.system_gpus, cfg.name)
    mark(out, "search_started")
    try:
        info = _run_independent_bound_system_search(
            cfg=cfg,
            profile_dir=out,
            db_path=db,
            graph_manifest=graph,
            side="system",
            worker_pool=pool,
        )
        mark(out, "search_finished")
    finally:
        pool.close()
    queue = TaskQueue(db)
    rows = [
        dict(row) for row in queue.conn.execute("SELECT * FROM tasks ORDER BY task_key")
    ]
    queue.close()
    from sregnas.execution.provenance import commit_hash

    for row in rows:
        row["commit_hash"] = commit_hash()
    (out / "tasks.json").write_text(json.dumps(rows, indent=2))
    if len(rows) != 16 or any(
        row["status"] not in {"OK", "OOM", "DIVERGE"} for row in rows
    ):
        raise RuntimeError("smoke did not account for all 16 search attempts")
    (out / "search_accounting.json").write_text(
        json.dumps(
            dict(
                commit_hash=commit_hash(),
                q=16,
                successful=sum(row["status"] == "OK" for row in rows),
                failed=sum(row["status"] != "OK" for row in rows),
                search_only=True,
                test_evaluations=0,
                formal_timing=False,
                native=info,
            ),
            indent=2,
        )
    )
    return ranked_candidates(rows, "SREGNAS"), graph.get(
        "retrain_graph_path", graph.get("full_graph_path")
    )
