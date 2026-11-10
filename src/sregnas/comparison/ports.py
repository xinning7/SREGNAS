

import json
import os
import subprocess
import sys

METHODS = ("GraphNAP", "AP", "PAS-PL")


def validate_port_request(cfg, method, mode, pool, *, paspl_full_nas=False):
    from sregnas.execution.options import resolve_executor

    if method not in METHODS:
        raise ValueError("unknown port")
    if paspl_full_nas and (method != "PAS-PL" or mode != "full_nas"):
        raise ValueError("paspl-full-nas requires PAS-PL full_nas")
    if mode == "fixed_pool":
        if pool is None:
            raise ValueError("fixed_pool requires --pool JSON candidate definitions")
        from sregnas.baselines.graphnap.search import load_pool
        load_pool(pool)
    elif pool is not None or (method == "PAS-PL" and not paspl_full_nas):
        raise ValueError("PAS-PL only supports fixed_pool; --pool requires fixed_pool")
    if cfg.graph.dataset not in {"flickr", "ogbn-arxiv", "reddit", "ogbn-products"}:
        raise ValueError("P0-B supports the four protocol datasets")
    from sregnas.icde27.x_authorization import seed_allowed
    seed_ok = seed_allowed(cfg.search.base_seed) or (
        paspl_full_nas and cfg.graph.dataset in {"flickr", "ogbn-arxiv"}
        and cfg.search.base_seed in (43, 44))
    if not seed_ok or cfg.search.num_layers != 2:
        raise ValueError("P0-B requires seed42 (X opt-in permits 43/44) and two layers")
    if mode == "full_nas" and (cfg.search.m != 16 or cfg.search.generator != "rl"
                                or cfg.search.pool_mode != "independent"):
        raise ValueError("D10 requires the independent canonical RL controller, M=16")
    if (cfg.train.lr, cfg.train.weight_decay, cfg.train.dropout) != (0.01, 5e-4, 0.0):
        raise ValueError("D4 requires Adam lr .01 / weight decay 5e-4 / dropout 0")
    cfg.validate()
    flags = resolve_executor(cfg)
    if any(value != "gpu" for value in flags.values()):
        raise ValueError("shared retrain requires G3 gpu/gpu/gpu")
    return flags


def run_port_search(cfg, method, out, *, mode, pool=None, stage="complete", shortlist_size=None,
                    pool_limit=None, capacity_probe=False, from_search_record=None, search_limit=None,
                    max_search_candidates=None, max_search_seconds=None, collect_search_garbage=False,
                    paspl_full_nas=False, paspl_collect_search_garbage=False, x16_paspl_fanout5=False,
                    cached_sampler=False):
    from .search import ranked_candidates

    job = out / "port_job.json"
    job.write_text(json.dumps(dict(config=cfg.to_dict(), method=method,
                                   mode=mode, pool=str(pool.resolve()) if pool else None,
                                   stage=stage, shortlist_size=shortlist_size, pool_limit=pool_limit,
                                   capacity_probe=capacity_probe, paspl_full_nas=paspl_full_nas,
                                   paspl_collect_search_garbage=paspl_collect_search_garbage,
                                   x16_paspl_fanout5=x16_paspl_fanout5,
                                   cached_sampler=cached_sampler,
                                   from_search_record=str(from_search_record.resolve()) if from_search_record else None,
                                   search_limit=search_limit, max_search_candidates=max_search_candidates,
                                   max_search_seconds=max_search_seconds, collect_search_garbage=collect_search_garbage,
                                   out=str(out)), indent=2))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, cfg.runtime.baseline_gpus)))
    env.pop("RAY_ADDRESS", None)
    command = [sys.executable, "-m", "sregnas.baselines.graphnap.runtime", "--job", str(job)]
    (out / "port_launch.json").write_text(json.dumps(dict(command=command), indent=2))
    with (out / "port_search.log").open("w") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    rows = json.loads((out / "port_candidates.json").read_text())
    normalized = [dict(row, architecture_json=json.dumps(row["architecture"])) for row in rows]
    ranked = ranked_candidates(normalized, method)
    graph = json.loads((out / "retrain_graph.json").read_text())["graph_path"]
    return ranked, graph
