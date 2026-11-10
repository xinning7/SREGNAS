

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .ports import METHODS as PORT_METHODS

METHODS = ("SREGNAS", "BaselineFair", "BaselineFull", "FastGNAS") + PORT_METHODS


def wait_gpu_idle_preflight(gpus, *, timeout_s=300.0, poll_s=5.0):
    pass
    from sregnas.execution.telemetry import gpu_idle_preflight

    started = time.monotonic()
    samples = []
    while True:
        snapshot = gpu_idle_preflight(gpus)
        samples.append(snapshot)
        elapsed = time.monotonic() - started
        result = dict(snapshot, wait_seconds=elapsed, timeout_seconds=timeout_s,
                      poll_seconds=poll_s, attempts=len(samples), samples=samples)
        record_dir = os.environ.get("SREGNAS_ICDE27_RECORD_DIR")
        if record_dir:
            path = Path(record_dir) / "gpu_idle_preflight.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            temporary.replace(path)
        if snapshot["idle"] or elapsed >= timeout_s:
            return result
        time.sleep(min(poll_s, timeout_s - elapsed))


def common_retrain(cfg, method, ranked, graph, out, *, shortlist_size=None):
    from sregnas.core.contracts import TaskKey
    from sregnas.execution.tasks import _cell_spec

    cfg = deepcopy(cfg)
    cfg.runtime.require_fi_gpu = False
    cfg.runtime.comparison_method = method
    cfg.train.retrain_seed_policy = "fingerprint_retrain_replica"
    gpus = (
        cfg.runtime.system_gpus if method == "FastGNAS" else cfg.runtime.baseline_gpus
    )
    wanted = shortlist_size if shortlist_size is not None else cfg.fixed_x
    if wanted is None:
        wanted = cfg.graph.num_gpus * cfg.selection.k
    trained, offset, wave = [], 0, 0

    def execute(gpu, specs, name):
        output = out / (name + ".json")
        job = out / (name + ".job.json")
        job.write_text(
            json.dumps(dict(method=method, specs=specs, output=str(output)), indent=2)
        )
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        with (out / (name + ".log")).open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sregnas.comparison.retrain",
                    "--job",
                    str(job),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        return json.loads(output.read_text())

    def make_spec(row, mode, gpu, index):
        key = TaskKey(f"{method}:baseline:{mode}", index, 0, row["id"])
        spec = _cell_spec(
            key=key,
            mode=mode,
            shard=0,
            architecture=row["architecture"],
            graph_path=graph,
            cfg=cfg,
            physical_gpu=gpu,
        )
        spec["checkpoint_path"] = str(out / "checkpoints" / (row["id"] + ".pt"))
        spec["model_family"] = "native" if method == "FastGNAS" else "canonical"
        if method == "FastGNAS" and cfg.runtime.native_retrain_dropout is not None:
            spec["dropout"] = cfg.runtime.native_retrain_dropout
        return spec

    from sregnas.icde27.ledger import event
    event("retrain_started", source="comparison.common_retrain")
    while len([r for r in trained if r["status"] == "OK"]) < wanted:
        needed = wanted - sum(r["status"] == "OK" for r in trained)
        selected = ranked[offset : offset + needed]
        if len(selected) < needed:
            raise RuntimeError(
                "global candidate ranking exhausted before X successful retrains; see retrained.json"
            )
        offset += len(selected)
        groups = {gpu: [] for gpu in gpus}
        for i, row in enumerate(selected):
            gpu = gpus[i % len(gpus)]
            groups[gpu].append(
                make_spec(row, "retrain", gpu, offset - len(selected) + i)
            )
        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            jobs = [
                executor.submit(execute, gpu, specs, f"retrain_w{wave}_gpu{gpu}")
                for gpu, specs in groups.items()
                if specs
            ]
            for job in jobs:
                trained.extend(job.result())
        (out / "retrained.json").write_text(json.dumps(trained, indent=2))
        if any(r["status"] not in {"OK", "OOM", "DIVERGE"} for r in trained):
            raise RuntimeError("unexpected retrain error; inspect worker output")
        wave += 1
    good = [r for r in trained if r["status"] == "OK"]
    winner = sorted(good, key=lambda r: (-r["value"], r["id"]))[0]
    if os.environ.get("SREGNAS_ICDE27_TIE_RULE") == "1":
        from sregnas.icde27.space import canonical_fingerprint
        winner = max(good, key=lambda r: (r["value"], canonical_fingerprint(r["architecture"])))
    event("winner_selected", winner=winner["id"], val_acc=winner["value"],
          tie_occurred=sum(r["value"] == winner["value"] for r in good) > 1,
          checkpoint_path=str(out / "checkpoints" / (winner["id"] + ".pt")))
    from sregnas.execution.campaign_timing import mark
    from sregnas.execution.provenance import commit_hash

    timing = mark(out, "winner_selected")
    (out / "selection.json").write_text(
        json.dumps(
            dict(
                commit_hash=commit_hash(),
                winner=winner,
                selection="validation_only_global_topX",
                nominees=wanted,
            ),
            indent=2,
        )
    )
    spec = make_spec(winner, "post_selection_test", gpus[0], 0)
    tested = execute(gpus[0], [spec], "winner_test")[0]
    from sregnas.icde27.ledger import stop_clock
    if method in PORT_METHODS:


        event("e2e_end", source="P0-B winner test subprocess returned")
    stop_clock()
    if tested["status"] != "OK" or "test_acc" not in tested["metrics"]:
        raise RuntimeError("winner test failed")
    result = dict(
        commit_hash=commit_hash(),
        timing=timing,
        method=method,
        winner=winner["id"],
        val_acc=winner["value"],
        test_acc=tested["metrics"]["test_acc"],
        test_evaluations=1,
        successful_retrains=len(good),
        failed_retrains=len(trained) - len(good),
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))
    return result


def validate_request(cfg, method, *, quality_only=False):
    from sregnas.execution.options import resolve_executor

    flags = resolve_executor(cfg)
    if any(value != "gpu" for value in flags.values()):
        raise ValueError("comparison retraining requires explicit gpu/gpu/gpu options")
    if (cfg.search.generator != "rl" and not (quality_only and method == "SREGNAS")) or cfg.search.pool_mode != "independent":
        raise ValueError("formal comparison uses independent RL search")
    if method == "SREGNAS" and not quality_only:
        if cfg.graph.partition_strategy != "fi" or not cfg.runtime.require_fi_gpu:
            raise ValueError("SREGNAS comparison requires FI and require_fi_gpu")
    if cfg.graph.dataset == "ogbn-products":
        from .products_protocol import confirmed_protocol

        confirmed_protocol(method)
    cfg.validate()
    return flags


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--native-config", type=Path)
    parser.add_argument("--mode", choices=("full_nas", "fixed_pool"), default="full_nas")
    parser.add_argument("--pool", type=Path, help="JSON canonical candidate definitions for fixed_pool")
    parser.add_argument("--pool-limit", type=int)
    parser.add_argument("--shortlist-size", type=int, choices=(2, 8), help="P0-B global shortlist override")
    parser.add_argument("--capacity-probe", action="store_true", help="PAS-PL preload plus one optimizer step before replay")
    parser.add_argument("--paspl-full-nas", action="store_true",
                        help="X15 opt-in: AP full NAS with the Exp7 PAS-PL whole-graph worker")
    parser.add_argument("--paspl-collect-search-garbage", action="store_true",
                        help="X15 opt-in: the same between-candidate collection as Exp1 AP on large graphs")
    parser.add_argument("--x16-paspl-fanout5", action="store_true",
                        help="X16 C2 opt-in: PAS-PL proxy fanout [5,5], preserving all other budgets")
    parser.add_argument("--pi-softmax-max-fix", action="store_true",
                        help="X17 opt-in: correct neighbor importance for sampled final softmax/max layers")
    parser.add_argument("--cached-sampler", action="store_true",
                        help="X17 opt-in: cache the unchanged AP/PAS-PL/GraphNAP worker neighbor sampler")
    parser.add_argument("--proxy-steps", type=int, default=None,
                        help="X18 opt-in: SREGNAS proxy optimizer steps")
    parser.add_argument("--proxy-batch-targets", type=int, default=None,
                        help="X18 opt-in: target batch on the immutable shard layer graphs")
    parser.add_argument("--proxy-ap-targets", action="store_true",
                        help="X18 opt-in: AP's seeded fresh shuffled loader, first target batch per step")
    parser.add_argument("--x18-experiment", action="store_true",
                        help="Explicitly requested X18 seed/partition/physical GPU matrix")
    parser.add_argument('--x18-memory-prep', action='store_true',
                        help='X18 opt-in: cold CSR and per-node permutations, RAM-only shared graph payloads')
    parser.add_argument("--from-search-record", type=Path, help="Nominate/retrain from an existing canonical search record")
    parser.add_argument("--search-limit", type=int)
    parser.add_argument("--max-search-candidates", type=int)
    parser.add_argument("--max-search-seconds", type=float)
    parser.add_argument("--collect-search-garbage", action="store_true", help="AP: collect unreachable loader cycles after each candidate")
    parser.add_argument("--x-queue-amendment", action="store_true", help="Opt in to the user-approved X0-X4 seed/gcfix scope")
    parser.add_argument("--stage", choices=("search", "complete"), default="complete")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--s0-parallel", action="store_true", help="Parallel independent prep and initial probes; preserve ordered refill")
    parser.add_argument("--quality-only", action="store_true")
    parser.add_argument("--allow-quality-sharing", action="store_true",
                        help="Allow foreign GPU processes for an explicitly shared quality-only run")
    parser.add_argument("--gpu-lock", type=Path, help="Override global recorder lock for a disjoint quality GPU group")
    parser.add_argument("--icde27-record", action="store_true", help="Opt-in flock, launch ledger and cold E2E recorder")
    parser.add_argument("--native-reset-architecture-rng", action="store_true")
    parser.add_argument("--canonical-winner-tie", action="store_true")
    parser.add_argument("--record-type", choices=("profile", "full_nas", "fixed_pool"), default="profile")
    parser.add_argument("--experiment-id", default="P0C")
    parser.add_argument("--max-wall-seconds", type=float, default=600)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Bounded N2/Q16 search only; never retrain or test",
    )
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--require-fi-gpu", action="store_true")
    parser.add_argument(
        "--partition-strategy", choices=("fi", "metis", "random", "hierarchical")
    )
    from sregnas.training.executor import add_executor_arguments

    add_executor_arguments(parser)
    args = parser.parse_args(argv)
    if args.x18_experiment:
        os.environ['SREGNAS_X_QUEUE_AMENDMENT']='1'
    if args.x18_memory_prep:
        if args.method!='SREGNAS':parser.error('x18-memory-prep requires SREGNAS')
        os.environ['SREGNAS_X18_MEMORY_PREP']='1'
    if args.proxy_ap_targets:
        if args.method!='SREGNAS':parser.error('proxy-ap-targets requires SREGNAS')
        os.environ['SREGNAS_PROXY_AP_TARGETS']='1'
    for name in ("proxy_steps", "proxy_batch_targets"):
        value = getattr(args, name)
        if value is not None:
            if args.method != "SREGNAS" or value <= 0:
                parser.error(name + " requires SREGNAS and a positive integer")
            os.environ["SREGNAS_" + name.upper()] = str(value)
    if args.pi_softmax_max_fix:
        os.environ['SREGNAS_PI_SOFTMAX_MAX_FIX']='1'
    if args.x16_paspl_fanout5 and (args.method != "PAS-PL" or not args.paspl_full_nas):
        parser.error("x16-paspl-fanout5 requires PAS-PL with paspl-full-nas")
    if args.cached_sampler and args.method not in PORT_METHODS:
        parser.error('cached-sampler requires AP, PAS-PL or GraphNAP')
    if args.paspl_collect_search_garbage and not (args.paspl_full_nas or (
            args.cached_sampler and args.method == 'PAS-PL' and args.mode == 'fixed_pool')):
        parser.error("paspl-collect-search-garbage requires paspl-full-nas")
    if args.paspl_full_nas and (args.method != "PAS-PL" or args.mode != "full_nas"):
        parser.error("paspl-full-nas requires PAS-PL full_nas")
    if args.method not in PORT_METHODS and (args.pool_limit is not None or args.shortlist_size is not None or args.capacity_probe):
        parser.error("pool-limit, shortlist-size and capacity-probe are P0-B options")
    if args.capacity_probe and (args.method != "PAS-PL" or args.mode != "fixed_pool"):
        parser.error("capacity-probe requires PAS-PL fixed_pool")
    if args.pool_limit is not None and (args.mode != "fixed_pool" or args.pool_limit <= 0):
        parser.error("pool-limit requires fixed_pool and a positive count")
    if args.x_queue_amendment:
        os.environ['SREGNAS_X_QUEUE_AMENDMENT']='1'
    limited = args.max_search_candidates is not None or args.max_search_seconds is not None
    if (limited or args.from_search_record or args.search_limit is not None or args.collect_search_garbage):
        if args.method not in ("GraphNAP", "AP") or args.mode != "full_nas" or args.stage != "complete":
            parser.error("bounded/reused search requires a complete GraphNAP/AP pipeline")
        from sregnas.icde27.x_authorization import gc_full_allowed
        if args.record_type != "profile" and not (args.collect_search_garbage and gc_full_allowed(args)):
            parser.error("bounded/reused search must be recorded as profile")
        if args.from_search_record and (limited or not args.search_limit or args.search_limit <= 0):
            parser.error("source reuse requires positive search-limit and no new-search limits")
        if args.search_limit is not None and not args.from_search_record:
            parser.error("search-limit requires from-search-record")
        if any(v is not None and v <= 0 for v in (args.max_search_candidates, args.max_search_seconds)):
            parser.error("search limits must be positive")
        if args.collect_search_garbage and args.method != "AP":
            parser.error("loader-cycle collection is scoped to AP")
    if args.quality_only:
        os.environ["SREGNAS_QUALITY_ONLY"] = "1"
    if args.s0_parallel:
        os.environ["SREGNAS_S0_PARALLEL"] = "1"
    if args.gpu_lock and not args.quality_only:
        parser.error("a group gpu-lock requires quality-only; formal timing retains the global lock")
    if args.allow_quality_sharing and not (args.quality_only and args.icde27_record):
        parser.error("allow-quality-sharing requires quality-only and icde27-record")
    if args.native_reset_architecture_rng:
        os.environ["SREGNAS_ICDE27_NATIVE_RNG"] = "1"
    if args.canonical_winner_tie:
        os.environ["SREGNAS_ICDE27_TIE_RULE"] = "1"
    if args.data_dir:
        os.environ["SREGNAS_DATA_DIR"] = str(args.data_dir.resolve())
    from sregnas.config import load_config
    from sregnas.execution.options import resolve_executor

    cfg = load_config(args.config)
    if args.x18_experiment:
        from sregnas.icde27.x18.authorization import validate as validate_x18
        validate_x18(cfg,args)
    resolve_executor(cfg, args)
    if args.x_queue_amendment:
        from sregnas.icde27.x_authorization import validate as validate_x
        validate_x(cfg,args.method,args.collect_search_garbage)
    if args.partition_strategy:
        cfg.graph.partition_strategy = args.partition_strategy
        if args.quality_only and args.partition_strategy != "fi":
            cfg.runtime.require_fi_gpu = False
    if args.require_fi_gpu:
        cfg.runtime.require_fi_gpu = True
    if args.method in PORT_METHODS:
        from .ports import validate_port_request

        flags = validate_port_request(cfg, args.method, args.mode, args.pool,
                                      paspl_full_nas=args.paspl_full_nas)
    else:
        if args.mode != "full_nas" or args.pool is not None:
            parser.error("--mode fixed_pool / --pool are P0-B port options")
        flags = validate_request(cfg, args.method, quality_only=args.quality_only)
    if args.smoke:
        from .smoke import validate_smoke

        validate_smoke(cfg, args.stage)
    if args.method == "FastGNAS" and not args.native_config:
        parser.error(
            "FastGNAS requires --native-config configs/comparison/native/fastgnas.yaml"
        )
    out = args.out.resolve()
    plan = dict(
        method=args.method,
        mode=args.mode,
        stage=args.stage,
        smoke=args.smoke,
        config=cfg.to_dict(),
        executor=flags,
        search_path=(
            "native"
            if args.method == "FastGNAS"
            else "central" if args.method.startswith("Baseline") or args.method in PORT_METHODS else "sharded"
        ),
        retrain="sregnas shared training",
        products_readiness=(
            "requires verification artifact"
            if cfg.graph.dataset == "ogbn-products"
            else "not_applicable"
        ),
    )
    if args.method in PORT_METHODS:
        plan.update(
            search_path="fixed_pool" if args.mode == "fixed_pool" else "canonical_central_reference_executor",
            proxy_budget="50 steps", proxy_inference="sampled_validation",
            model_backend="canonical", protocol_decisions="P0-B D1/D3/D4/D8/D10",
            search_feature_residency="cpu" if args.method == "AP" else "gpu",
            retrain_feature_residency="gpu",
            products_readiness="P0-B explicit user authorization",
        )
    if args.method == "FastGNAS":
        from .search import native_config

        plan["native_config"], _ = native_config(cfg, args.native_config, out)
    if args.paspl_full_nas:
        plan.update(paspl_full_nas=True, search_feature_residency="whole_graph_preload",
                    collect_search_garbage=args.paspl_collect_search_garbage,
                    search_worker="sregnas.baselines.paspl.adapter.Worker",
                    preload_components=["features", "labels", "adjacency"])
    if args.x16_paspl_fanout5:
        plan.update(x16_paspl_fanout5=True, proxy_fanout=[5, 5])
    if args.cached_sampler:
        plan.update(cached_sampler=True, sampler_cache_scope='one CSC per worker; train and validation share it')
    if args.pi_softmax_max_fix:
        plan.update(pi_softmax_max_fix='pi_softmax_max_v1')
    if limited or args.from_search_record or args.collect_search_garbage:
        plan.update(from_search_record=str(args.from_search_record.resolve()) if args.from_search_record else None,
                    search_limit=args.search_limit, max_search_candidates=args.max_search_candidates,
                    max_search_seconds=args.max_search_seconds, collect_search_garbage=args.collect_search_garbage)
    if args.from_search_record:
        from sregnas.baselines.graphnap.truncation import select_search_record
        select_search_record(args.from_search_record, args.search_limit, cfg, args.method)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.allow_gpu:
        parser.error("execution requires explicit GPU approval and --allow-gpu")
    if args.icde27_record:
        if args.method in PORT_METHODS:
            from sregnas.baselines.graphnap.recording import record_comparison
        else:
            from sregnas.icde27.recorder import record_comparison
        return record_comparison(args, cfg, argv)
    if not args.smoke and args.method not in PORT_METHODS:
        from .products_approval import assert_products_method_approval

        assert_products_method_approval(cfg, args.method, hardware_path=args.hardware)
    from sregnas.execution.provenance import freeze_run_commit, implementation_sha256

    run_commit = freeze_run_commit()
    plan["commit_hash"] = run_commit
    plan["implementation_sha256"] = implementation_sha256()
    os.environ["SREGNAS_STOP_ON_INVALID"] = "1"
    if args.method == "SREGNAS" and not args.smoke:
        if args.stage != "complete":
            parser.error("SREGNAS entry runs the existing complete pipeline")
        from sregnas.execution.runner import run

        run(cfg, out, args.hardware)
        return 0
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("use a fresh comparison output directory")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise ValueError(
            "unset CUDA_VISIBLE_DEVICES; select physical IDs in the config"
        )
    import yaml

    machine = yaml.safe_load(args.hardware.read_text())
    gpus = (
        cfg.runtime.system_gpus
        if args.method in {"SREGNAS", "FastGNAS"}
        else cfg.runtime.baseline_gpus
    )
    if len(gpus) > machine["available_gpus"]:
        raise ValueError("allocation exceeds hardware profile")
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"], text=True
    ).splitlines()
    devices = {
        int(line.split(",", 1)[0]): line.split(",", 1)[1].strip() for line in rows
    }
    if any(
        g not in devices or machine["name_contains"] not in devices[g] for g in gpus
    ):
        raise ValueError("physical GPU IDs/type mismatch")
    from sregnas.execution.telemetry import (
        start_monitors,
        stop_monitors,
    )

    preflight = wait_gpu_idle_preflight(gpus)
    if not preflight["idle"]:
        raise RuntimeError("requested GPUs are occupied")
    out.mkdir(parents=True, exist_ok=True)
    (out / "launch.json").write_text(
        json.dumps(dict(plan=plan, preflight=preflight), indent=2)
    )
    saved = {
        k: os.environ.get(k) for k in ("SREGNAS_ALLOW_GPU", "SREGNAS_APPROVED_GPUS")
    }
    os.environ["SREGNAS_ALLOW_GPU"] = "1"
    os.environ["SREGNAS_APPROVED_GPUS"] = ",".join(map(str, gpus))
    monitor = out / "monitor"
    monitor.mkdir()
    monitors = start_monitors(monitor, gpus, cfg.runtime.monitor_interval_s)
    started = time.perf_counter()
    try:
        from .search import run_central_search, run_native_search

        if args.method in PORT_METHODS:
            from .ports import run_port_search

            ranked, graph = run_port_search(cfg, args.method, out, mode=args.mode, pool=args.pool,
                                           stage=args.stage, shortlist_size=args.shortlist_size,
                                           pool_limit=args.pool_limit, capacity_probe=args.capacity_probe,
                                           from_search_record=args.from_search_record, search_limit=args.search_limit,
                                           max_search_candidates=args.max_search_candidates, max_search_seconds=args.max_search_seconds,
                                           collect_search_garbage=args.collect_search_garbage,
                                           paspl_full_nas=args.paspl_full_nas,
                                           paspl_collect_search_garbage=args.paspl_collect_search_garbage,
                                           x16_paspl_fanout5=args.x16_paspl_fanout5,
                                           cached_sampler=args.cached_sampler)
        elif args.method == "SREGNAS":
            from .smoke import run_sharded_smoke

            ranked, graph = run_sharded_smoke(cfg, out)
        elif args.method == "FastGNAS":
            ranked, graph = run_native_search(cfg, args.native_config, out)
        else:
            ranked, graph = run_central_search(cfg, args.method, out)
        if args.smoke and not ranked:
            account = json.loads((out / "search_accounting.json").read_text())
            if account.get("successful") == 0 and account.get("failed") == cfg.search.q:
                (out / "smoke_oom.json").write_text(
                    json.dumps(
                        dict(
                            commit_hash=run_commit,
                            status="OOM_NO_SUCCESSFUL_CANDIDATE",
                            formal_run_eligible=False,
                            search_accounting=account,
                        ),
                        indent=2,
                    )
                )
                return 0
            raise RuntimeError(
                "smoke produced no successful candidate; inspect failed attempts"
            )
        (out / "ranked_candidates.json").write_text(json.dumps(ranked, indent=2))
        (out / "retrain_graph.json").write_text(json.dumps(dict(graph_path=graph)))
        if args.stage == "complete" and args.mode != "fixed_pool" and args.method not in PORT_METHODS:
            common_retrain(cfg, args.method, ranked, graph, out)
    finally:
        stop_monitors(monitors)
        (out / "reservation.json").write_text(
            json.dumps(
                dict(
                    reserved_gpu_hours=len(gpus)
                    * (time.perf_counter() - started)
                    / 3600
                )
            )
        )
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return 0
