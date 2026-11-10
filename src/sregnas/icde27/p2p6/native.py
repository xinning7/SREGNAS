
import json
import os
import tempfile
import time
from pathlib import Path

from sregnas.paths import PROJECT_ROOT as ROOT
from sregnas.icde27 import ledger
from sregnas.icde27.recorder import dump, read_rows
from .plan import selected_rows, digest, logical_shards
from .resources import mark_stage
from .profiling import worker_tail
from .exp3_hooks import scope


def execute(cell, cfg, out):
    import torch
    import ray
    from sregnas.comparison.search import native_config
    from sregnas.baselines.fastgnas.search import merge_dataset_config, set_random_seed
    from sregnas.baselines.fastgnas.data.partition import PartitionGraph
    from sregnas.baselines.fastgnas.trainer.worker import TrainerWorker
    from sregnas.baselines.fastgnas.core.dispatcher_v2 import Dispatcher
    from sregnas.baselines.fastgnas.core.load_estimator import MloadPredictor
    from sregnas.baselines.fastgnas.timing_hook import set_phase, emit
    cfg.search.q = cell["count"]
    config, name = native_config(cfg, ROOT/"configs/icde27/fastgnas_native.yaml", out)
    config = merge_dataset_config(config, name, mode="fixed_pool", qbench=cell["count"])
    if cfg.graph.dataset == "ogbn-products":
        from sregnas.comparison.products_protocol import DIRECTIVE, assert_native_products_config
        os.environ["SREGNAS_PRODUCTS_PROTOCOL"] = str(DIRECTIVE)
        assert_native_products_config(config)
    training = config["training"]
    n = logical_shards(cell)
    training["logical_partitions"] = n
    dump(out/"native_config.json", config)
    os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["TIMING_RUN_DIR"] = str(out)
    os.environ["TIMING_RAY_TMP"] = tempfile.mkdtemp(prefix="p2p6-native-", dir="/tmp")
    set_random_seed(42)
    defer_progress = os.environ.get('SREGNAS_X23_POOL_PROGRESS') == '1'
    ledger.start_clock("P2P6 before native local graph read")
    p = cell["P"]
    ray.init(num_gpus=p, num_cpus=40, resources={f"pilot_gpu_{g}": n//p for g in range(p)},
             include_dashboard=False, _temp_dir=os.environ["TIMING_RAY_TMP"])
    try:
        mark_stage(out, "prep/partition")
        parts = PartitionGraph.get_partitions(dataset_name=name, num_partitions=n, max_hops=2,
            strategy=config["fastgnas_backends"]["partition_strategy"])
        training["num_classes"] = max(int(part.y.max())+1 for part in parts)
        mark_stage(out, "prep/worker_init")
        workers = [TrainerWorker.options(num_gpus=0, num_cpus=4, resources={f"pilot_gpu_{i%p}": 1},
            runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": str(i%p), "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}}).remote(
                worker_id=i, gpu_id=i%p, data_partition=parts[i], config=training, collector_ref=None) for i in range(n)]
        predictor = MloadPredictor.load_from_file(config["nas"]["balancing_strategy"]["model_path"])
        dispatcher = Dispatcher.remote(worker_refs=workers, train_config=training, scheduler_type="rotation",
            balance_controller=None, load_predictor=predictor, enable_balancing=True)
        set_phase("warmup")
        mark_stage(out, "prep/warmup")
        emit("warmup_start")
        warmup = ray.get(dispatcher.warmup_system.remote(batch_size=4, min_architectures=8))
        emit("warmup_end", result=warmup)
        if not warmup["ok"]: raise RuntimeError(f"native warmup failed: {warmup}")
        selected = selected_rows(cell)
        rows = []
        deferred_waves = []
        set_phase("search")
        mark_stage(out, "pool/native_training_and_evaluation")
        ledger.event("pool_start")
        start = time.perf_counter()
        for offset in range(0, len(selected), n):
            batch = selected[offset:offset+n]
            assert len(batch) == n, "native rotation requires complete N-candidate waves"
            specs = [dict(arch_id=r["fingerprint"], arch_spec=json.loads(r["native_architecture_json"])) for r in batch]
            with scope("native_round", round_index=offset//n+1, fingerprints=[r["fingerprint"] for r in batch],
                       global_validation_enabled=config["nas"]["enable_global_evaluation"]):
                ray.get(dispatcher.submit_architectures.remote(architectures=specs, strategy="rotation"))
                before = time.perf_counter()
                trained = ray.get(dispatcher.run_training.remote(num_rounds=training["num_rounds"],
                    epochs_per_round=training["epochs_per_round"], enable_global_evaluation=config["nas"]["enable_global_evaluation"]))
            if defer_progress: deferred_waves.append((offset//n, trained))
            else: dump(out/f"wave_{offset//n}.json", trained)
            for row in batch:
                fp = row["fingerprint"]
                score = trained.get("final_accuracies", {}).get(fp)
                status = "OOM" if fp in trained.get("oom_architectures", []) else "OK" if score is not None else "incomplete"
                rows.append(dict(fingerprint=fp, status=status, value=score,
                                 wave_wall_s=time.perf_counter()-before, wave=offset//n,
                                 failure_stage="pool/native_training_and_evaluation" if status != "OK" else None))
            if not defer_progress: dump(out/"partial_candidates.json", rows)
        end = time.perf_counter()
        ledger.event("pool_end"); ledger.stop_clock()
        if defer_progress: dump(out/"partial_candidates.json", rows)
        for index, trained in deferred_waves: dump(out/f"wave_{index}.json", trained)
        segments = [r for r in read_rows(out/"native_segments.jsonl") if r["event"]=="segment_end" and r["phase"]=="search"]
        spans = [dict(worker=r["worker"], start_perf_s=r["start_perf_s"], end_perf_s=r["perf_s"]) for r in segments]
        dump(out/"worker_tails.json", worker_tail(spans, start, end))
        dump(out/"communication.json", [dict(category="model_rotation", worker=r["worker"],
            fingerprint=r["fingerprint"], bytes_in=r["migration_bytes_in"], bytes_out=r["migration_bytes_out"],
            definition="logical state tensor bytes; not physical network bytes") for r in segments])
        return dict(rows=rows, controller_updates=0, native_config_hash=digest(config), native_warmup_policy="existing default warmup; no RL controller",
                    worker_placements=[dict(worker=i, physical_gpu=i%p, partition=i) for i in range(n)])
    finally:
        try:
            if os.environ.get('SREGNAS_X22_INCREMENTAL_AUDIT')=='1' or os.environ.get('SREGNAS_X22_DEFER_RECORDS')=='1':
                from sregnas.icde27.x22.ray_finish import close_workers
                close_workers(locals().get('workers',[]))
        finally:
            ray.shutdown()
