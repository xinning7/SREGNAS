


def estimate(cfg, hardware):
    n = len(cfg.runtime.system_gpus)
    smoke = cfg.search.q < 400
    if hardware == "p100":
        minutes = [20, 60] if smoke else [180, 480]
        basis = "Uncalibrated planning allowance only: P100 has no local timing/memory measurements. Re-estimate after an approved pilot."
    elif smoke:
        minutes = [10, 25]
        basis = "Q20/E25 pilot plus unchanged 200-epoch finalist retraining; baseline and system run sequentially."
    else:
        minutes = {2: [100, 140], 4: [60, 90], 8: [45, 75]}.get(n, [60, 140])
        basis = "Historical A6000 system+baseline: N2 about 90 min; N4 about 53 min; 8 shards/4 GPUs about 35-42 min. Native 8-GPU range is an extrapolation. Includes planning margin."
    if cfg.runtime.execution_scope == "system":
        if hardware == "p100":
            minutes = None
            basis = "System-only P100 runtime is uncalibrated; verify memory capacity and an approved pilot first."
        else:
            minutes = (
                [1, 5]
                if smoke
                else {2: [20, 30], 4: [10, 20], 8: [5, 15]}.get(n, [10, 30])
            )
            basis = "System-only estimate: measured A6000 N2 Q20/E25 search 17.06 s and K2 retraining 35.41 s; Q400/E100/K8 extrapolation is about 25 min. Other GPU counts are unmeasured planning ranges. Excludes baseline."
    if cfg.graph.dataset != "ogbn-arxiv" or cfg.graph.partition_strategy in {
        "fi",
        "random",
    }:
        minutes = None
        basis = "This dataset/partition/executor combination has no SREGNAS GPU calibration. CPU migration checks do not establish GPU time or peak memory; estimate after an approved pilot."
    return dict(
        measured=False,
        hardware=hardware,
        execution_scope=cfg.runtime.execution_scope,
        physical_gpus=n,
        logical_shards=cfg.graph.num_gpus,
        q_per_side=cfg.search.q,
        proxy_epochs=cfg.train.proxy_epochs,
        retrain_epochs=cfg.train.retrain_epochs,
        wall_minutes_range=minutes,
        reserved_gpu_hours_range=(
            [round(len(cfg.allocated_gpus) * x / 60, 2) for x in minutes]
            if minutes is not None
            else None
        ),
        basis=basis,
        approval_required=True,
    )
