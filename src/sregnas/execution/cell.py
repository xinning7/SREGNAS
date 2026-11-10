


from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

from sregnas.paths import PROJECT_ROOT as ROOT
from sregnas.icde27.ledger import call as recorded_call

from sregnas.models.canonical import provenance
from sregnas.selection.scoring import ALLOCATION_SCORE_MODE, run_allocation_score_cell
from sregnas.core.contracts import CellResult, Status
from sregnas.training.fullbatch import run_cell
from sregnas.training.minibatch import run_products_cell


TIMING_FIELDS = (
    "process_spawn_s",
    "cuda_init_s",
    "data_load_h2d_s",
    "train_compute_s",
    "teardown_s",
)


def main() -> int:
    from sregnas.paths import require_gpu_permission

    require_gpu_permission()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("approved GPU cell cannot find a usable CUDA device")
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    from sregnas.training.executor import (
        add_executor_arguments,
        apply_executor_arguments,
    )

    add_executor_arguments(parser)
    args = parser.parse_args()
    spec = apply_executor_arguments(args, json.loads(Path(args.spec).read_text()))
    prov = provenance(
        __file__,
        {
            "phase": spec["phase"],
            "physical_gpu": spec["gpu"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )
    try:
        from .startup import assert_cell_start

        startup = assert_cell_start(spec)
        if spec.get("mode") == "retrain_s1_stage":
            from sregnas.training.halving import run_s1_stage

            result = run_s1_stage(spec)
        elif spec.get("mode") == ALLOCATION_SCORE_MODE:
            result = run_allocation_score_cell(spec)
        elif "products_graph_manifest_path" in spec and spec.get("mode") != "proxy":
            result = recorded_call(spec, run_products_cell)
        else:
            result = recorded_call(spec, run_cell)
        result.metrics["provenance"] = prov
        result.metrics["cell_start"] = startup
    except Exception as exc:
        traceback.print_exc()
        fallback_timing = {field: 0.0 for field in TIMING_FIELDS}
        fallback_timing["process_spawn_s"] = max(
            0.0,
            time.perf_counter()
            - float(spec.get("parent_launch_perf_s", time.perf_counter())),
        )
        result = CellResult(
            status=Status.CUDA_FAULT if "cuda" in str(exc).lower() else Status.ERROR,
            fingerprint=spec["fingerprint"],
            phase=spec["phase"],
            gpu=int(spec["gpu"]),
            round=int(spec["round"]),
            shard=int(spec["shard"]),
            architecture=spec["architecture"],
            error=f"{type(exc).__name__}: {exc}",
            wall_s=sum(fallback_timing.values()),
            metrics={"provenance": prov, **fallback_timing},
        )
    if result.metrics.get("timing_schema") != "products_minibatch_v1":
        for field in TIMING_FIELDS:
            result.metrics.setdefault(field, 0.0)
    result.dump(Path(args.out))
    print(
        f"CELL_RESULT phase={result.phase} fp={result.fingerprint} "
        f"status={result.status} value={result.value} wall_s={result.wall_s:.3f}",
        flush=True,
    )
    return 0 if result.status not in {Status.ERROR, Status.CUDA_FAULT} else 2


if __name__ == "__main__":
    raise SystemExit(main())
