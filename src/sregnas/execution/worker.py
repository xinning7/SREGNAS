


from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from sregnas.paths import PROJECT_ROOT as ROOT
from sregnas.icde27.ledger import call as recorded_call

from sregnas.models.canonical import provenance
from sregnas.selection.scoring import ALLOCATION_SCORE_MODE, run_allocation_score_cell
from sregnas.core.contracts import CellResult, Status
from sregnas.training.fullbatch import (
    CELL_TIMING_FIELDS,
    PersistentCellRuntime,
    run_cell,
)
from sregnas.training.minibatch import ProductsRuntimeCache, run_products_cell


RESTART_STATUSES = {Status.OOM, Status.PROBE_OOM, Status.CUDA_FAULT}


def _fallback(spec: dict, prov: dict, exc: Exception) -> CellResult:
    traceback.print_exc(file=sys.stderr)
    timings = {field: 0.0 for field in CELL_TIMING_FIELDS}
    timings["process_spawn_s"] = max(
        0.0,
        time.perf_counter()
        - float(spec.get("parent_launch_perf_s", time.perf_counter())),
    )
    return CellResult(
        status=Status.CUDA_FAULT if "cuda" in str(exc).lower() else Status.ERROR,
        fingerprint=spec["fingerprint"],
        phase=spec["phase"],
        gpu=int(spec["gpu"]),
        round=int(spec["round"]),
        shard=int(spec["shard"]),
        architecture=spec["architecture"],
        wall_s=sum(timings.values()),
        error=f"{type(exc).__name__}: {exc}",
        metrics={"provenance": prov, **timings},
    )


def main() -> int:
    from sregnas.paths import require_gpu_permission

    require_gpu_permission()
    import argparse
    from sregnas.training.executor import (
        add_executor_arguments,
        apply_executor_arguments,
    )

    parser = argparse.ArgumentParser()
    add_executor_arguments(parser)
    executor_args = parser.parse_args()
    prov = provenance(
        __file__,
        {
            "physical_gpu": int(os.environ["SREGNAS_PHYSICAL_GPU"]),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "execution_mode": "persistent_worker",
        },
    )
    try:
        runtime = PersistentCellRuntime()
        from sregnas.icde27.s0 import observe_cache_upload
        observe_cache_upload(runtime)
        if runtime.device.type != "cuda":
            raise RuntimeError("approved GPU worker cannot find a usable CUDA device")
        products_runtime = ProductsRuntimeCache()
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        print(
            "WORKER_FATAL "
            + json.dumps(
                {"type": type(exc).__name__, "error": str(exc)},
                sort_keys=True,
            ),
            flush=True,
        )
        return 2
    if os.environ.get('SREGNAS_X22_INCREMENTAL_AUDIT') == '1':
        from sregnas.icde27.x22.origins import audit
        audit(full=True)
    print(
        "WORKER_READY "
        + json.dumps(
            {
                "pid": os.getpid(),
                "physical_gpu": int(os.environ["SREGNAS_PHYSICAL_GPU"]),
                "cuda_init_s": runtime.startup_cuda_init_s,
                "canonical_model_path": prov["canonical_model_path"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for line in sys.stdin:
        if not line.strip():
            continue
        command = json.loads(line)
        if command.get("command") == "STOP":
            if os.environ.get('SREGNAS_X22_INCREMENTAL_AUDIT') == '1':
                from sregnas.icde27.x22.origins import finalize
                finalize()
                from sregnas.icde27.x22.receipt_cache import finalize as finalize_receipts
                finalize_receipts()
            if os.environ.get('SREGNAS_X22_DEFER_RECORDS') == '1':
                from sregnas.icde27.x22.deferred import flush
                flush()
            print("WORKER_STOPPED", flush=True)
            return 0
        spec = apply_executor_arguments(executor_args, command["spec"])
        out_path = Path(command["out_path"])
        try:
            from .startup import assert_cell_start

            startup = assert_cell_start(spec)
            if spec.get("mode") == "retrain_s1_stage":
                from sregnas.training.halving import run_s1_stage

                result = run_s1_stage(spec, runtime=runtime)
            elif spec.get("mode") == ALLOCATION_SCORE_MODE:
                result = run_allocation_score_cell(
                    spec,
                    device=runtime.device,
                    runtime=runtime,
                    products_runtime=products_runtime,
                )
            elif "products_graph_manifest_path" in spec and spec.get("mode") != "proxy":
                result = recorded_call(
                    spec, run_products_cell,
                    device=runtime.device,
                    runtime=products_runtime,
                )
            else:
                result = recorded_call(spec, run_cell, runtime=runtime)
            result.metrics["provenance"] = prov
            result.metrics["cell_start"] = startup
            from .startup import assert_project_module_origins

            origins_after = assert_project_module_origins()
            result.metrics["project_module_origins_after"] = origins_after
            if os.environ.get('SREGNAS_X22_DEFER_RECORDS') == '1':
                from sregnas.icde27.x22.deferred import audit_log
                audit_log('CELL_MODULES_AFTER ', origins_after)
            else:
                print(
                    "CELL_MODULES_AFTER " + json.dumps(origins_after, sort_keys=True),
                    flush=True,
                )
        except Exception as exc:
            result = _fallback(spec, prov, exc)
        if result.metrics.get("timing_schema") != "products_minibatch_v1":
            for field in CELL_TIMING_FIELDS:
                result.metrics.setdefault(field, 0.0)
        result.metrics.update(
            {
                "execution_mode": "persistent_worker",
                "worker_pid": os.getpid(),
            }
        )
        if os.environ.get('SREGNAS_X22_DEFER_RECORDS') == '1':
            from sregnas.icde27.x22.deferred import worker_snapshot
            wire_result = worker_snapshot(result)
        else:
            result.dump(out_path)
        print(
            "WORKER_RESULT "
            + json.dumps(
                {
                    "out_path": str(out_path),
                    "status": str(result.status),
                    "fingerprint": result.fingerprint,
                    **({'result': wire_result} if os.environ.get('SREGNAS_X22_DEFER_RECORDS') == '1' else {}),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if result.status in RESTART_STATUSES:
            return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
