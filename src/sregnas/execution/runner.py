

import hashlib
import json
import os
import subprocess
import time

import yaml

from sregnas.models.canonical import provenance
from sregnas.reporting.report import generate_report
from .queue import TaskQueue
from .telemetry import gpu_idle_preflight, start_monitors, stop_monitors, write_snapshot


def run(cfg, run_dir, hardware_path, *, resume=False, allow_quality_sharing=False):
    import fcntl
    from sregnas.comparison.products_approval import assert_products_method_approval

    assert_products_method_approval(cfg, hardware_path=hardware_path)
    from .provenance import freeze_run_commit

    freeze_run_commit()
    os.environ["SREGNAS_STOP_ON_INVALID"] = "1"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "another process is already using this run directory"
            ) from None
        return _run_locked(cfg, run_dir, hardware_path, resume=resume,
                           allow_quality_sharing=allow_quality_sharing)


def _assert_idle_or_shared(preflight, *, allow_quality_sharing=False):
    if not preflight['idle'] and not allow_quality_sharing:
        raise RuntimeError('requested GPUs are occupied; select an idle allocation before running')


def _run_locked(cfg, run_dir, hardware_path, *, resume=False, allow_quality_sharing=False):
    from .options import resolve_executor

    resolve_executor(cfg)
    if any(p.name != ".run.lock" for p in run_dir.iterdir()) and not resume:
        raise FileExistsError(
            "nonempty output directory; use a fresh directory or explicit --resume"
        )
    config = cfg.to_dict()
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    request = run_dir / "request.json"
    if resume:
        if (
            not request.exists()
            or json.loads(request.read_text())["config_sha256"] != signature
        ):
            raise ValueError(
                "resume requires the same resolved experiment configuration"
            )
    machine = yaml.safe_load(hardware_path.read_text())
    from sregnas.paths import DATA_DIR
    from sregnas.search.memory_gate import STATIC_GATE_PREDICATE

    from .provenance import commit_hash, implementation_sha256

    identity = dict(
        commit_hash=commit_hash(),
        implementation_sha256=implementation_sha256(),
        data_dir=str(DATA_DIR),
        static_gate_predicate=STATIC_GATE_PREDICATE,
        hardware=machine,
    )
    if resume and json.loads(request.read_text()).get("identity") != identity:
        raise ValueError(
            "resume requires identical source, data directory, gate and hardware profile"
        )
    gpus = cfg.allocated_gpus
    if len(gpus) > machine["available_gpus"]:
        raise ValueError("configuration exceeds this server's GPU allocation")

    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        raise ValueError(
            "unset CUDA_VISIBLE_DEVICES for GPU runs; choose physical IDs in runtime.*_gpus"
        )
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    devices = {}
    for row in rows:
        index, name, memory, uuid = [value.strip() for value in row.split(",")]
        devices[int(index)] = dict(name=name, memory_mib=int(memory), uuid=uuid)
    if any(
        gpu not in devices or machine["name_contains"] not in devices[gpu]["name"]
        for gpu in gpus
    ):
        raise ValueError("GPU type/IDs differ from the selected hardware profile")
    run_dir.mkdir(parents=True, exist_ok=True)
    request.write_text(
        json.dumps(
            dict(
                config_sha256=signature,
                config=config,
                hardware=machine,
                identity=identity,
                physical_devices={gpu: devices[gpu] for gpu in gpus},
            ),
            indent=2,
        )
        + "\n"
    )
    monitor_dir = run_dir / "monitor"
    monitor_dir.mkdir(exist_ok=True)
    preflight = gpu_idle_preflight(gpus)
    preflight['allow_quality_sharing'] = allow_quality_sharing
    (monitor_dir / "preflight.json").write_text(json.dumps(preflight, indent=2))
    _assert_idle_or_shared(preflight, allow_quality_sharing=allow_quality_sharing)
    write_snapshot(monitor_dir / "launch_snapshot.txt", gpus)
    db_path = run_dir / "queue.sqlite"
    queue = TaskQueue(db_path)
    queue.reset_running()
    queue.reconcile_prelaunch_events()
    queue.reconcile_fingerprint_pretrain_errors()
    queue.set_state("status", "RUNNING")
    queue.set_state("current_phase", cfg.name)
    queue.set_state("pid", os.getpid())
    if cfg.runtime.side_timer == "continuous":
        queue.set_state("continuous_side_campaign", {"source": "runtime.side_timer"})
    queue.close()
    record = provenance(
        __file__, dict(physical_devices=devices, **identity, config_sha256=signature)
    )
    (run_dir / "provenance.json").write_text(json.dumps(record, indent=2))
    os.environ["SREGNAS_ALLOW_GPU"] = "1"
    monitors = start_monitors(monitor_dir, gpus, cfg.runtime.monitor_interval_s)
    started = time.perf_counter()
    try:
        from .orchestrator import run_profile

        run_profile(cfg, run_dir, db_path)
    except BaseException:
        queue = TaskQueue(db_path)
        queue.set_state("status", "FAILED")
        queue.close()
        raise
    else:
        queue = TaskQueue(db_path)
        queue.set_state("status", "COMPLETE")
        queue.set_state("current_phase", "COMPLETE")
        queue.close()
    finally:
        stop_monitors(monitors)
        elapsed = time.perf_counter() - started
        (run_dir / "reservation.json").write_text(
            json.dumps(
                dict(
                    elapsed_wall_s=elapsed,
                    reserved_gpus=len(gpus),
                    reserved_gpu_hours=elapsed * len(gpus) / 3600,
                    includes_preparation=True,
                    resumed=resume,
                ),
                indent=2,
            )
        )
        os.environ.pop("SREGNAS_ALLOW_GPU", None)
        generate_report(run_dir)
