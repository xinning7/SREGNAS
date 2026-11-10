

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from sregnas.core.contracts import CellResult, Status


from sregnas.paths import PROJECT_ROOT as ROOT

WORKER_SCRIPT = Path(__file__).with_name("worker.py")
CELL_TIMING_FIELDS = (
    "process_spawn_s",
    "cuda_init_s",
    "data_load_h2d_s",
    "train_compute_s",
    "teardown_s",
)
RESTART_STATUSES = {Status.OOM, Status.PROBE_OOM, Status.CUDA_FAULT}


class WorkerProtocolError(RuntimeError):
    pass


class PersistentGpuWorker:
    pass

    def __init__(self, run_dir: Path, physical_gpu: int, label: str):
        self.run_dir = run_dir
        self.physical_gpu = int(physical_gpu)
        self.label = label
        self.generation = 0
        self.restart_count = 0
        self.startup_events: list[dict[str, Any]] = []
        self.shutdown_s = 0.0
        self._proc: subprocess.Popen[str] | None = None
        self._stdout_buffer = bytearray()
        self._log = None
        self._lock = threading.Lock()

    def _record_line(self, line: str) -> None:
        if self._log is not None:
            self._log.write(line + "\n")
            if os.environ.get('SREGNAS_X22_DEFER_RECORDS') != '1':
                self._log.flush()

    def _read_until(
        self, prefixes: tuple[str, ...], timeout_s: float
    ) -> tuple[str, str]:
        if self._proc is None or self._proc.stdout is None:
            raise WorkerProtocolError("worker stdout is unavailable")
        deadline = time.monotonic() + timeout_s
        while True:




            if b"\n" in self._stdout_buffer:
                raw, _, rest = self._stdout_buffer.partition(b"\n")
                self._stdout_buffer = rest
                line = raw.decode(
                    self._proc.stdout.encoding or "utf-8", errors="replace"
                )
                self._record_line(line)
                for prefix in prefixes:
                    if line.startswith(prefix):
                        return prefix, line[len(prefix) :].strip()
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerProtocolError(
                    f"GPU {self.physical_gpu} worker protocol timeout"
                )
            ready, _, _ = select.select(
                [self._proc.stdout.fileno()], [], [], min(remaining, 5.0)
            )
            if not ready:
                if self._proc.poll() is not None:
                    raise WorkerProtocolError(
                        f"GPU {self.physical_gpu} worker exited "
                        f"{self._proc.returncode}"
                    )
                continue
            chunk = os.read(self._proc.stdout.fileno(), 65536)
            if not chunk:
                raise WorkerProtocolError(
                    f"GPU {self.physical_gpu} worker closed stdout"
                )
            self._stdout_buffer.extend(chunk)

    def _start(self) -> float:
        from sregnas.paths import require_gpu_permission

        require_gpu_permission()
        if self._proc is not None:
            return 0.0
        self.generation += 1
        worker_dir = self.run_dir / "workers"
        worker_dir.mkdir(parents=True, exist_ok=True)
        log_path = (
            worker_dir / f"{self.label}_gpu{self.physical_gpu}_gen{self.generation}.log"
        )
        self._log = log_path.open("a", buffering=1024*1024 if os.environ.get('SREGNAS_X22_DEFER_RECORDS') == '1' else 1)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(self.physical_gpu)
        env["SREGNAS_PHYSICAL_GPU"] = str(self.physical_gpu)
        started = time.perf_counter()
        if os.environ.get('SREGNAS_X18_MEMORY_PREP')=='1':
            from sregnas.icde27.ledger import event as preparation_event
            preparation_event('prep_step_start',step='worker_startup',physical_gpu=self.physical_gpu)
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "sregnas.execution.worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
            env=env,
            text=True,
            bufsize=1,
        )
        prefix, payload = self._read_until(
            ("WORKER_READY ", "WORKER_FATAL "), timeout_s=180.0
        )
        startup_s = time.perf_counter() - started
        if os.environ.get('SREGNAS_X18_MEMORY_PREP')=='1':
            preparation_event('prep_step_end',step='worker_startup',physical_gpu=self.physical_gpu,
                start_perf_s=started,wall_s=startup_s)
        if prefix == "WORKER_FATAL ":
            self._dispose()
            raise WorkerProtocolError(f"worker startup failed: {payload}")
        ready = json.loads(payload)
        event = {
            "generation": self.generation,
            "physical_gpu": self.physical_gpu,
            "startup_wall_s": startup_s,
            **ready,
        }
        self.startup_events.append(event)
        return startup_s

    def _dispose(self) -> None:
        proc, self._proc = self._proc, None
        self._stdout_buffer.clear()
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        if proc is not None:
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except OSError:
                    pass
            if proc.stdout is not None:
                try:
                    proc.stdout.close()
                except OSError:
                    pass
        if self._log is not None:
            try:
                self._log.close()
            except OSError:
                pass
            self._log = None

    @staticmethod
    def _failure_result(spec: dict[str, Any], wall_s: float, error: str) -> CellResult:
        return CellResult(
            status=Status.CUDA_FAULT,
            fingerprint=spec["fingerprint"],
            phase=spec["phase"],
            gpu=int(spec["gpu"]),
            round=int(spec["round"]),
            shard=int(spec["shard"]),
            architecture=spec["architecture"],
            wall_s=wall_s,
            error=error,
            metrics={
                "process_spawn_s": wall_s,
                "cuda_init_s": 0.0,
                "data_load_h2d_s": 0.0,
                "train_compute_s": 0.0,
                "teardown_s": 0.0,
                "timing_sum_s": wall_s,
                "timing_schema": "persistent_worker_v1_failure",
                "execution_mode": "persistent_worker",
            },
        )

    def execute(self, spec: dict[str, Any], out_path: Path) -> CellResult:
        with self._lock:
            parent_started = time.perf_counter()
            try:
                startup_s = self._start()
                if self._proc is None or self._proc.stdin is None:
                    raise WorkerProtocolError("worker stdin is unavailable")
                command_spec = dict(spec)
                command_spec["parent_launch_perf_s"] = parent_started
                self._proc.stdin.write(
                    json.dumps(
                        {
                            "command": "RUN",
                            "spec": command_spec,
                            "out_path": str(out_path),
                        }
                    )
                    + "\n"
                )
                self._proc.stdin.flush()
                _prefix, payload = self._read_until(
                    ("WORKER_RESULT ",), timeout_s=7200.0
                )
                response = json.loads(payload)
                if response["out_path"] != str(out_path):
                    raise WorkerProtocolError(
                        "worker returned an unexpected result path"
                    )
                result = (CellResult(**response['result']) if os.environ.get('SREGNAS_X22_DEFER_RECORDS') == '1'
                          else CellResult.load(out_path))
                restart = result.status in RESTART_STATUSES
                if restart:
                    self.restart_count += 1
                    if self._proc is not None:
                        self._proc.wait(timeout=30)
                    self._dispose()
                parent_wall_s = time.perf_counter() - parent_started
                if result.metrics.get("timing_schema") == "products_minibatch_v1":
                    result.metrics.update(
                        {
                            "parent_dispatch_wall_s": parent_wall_s,
                            "worker_generation": self.generation,
                            "worker_startup_wall_s": startup_s,
                            "worker_restart_after_cell": restart,
                        }
                    )
                else:
                    for field in CELL_TIMING_FIELDS:
                        result.metrics.setdefault(field, 0.0)
                    measured = sum(
                        float(result.metrics[field]) for field in CELL_TIMING_FIELDS
                    )
                    result.metrics["teardown_s"] = float(
                        result.metrics["teardown_s"]
                    ) + max(0.0, parent_wall_s - measured)
                    result.metrics.update(
                        {
                            "timing_sum_s": sum(
                                float(result.metrics[field])
                                for field in CELL_TIMING_FIELDS
                            ),
                            "timing_schema": "persistent_worker_v1",
                            "worker_generation": self.generation,
                            "worker_startup_wall_s": startup_s,
                            "worker_restart_after_cell": restart,
                        }
                    )
                result.wall_s = parent_wall_s
                result.dump(out_path)
                return result
            except Exception as exc:
                wall_s = time.perf_counter() - parent_started
                self.restart_count += 1
                self._dispose()
                result = self._failure_result(
                    spec, wall_s, f"{type(exc).__name__}: {exc}"
                )
                result.metrics["worker_generation"] = self.generation
                result.metrics["worker_restart_after_cell"] = True
                result.dump(out_path)
                return result

    def close(self) -> None:
        with self._lock:
            if self._proc is None:
                return
            started = time.perf_counter()
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.write(json.dumps({"command": "STOP"}) + "\n")
                    self._proc.stdin.flush()
                self._read_until(("WORKER_STOPPED",), timeout_s=30.0)
                self._proc.wait(timeout=30)
            except Exception:
                pass
            finally:
                self.shutdown_s += time.perf_counter() - started
                self._dispose()


class PersistentWorkerPool:
    pass

    def __init__(self, run_dir: Path, gpus: list[int], label: str):
        self.workers = {
            int(gpu): PersistentGpuWorker(run_dir, int(gpu), label) for gpu in gpus
        }

    def execute(
        self, physical_gpu: int, spec: dict[str, Any], out_path: Path
    ) -> CellResult:
        return self.workers[int(physical_gpu)].execute(spec, out_path)

    def close(self) -> None:
        for worker in self.workers.values():
            worker.close()

    def audit(self) -> dict[str, Any]:
        return {
            "workers": [
                {
                    "physical_gpu": worker.physical_gpu,
                    "generations": worker.generation,
                    "restart_count": worker.restart_count,
                    "startup_events": worker.startup_events,
                    "shutdown_s": worker.shutdown_s,
                }
                for worker in self.workers.values()
            ]
        }


class ShardPersistentWorkerPool:
    pass

    def __init__(
        self,
        run_dir: Path,
        gpus: list[int],
        label: str,
        num_shards: int,
        retain_proxy_top1: bool = False,
    ):
        if not gpus or len(set(gpus)) != len(gpus):
            raise ValueError("Physical GPUs must be nonempty and unique")
        if num_shards <= 0 or num_shards % len(gpus):
            raise ValueError("Shards must map evenly onto the physical GPUs")
        self.retain_proxy_top1 = retain_proxy_top1
        self.shard_workers = {
            shard: PersistentGpuWorker(
                run_dir, int(gpus[shard % len(gpus)]), f"{label}_shard{shard}"
            )
            for shard in range(num_shards)
        }

    def execute(
        self, physical_gpu: int, spec: dict[str, Any], out_path: Path
    ) -> CellResult:
        shard = int(spec["shard"])
        worker = self.shard_workers[shard]
        if worker.physical_gpu != int(physical_gpu) or int(spec["gpu"]) != int(
            physical_gpu
        ):
            raise ValueError(f"Shard {shard} was dispatched to the wrong physical GPU")
        if self.retain_proxy_top1:
            spec = dict(spec, retain_proxy_top1=True)
        return worker.execute(spec, out_path)

    def close(self) -> None:
        for worker in self.shard_workers.values():
            worker.close()

    def audit(self) -> dict[str, Any]:
        return {
            "worker_binding": "one_process_per_shard",
            "workers": [
                {
                    "worker_id": shard,
                    "shard": shard,
                    "physical_gpu": worker.physical_gpu,
                    "generations": worker.generation,
                    "restart_count": worker.restart_count,
                    "startup_events": worker.startup_events,
                    "shutdown_s": worker.shutdown_s,
                }
                for shard, worker in self.shard_workers.items()
            ],
        }
