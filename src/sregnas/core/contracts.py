

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class Status(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    OK = "OK"
    OOM = "OOM"
    DIVERGE = "DIVERGE"
    STATIC_REJECT = "STATIC_REJECT"
    DEDUP_REJECT = "DEDUP_REJECT"
    RESAMPLE_EXHAUSTED = "RESAMPLE_EXHAUSTED"
    CUDA_FAULT = "CUDA_FAULT"
    ERROR = "ERROR"
    PROBE_OOM = "PROBE_OOM"
    COST_GATE_STOP = "COST_GATE_STOP"
    COST_ANCHOR_EXCEEDED = "COST_ANCHOR_EXCEEDED"
    PIPELINE_SANITY_WARNING = "PIPELINE_SANITY_WARNING"
    PIPELINE_SANITY_STOP = "PIPELINE_SANITY_STOP"
    PROBE_AUDIT_FAILED = "PROBE_AUDIT_FAILED"
    NOT_IMPLEMENTED_M4 = "NOT_IMPLEMENTED_M4"


class EvidenceKind(StrEnum):
    RAW = "RAW"
    MEASURED = "MEASURED"
    PROJECTED = "PROJECTED"
    ANALYTICAL = "ANALYTICAL"


TERMINAL_TASK_STATUSES = {
    Status.OK,
    Status.OOM,
    Status.DIVERGE,
    Status.CUDA_FAULT,
    Status.ERROR,
    Status.PROBE_OOM,
}


@dataclass(frozen=True)
class TaskKey:
    phase: str
    round: int
    gpu: int
    fingerprint: str

    def as_string(self) -> str:
        return f"{self.phase}:{self.round}:{self.gpu}:{self.fingerprint}"


@dataclass
class CellResult:
    status: str
    fingerprint: str
    phase: str
    gpu: int
    round: int
    shard: int
    architecture: list[dict[str, Any]]
    value: float | None = None
    wall_s: float = 0.0
    peak_allocated_gb: float | None = None
    peak_reserved_gb: float | None = None
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    commit_hash: str | None = None

    def dump(self, path: Path) -> None:
        from sregnas.execution.provenance import commit_hash

        if self.commit_hash is None:
            self.commit_hash = commit_hash()
        if os.environ.get('SREGNAS_X22_DEFER_RECORDS') == '1':
            from sregnas.icde27.x22.deferred import atomic_json
            atomic_json(path, asdict(self))
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "CellResult":
        if os.environ.get('SREGNAS_X22_DEFER_RECORDS') == '1':
            from sregnas.icde27.x22.deferred import flush
            flush()
        return cls(**json.loads(path.read_text()))
