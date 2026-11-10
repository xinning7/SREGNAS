

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from sregnas.core.contracts import CellResult, TaskKey


class TaskQueue:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if os.environ.get('SREGNAS_X22_MEMORY_QUEUE') == '1':
            from sregnas.icde27.x22.memory_queue import connect
            self.conn = connect(self.path)
        else:
            self.conn = sqlite3.connect(self.path, timeout=60)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA busy_timeout=60000")
        for attempt in range(100):
            try:
                self.conn.execute("PRAGMA journal_mode=WAL")
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == 99:
                    raise
                time.sleep(0.01)
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_key TEXT PRIMARY KEY,
                phase TEXT NOT NULL,
                round INTEGER NOT NULL,
                gpu INTEGER NOT NULL,
                shard INTEGER NOT NULL,
                fingerprint TEXT NOT NULL,
                architecture_json TEXT NOT NULL,
                status TEXT NOT NULL,
                value REAL,
                wall_s REAL,
                result_json TEXT,
                started_at REAL,
                finished_at REAL,
                task_gpu_s REAL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS tasks_phase_status
                ON tasks(phase, status);
            CREATE TABLE IF NOT EXISTS proposal_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phase TEXT NOT NULL,
                round INTEGER NOT NULL,
                gpu INTEGER NOT NULL,
                shard INTEGER NOT NULL,
                fingerprint TEXT NOT NULL,
                architecture_json TEXT NOT NULL,
                status TEXT NOT NULL,
                reward REAL,
                consumes_q INTEGER NOT NULL,
                retry_index INTEGER NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS phase_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phase TEXT NOT NULL,
                event TEXT NOT NULL,
                evidence_kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS dispatch_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                phase TEXT NOT NULL,
                side TEXT NOT NULL,
                scheduler_mode TEXT NOT NULL,
                worker_id INTEGER NOT NULL,
                task_index INTEGER NOT NULL,
                static_worker_id INTEGER NOT NULL,
                dispatch_timestamp REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS dispatch_events_phase_mode
                ON dispatch_events(phase, scheduler_mode);
            CREATE TABLE IF NOT EXISTS run_state (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            """
        )




        task_columns = {
            str(row["name"]) for row in self.conn.execute("PRAGMA table_info(tasks)")
        }
        for column, declaration in (
            ("started_at", "REAL"),
            ("finished_at", "REAL"),
            ("task_gpu_s", "REAL"),
        ):
            if column not in task_columns:
                self.conn.execute(
                    f"ALTER TABLE tasks ADD COLUMN {column} {declaration}"
                )
        self.conn.commit()

    def record_dispatch(
        self,
        *,
        task_id: str,
        phase: str,
        side: str,
        scheduler_mode: str,
        worker_id: int,
        task_index: int,
        static_worker_id: int,
    ) -> None:
        pass
        self.conn.execute(
            """
            INSERT INTO dispatch_events(
              task_id,phase,side,scheduler_mode,worker_id,task_index,
              static_worker_id,dispatch_timestamp
            ) VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                task_id,
                phase,
                side,
                scheduler_mode,
                worker_id,
                task_index,
                static_worker_id,
                time.time(),
            ),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def set_state(self, key: str, value: Any) -> None:
        self.conn.execute(
            """
            INSERT INTO run_state(key,value_json,updated_at) VALUES(?,?,?)
            ON CONFLICT(key) DO UPDATE SET
              value_json=excluded.value_json, updated_at=excluded.updated_at
            """,
            (key, json.dumps(value, sort_keys=True), time.time()),
        )
        self.conn.commit()

    def get_state(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute(
            "SELECT value_json FROM run_state WHERE key=?", (key,)
        ).fetchone()
        return default if row is None else json.loads(row["value_json"])

    def add_event(
        self, phase: str, event: str, evidence_kind: str, payload: dict[str, Any]
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO phase_events(phase,event,evidence_kind,payload_json,created_at)
            VALUES(?,?,?,?,?)
            """,
            (
                phase,
                event,
                evidence_kind,
                json.dumps(payload, sort_keys=True),
                time.time(),
            ),
        )
        self.conn.commit()

    def add_proposal(
        self,
        *,
        phase: str,
        round_idx: int,
        gpu: int,
        shard: int,
        fingerprint: str,
        architecture: list[dict[str, Any]],
        status: str,
        reward: float | None,
        consumes_q: bool,
        retry_index: int,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO proposal_events(
              phase,round,gpu,shard,fingerprint,architecture_json,status,reward,
              consumes_q,retry_index,created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                phase,
                round_idx,
                gpu,
                shard,
                fingerprint,
                json.dumps(architecture),
                status,
                reward,
                int(consumes_q),
                retry_index,
                time.time(),
            ),
        )
        self.conn.commit()

    def ensure_task(
        self,
        key: TaskKey,
        shard: int,
        architecture: list[dict[str, Any]],
        physical_gpu: int | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO tasks(
              task_key,phase,round,gpu,shard,fingerprint,architecture_json,status,updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?)
            ON CONFLICT(task_key) DO UPDATE SET
              architecture_json=excluded.architecture_json,
              shard=excluded.shard,
              gpu=CASE
                WHEN tasks.result_json IS NULL THEN excluded.gpu
                ELSE tasks.gpu
              END
            """,
            (
                key.as_string(),
                key.phase,
                key.round,
                key.gpu if physical_gpu is None else physical_gpu,
                shard,
                key.fingerprint,
                json.dumps(architecture),
                "PENDING",
                time.time(),
            ),
        )
        self.conn.commit()

    def task_result(self, key: TaskKey) -> CellResult | None:
        row = self.conn.execute(
            "SELECT result_json,status FROM tasks WHERE task_key=?",
            (key.as_string(),),
        ).fetchone()
        if row is None or row["result_json"] is None:
            return None
        return CellResult(**json.loads(row["result_json"]))

    def mark_running(self, key: TaskKey, *, started_at: float | None = None) -> float:
        started = time.time() if started_at is None else float(started_at)
        self.conn.execute(
            "UPDATE tasks SET status='RUNNING',started_at=?,finished_at=NULL,"
            "task_gpu_s=NULL,updated_at=? WHERE task_key=?",
            (started, started, key.as_string()),
        )
        self.conn.commit()
        return started

    def requeue(self, key: TaskKey) -> None:
        pass

        self.conn.execute(
            """
            UPDATE tasks SET status='PENDING',value=NULL,wall_s=NULL,
              result_json=NULL,started_at=NULL,finished_at=NULL,
              task_gpu_s=NULL,updated_at=? WHERE task_key=?
            """,
            (time.time(), key.as_string()),
        )
        self.conn.commit()

    def complete(self, key: TaskKey, result: CellResult) -> None:
        finished = time.time()
        self.conn.execute(
            """
            UPDATE tasks SET status=?,value=?,wall_s=?,result_json=?,
              finished_at=?,task_gpu_s=?,updated_at=?
            WHERE task_key=?
            """,
            (
                result.status,
                result.value,
                result.wall_s,
                json.dumps(result.__dict__, sort_keys=True),
                finished,
                float(result.wall_s),
                finished,
                key.as_string(),
            ),
        )
        self.conn.commit()

    def reset_running(self) -> int:
        cur = self.conn.execute(
            "UPDATE tasks SET status='PENDING',updated_at=? WHERE status='RUNNING'",
            (time.time(),),
        )
        self.conn.commit()
        return cur.rowcount

    def requeue_cuda_faults(self) -> int:
        pass
        cur = self.conn.execute(
            """
            UPDATE tasks SET status='PENDING',value=NULL,wall_s=NULL,
              result_json=NULL,started_at=NULL,finished_at=NULL,
              task_gpu_s=NULL,updated_at=?
            WHERE status='CUDA_FAULT'
            """,
            (time.time(),),
        )
        self.conn.commit()
        return cur.rowcount

    def reconcile_prelaunch_events(self) -> int:
        pass
        cur = self.conn.execute(
            """
            UPDATE proposal_events
            SET status='ABORTED_PRELAUNCH', consumes_q=0
            WHERE status='LAUNCHED'
              AND NOT EXISTS (
                SELECT 1 FROM tasks
                WHERE tasks.phase=proposal_events.phase
                  AND tasks.round=proposal_events.round
                  AND tasks.gpu=proposal_events.gpu
                  AND tasks.fingerprint=proposal_events.fingerprint
              )
            """
        )
        self.conn.commit()
        return cur.rowcount

    def reconcile_fingerprint_pretrain_errors(self) -> int:
        pass
        rows = self.conn.execute(
            """
            SELECT task_key,phase,round,gpu,fingerprint,result_json
            FROM tasks WHERE status='ERROR' AND result_json IS NOT NULL
            """
        ).fetchall()
        repaired = 0
        for row in rows:
            result = json.loads(row["result_json"])
            error = str(result.get("error", ""))
            pretrain_signatures = (
                "architecture/fingerprint mismatch",
                "Invalid device argument",
            )
            if not any(signature in error for signature in pretrain_signatures):
                continue
            self.conn.execute(
                """
                UPDATE tasks SET status='PENDING',value=NULL,wall_s=NULL,
                  result_json=NULL,updated_at=? WHERE task_key=?
                """,
                (time.time(), row["task_key"]),
            )
            self.conn.execute(
                """
                UPDATE proposal_events
                SET status='ABORTED_PRETRAIN',consumes_q=0
                WHERE phase=? AND round=? AND gpu=? AND fingerprint=?
                  AND status='LAUNCHED'
                """,
                (
                    row["phase"],
                    row["round"],
                    row["gpu"],
                    row["fingerprint"],
                ),
            )
            repaired += 1
        self.conn.commit()
        return repaired

    def mark_controller_round_for_replay(
        self, *, phase: str, round_idx: int, shard: int
    ) -> int:
        pass
        cur = self.conn.execute(
            """
            UPDATE proposal_events
            SET status='ABORTED_STREAM_REPLAY', consumes_q=0
            WHERE phase=? AND round=? AND shard=?
              AND status IN ('ADMITTED','STATIC_REJECT','DEDUP_REJECT')
            """,
            (phase, round_idx, shard),
        )
        self.conn.commit()
        return cur.rowcount

    def record_training_launch(
        self,
        *,
        phase: str,
        round_idx: int,
        gpu: int,
        shard: int,
        fingerprint: str,
        architecture: list[dict[str, Any]],
    ) -> None:
        prior = self.conn.execute(
            """
            SELECT COUNT(*) n FROM proposal_events
            WHERE phase=? AND round=? AND gpu=? AND fingerprint=?
              AND status='LAUNCHED'
            """,
            (phase, round_idx, gpu, fingerprint),
        ).fetchone()["n"]
        self.add_proposal(
            phase=phase,
            round_idx=round_idx,
            gpu=gpu,
            shard=shard,
            fingerprint=fingerprint,
            architecture=architecture,
            status="RESUME_RELAUNCH" if prior else "LAUNCHED",
            reward=None,
            consumes_q=not bool(prior),
            retry_index=0,
        )

    def rows(self, table: str) -> list[dict[str, Any]]:
        if table not in {
            "tasks",
            "proposal_events",
            "phase_events",
            "dispatch_events",
            "run_state",
        }:
            raise ValueError(table)
        return [dict(row) for row in self.conn.execute(f"SELECT * FROM {table}")]

    def status_summary(self) -> dict[str, Any]:
        tasks = {
            f"{row['phase']}:{row['status']}": int(row["n"])
            for row in self.conn.execute(
                "SELECT phase,status,COUNT(*) n FROM tasks GROUP BY phase,status"
            )
        }
        proposals = {
            f"{row['phase']}:{row['status']}": int(row["n"])
            for row in self.conn.execute(
                """
                SELECT phase,status,COUNT(*) n
                FROM proposal_events GROUP BY phase,status
                """
            )
        }
        return {
            "run_status": self.get_state("status", "UNKNOWN"),
            "current_phase": self.get_state("current_phase", "UNKNOWN"),
            "tasks": tasks,
            "proposals": proposals,
            "cost_gate": self.get_state("cost_gate"),
        }
