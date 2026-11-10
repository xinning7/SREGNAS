
import argparse
import fcntl
import os
from pathlib import Path

from .plan import AUTHORIZATION, read, write


def make_job(cell, source):
    from sregnas.icde27.quality.records import source_records
    from sregnas.icde27.quality.tables import final_accuracy
    from sregnas.icde27.recorder import sha
    info, _, specs = source_records(source)
    _, _, winner = final_accuracy(source, info)
    matches = [s for s in specs if s["fingerprint"] == winner]
    if not matches or any(s != matches[0] for s in matches):
        raise ValueError("missing/ambiguous seed42 winner retrain specification")
    return dict(kind="winner_replica", experiment_id="X15", record_type="reference",
                object_id=f"X15_{cell['dataset']}_PAS-PL", method="PAS-PL", dataset=cell["dataset"],
                replica=cell["replica"], gpus=[cell["gpu"]], K=1, config=cell["config"],
                source=str(source), source_run_id=info["run_id"], source_run_sha256=sha(source / "run.json"),
                source_record_type=info["record_type"], source_Q_attempt=info["Q_attempt"],
                source_Q_trunc=info.get("Q_trunc"), template=matches[0], quality_only=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    parser.add_argument("--gpu-lock", type=Path, required=True)
    args = parser.parse_args(argv)
    if os.environ.get("SREGNAS_X15_AUTHORIZATION") != AUTHORIZATION:
        parser.error("use the authorized X15 phase-2 driver")
    from sregnas.icde27.recorder import LOCK
    with LOCK.open("a") as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise RuntimeError("replica group must already hold the global GPU lock")
    plan = read(args.work / "plan.json")
    state = read(args.work / "state.json")
    cell = next(c for c in plan["cells"] if c["cell_id"] == args.cell)
    parent = state["cells"][cell["requires"]]
    if parent["status"] != "OK":
        raise ValueError("seed42 source did not finish successfully")
    job = make_job(cell, Path(parent["result_path"]))
    write(args.work / "jobs" / (args.cell + ".json"), job)
    from sregnas.icde27.quality.entry import record
    return record(job, args.out, args.max_wall_seconds, args.gpu_lock)


if __name__ == "__main__":
    raise SystemExit(main())
