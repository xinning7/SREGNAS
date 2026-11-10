
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from artifact import ROOT, jobs, read, environment, command, external_missing, resolve_job_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--group")
    parser.add_argument("--job")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    rows = [j for j in jobs() if args.group is None or j["group"] == args.group]
    if args.list:
        for row in rows:
            print(f'{row["id"]}\t{row["method"]}\t{row["dataset"]}\t{row["group"]}')
        return 0
    matches = [j for j in rows if j["id"] == args.job]
    if len(matches) != 1:
        parser.error("select one --job from --list")
    job = matches[0]
    data = args.data.expanduser().resolve()
    default_output = ROOT / "outputs" / ("runs" if job["kind"] == "locality" else "searches") / job["id"]
    output = (args.output or default_output).expanduser().resolve()
    env = environment(job, data)
    cmd = command(job, output, data)
    report = dict(job=job["id"], source_run_id=job["source_run_id"],
                  command=cmd, output=str(output),
                  environment={k: v for k, v in env.items() if k.startswith(("SREGNAS_", "NUMBA_")) or k.endswith("NUM_THREADS")},
                  external_dependency_missing=job["needs_external"] and bool(external_missing()),
                  historical_supervisor_timeout_s=job.get("historical_supervisor_timeout_s"))
    if not args.execute:
        print(json.dumps(report, indent=2))
        return 0
    if output.exists():
        parser.error("output already exists; choose a fresh directory")
    if job["needs_external"] and external_missing():
        parser.error("authorized external baseline bundle is absent; see docs/EXTERNAL_DEPENDENCIES.md")
    from sregnas.icde27.x22.canonical_input import KEYS, manifest_path
    if not manifest_path(KEYS[job["dataset"]]).is_file():
        parser.error("canonical input absent; run scripts/prepare_data.py first")
    if job["method"] == "SREGNAS" and not any(Path(env["NUMBA_CACHE_DIR"]).rglob("*.nbc")):
        parser.error("compiled partition kernels absent; run prepare_data.py with this job's cache group")
    Path(env["SREGNAS_GPU_LOCK"]).parent.mkdir(parents=True, exist_ok=True)

    if job["kind"] == "locality":
        os.environ.update(env)
        from sregnas.icde27.x28 import a_runner
        frozen = resolve_job_paths(read(ROOT / job["config"]))
        frozen["cell_id"] = output.name
        a_runner.OUT = output.parent.parent
        if output.parent.name != "runs":
            parser.error("locality --output must end in runs/<new-run-name>")
        with Path(env["SREGNAS_GPU_LOCK"]).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            os.environ["SREGNAS_X28_LOCKED"] = "1"
            result = a_runner.run(frozen)
        return 0 if result["status"] == "OK" else 2
    if job["kind"] == "pool":
        frozen = resolve_job_paths(read(ROOT / job["config"]))
        if job.get("requires_degree_input") and not Path(frozen["graph_manifest"]).is_file():
            parser.error("degree inputs absent; use scripts/prepare_data.py --degree")
        runtime = ROOT / ".runtime/jobs"
        runtime.mkdir(parents=True, exist_ok=True)
        resolved = runtime / (job["id"] + ".json")
        resolved.write_text(json.dumps(frozen, indent=2) + "\n")
        cmd[cmd.index("--job") + 1] = str(resolved)
    output.parent.mkdir(parents=True, exist_ok=True)

    child = subprocess.Popen(cmd, cwd=ROOT, env=env, start_new_session=True)
    try:
        return child.wait(timeout=job.get("historical_supervisor_timeout_s"))
    except subprocess.TimeoutExpired:
        from sregnas.icde27.p2p6.entry import terminate
        terminate(child)
        print("Recorded outer per-run timeout reached.", file=sys.stderr)
        return 124
    finally:
        if child.poll() is None:
            from sregnas.icde27.p2p6.entry import terminate
            terminate(child)


if __name__ == "__main__":
    raise SystemExit(main())
