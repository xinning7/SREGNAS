
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def read(path):
    return json.loads(Path(path).read_text())


def jobs():
    return read(ROOT / "experiments/manifest.json")


def environment(job, data):

    saved = {k: v for k, v in os.environ.items() if k.startswith("SREGNAS_")}
    for key in saved:
        os.environ.pop(key)
    try:
        os.environ["SREGNAS_DATA_DIR"] = str(data)
        if "SREGNAS_GPU_LOCK" in saved:
            os.environ["SREGNAS_GPU_LOCK"] = saved["SREGNAS_GPU_LOCK"]
        if job.get("environment_family") == "x18":
            from sregnas.icde27.x18.common import environment as recorded_environment
            env = recorded_environment(ROOT, job.get("variant"), job.get("dataset"))
            env.update(job.get("environment", {}))
        elif job.get("environment_family") == "x23":
            from sregnas.icde27.x23.common import environment as recorded_environment
            env = recorded_environment(ROOT, job.get("variant"), job.get("dataset"), job.get("environment", {}))
        else:
            from sregnas.icde27.x25.common import environment as recorded_environment
            env = recorded_environment(ROOT, job.get("variant"), job.get("dataset"), job.get("environment", {}))
    finally:
        for key in tuple(os.environ):
            if key.startswith("SREGNAS_"):
                os.environ.pop(key)
        os.environ.update(saved)
    env.update(job.get("numeric_environment", {}))
    env.update(SREGNAS_ROOT=str(ROOT), SREGNAS_DATA_DIR=str(data),
               PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
               SREGNAS_GPU_LOCK=saved.get("SREGNAS_GPU_LOCK", str(ROOT / ".runtime/gpu.lock")))
    if job.get("cache_group"):
        env["NUMBA_CACHE_DIR"] = str(ROOT / "results/icde27" / job["cache_group"] / "numba_cache")
    return env


def command(job, output, data):
    args = []
    for arg in job["argv"]:
        arg = arg.replace("{output}", str(output)).replace("{data}", str(data))
        if arg.startswith(("configs/", "pools/")):
            arg = str(ROOT / arg)
        args.append(arg)
    return [sys.executable, *args]


def external_missing():
    manifest = read(ROOT / "provenance/external_files.json")
    return [r["target"] for r in manifest["files"] if not (ROOT / r["target"]).is_file()]


def resolve_job_paths(value):
    if isinstance(value, list):
        return [resolve_job_paths(v) for v in value]
    if isinstance(value, dict):
        return {k: resolve_job_paths(v) for k, v in value.items()}
    if isinstance(value, str) and value.startswith(("results/", "configs/", "pools/")):
        return str(ROOT / value)
    return value
