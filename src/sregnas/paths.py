

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = (
    PACKAGE_ROOT.parent.parent
    if PACKAGE_ROOT.parent.name == "src"
    and (PACKAGE_ROOT.parent.parent / "pyproject.toml").is_file()
    else None
)
PROJECT_ROOT = (
    Path(os.environ.get("SREGNAS_ROOT", SOURCE_ROOT or Path.cwd()))
    .expanduser()
    .resolve()
)
DATA_DIR = (
    Path(os.environ.get("SREGNAS_DATA_DIR", PROJECT_ROOT / "data"))
    .expanduser()
    .resolve()
)
REFERENCE_DIR = PACKAGE_ROOT / "resources"


def require_gpu_permission():
    if os.environ.get("SREGNAS_ALLOW_GPU") != "1":
        raise RuntimeError(
            "GPU execution requires explicit approval and the run --allow-gpu option"
        )

GPU_LOCK = Path(os.environ.get("SREGNAS_GPU_LOCK", PROJECT_ROOT / ".runtime/gpu.lock")).expanduser().resolve()
