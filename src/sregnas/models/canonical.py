

import hashlib
import importlib
import inspect
import json
import os
import sys
from pathlib import Path

MODEL_PATH = Path(__file__).with_name("gnn.py").resolve()


def load_model_module(force_native_scatter=False):
    if force_native_scatter:
        import torch_geometric.typing as pyg_typing

        pyg_typing.WITH_TORCH_SCATTER = False
    module = importlib.import_module("sregnas.models.gnn")
    if Path(inspect.getfile(module.GNASModel)).resolve() != MODEL_PATH:
        raise AssertionError("unexpected canonical model implementation")
    return module


def assert_model(model, model_module):
    canonical = getattr(model, "canonical_model", model)
    if type(canonical) is not model_module.GNASModel:
        raise AssertionError("model identity failure")
    if Path(inspect.getfile(type(canonical))).resolve() != MODEL_PATH:
        raise AssertionError("model source identity failure")


def provenance(script_path, extra=None):
    from importlib.metadata import version, PackageNotFoundError

    from sregnas.execution.provenance import commit_hash

    git_hash = commit_hash()
    versions = {}
    for name in (
        "torch",
        "torch-geometric",
        "torch-scatter",
        "torch-sparse",
        "ogb",
        "numpy",
        "scipy",
    ):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    package_root = Path(__file__).resolve().parents[1]
    source_hashes = {
        str(p.relative_to(package_root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package_root.rglob("*.py"))
    }
    record = dict(
        argv=sys.argv,
        script_path=str(Path(script_path).resolve()),
        python_executable=str(Path(sys.executable).resolve()),
        cwd=str(Path.cwd()),
        git_hash=git_hash,
        pid=os.getpid(),
        canonical_model_path=str(MODEL_PATH),
        model_sha256=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        source_sha256=source_hashes,
        versions=versions,
    )
    record.update(extra or {})
    print("PROVENANCE " + json.dumps(record, sort_keys=True), flush=True)
    return record
