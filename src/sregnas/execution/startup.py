

import hashlib
import json
import math
import os
import sys
from pathlib import Path

from sregnas.training.executor import DEFAULTS, OPTIONS
from sregnas.paths import PACKAGE_ROOT, SOURCE_ROOT


def partition_receipt(path):
    path = Path(path).resolve()
    content = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}


_PROJECT_NAMES = {"sregnas", "distnas", "trainer", "core", "experiment", "fastgnas"}


def assert_project_module_origins():
    pass
    if os.environ.get('SREGNAS_X22_INCREMENTAL_AUDIT') == '1':
        from sregnas.icde27.x22.origins import audit
        return audit()
    root = SOURCE_ROOT or PACKAGE_ROOT
    origins = {}
    rejected = {}
    resolve = lambda value: Path(value).resolve()
    if os.environ.get('SREGNAS_X20_ORIGIN_PREFIX_CACHE') == '1':
        from sregnas.icde27.x20.origins import resolver
        resolve = resolver()
    for name, module in tuple(sys.modules.items()):
        if module is None:
            continue
        paths = set()
        attributes = vars(module)
        for value in (
            attributes.get("__file__"),
            getattr(attributes.get("__spec__"), "origin", None),
        ):
            if value and value not in {"built-in", "frozen"}:
                paths.add(resolve(value))
        for value in attributes.get("__path__", ()):
            paths.add(resolve(value))
        project = name.split(".")[0].lower() in _PROJECT_NAMES or any(
            path.is_relative_to(root)
            or any(
                part.lower() in {"distnas", "sregnas", "fastgnas"}
                or part.lower().startswith(("distnas-", "fastgnas_snapshot"))
                for part in path.parts
            )
            and "site-packages" not in path.parts
            and ".venv" not in path.parts
            for path in paths
        )
        if not project:
            continue
        origins[name] = sorted(map(str, paths))
        if not paths or any(not path.is_relative_to(root) for path in paths):
            rejected[name] = origins[name]
    if rejected:
        raise AssertionError(
            "project modules outside SREGNAS: " + json.dumps(rejected, sort_keys=True)
        )
    return {"root": str(root), "loaded_modules": origins, "assertion": "PASS"}


def assert_cell_start(spec):
    from .provenance import commit_hash

    module_origins = assert_project_module_origins()
    flags = {key: spec.get(key, default) for key, default in DEFAULTS.items()}
    for key, value in flags.items():
        if value not in OPTIONS[key]:
            raise AssertionError(f"invalid cell executor {key}={value}")
    if spec.get("expected_executor", flags) != flags:
        raise AssertionError("cell executor differs from the resolved campaign")
    strategy = spec.get("partition_strategy")
    receipt = spec.get("partition_receipt")
    selected = None
    if receipt:
        path = Path(receipt["path"])
        if __import__('os').environ.get('SREGNAS_X22_INCREMENTAL_AUDIT') == '1':
            from sregnas.icde27.x22.receipt_cache import load
            manifest = load(path, receipt['sha256'])
        else:
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != receipt["sha256"]:
                raise AssertionError("partition manifest changed after graph preparation")
            manifest = json.loads(content)
        if (
            manifest["partition_strategy"] != strategy
            or manifest["dataset"] != spec["dataset"]
            or manifest["num_shards"] != spec["partition_parts"]
        ):
            raise AssertionError("cell partition differs from the prepared graph")
        if strategy == "fi":
            choice = manifest["partition_selection"]
            if (
                choice["requested_strategy"] != "fi"
                or choice["random_seed"] != spec["partition_seed"]
                or choice["label_scope"] != spec["partition_label_scope"]
                or choice["test_labels_used"] is not False
            ):
                raise AssertionError("cell FI protocol differs from graph selection")
            scores = choice["scores"]
            if choice.get("partition_mode") == "proxy_aware":
                from sregnas.data.proxy_partition_mode import validate_choice
                selected = validate_choice(choice)
            else:
                if not all(math.isfinite(scores[key]["FI"]) for key in ("metis", "random")):
                    raise AssertionError("nonfinite FI score")
                selected = max(("metis", "random"), key=lambda key: scores[key]["FI"])
                if selected != choice["selected"]:
                    raise AssertionError("prepared FI choice does not maximize FI")
    if spec.get("require_fi_gpu"):
        if (
            strategy != "fi"
            or not receipt
            or any(value != "gpu" for value in flags.values())
        ):
            raise AssertionError("formal cell requires verified FI and gpu/gpu/gpu")
    sampled = (
        spec.get("dataset") == "ogbn-products"
        and spec.get("mode")
        not in {"proxy", "native_search", "full_neighbor_feasibility"}
        and not (
            spec.get("mode") == "full"
            and spec.get("comparison_method") == "BaselineFull"
        )
    ) or (
        spec.get("dataset") == "reddit"
        and spec.get("mode") in {"retrain", "probe", "post_selection_test"}
    )
    record = dict(
        commit_hash=commit_hash(),
        phase=spec.get("phase"),
        mode=spec.get("mode"),
        dataset=spec.get("dataset"),
        partition_strategy=strategy,
        partition_selected=selected,
        partition_receipt=receipt,
        **flags,
        executor_applies=sampled,
        execution_path=(
            "native_search"
            if spec.get("mode") == "native_search"
            else "sampled_executor" if sampled else "full_batch"
        ),
        required_fi_gpu=bool(spec.get("require_fi_gpu")),
        project_module_origins=module_origins,
        assertion="PASS",
    )
    if __import__('os').environ.get('SREGNAS_X22_DEFER_RECORDS') == '1':
        from sregnas.icde27.x22.deferred import audit_log
        audit_log('CELL_START ', record)
    else:
        print("CELL_START " + json.dumps(record, sort_keys=True), flush=True)
    return record
