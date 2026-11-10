

import hashlib
import json
from pathlib import Path

from sregnas.paths import PROJECT_ROOT as ROOT
DIRECTIVE = ROOT / "configs/protocol/products.json"
DECISION = "Frozen Products compatibility contract v1"


def confirmed_protocol(method, reference=None):
    path = Path(reference) if reference else DIRECTIVE
    if not path.is_absolute():
        path = ROOT / path
    record = json.loads(path.read_text())
    if (
        record.get("schema") != "products_user_directive_v1"
        or record.get("user_instruction") != DECISION
    ):
        raise AssertionError("missing recorded user directive")
    if record.get("methods", {}).get(method, {}).get("protocol_status") != "CONFIRMED":
        raise AssertionError("Products method protocol is not confirmed")
    source = record["protocol_source"]
    if (
        hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest()
        != source["sha256"]
    ):
        raise AssertionError("confirmed Products protocol source changed")
    return record


def assert_user_directive(cfg, method, reference, *, hardware_path=None):
    from .products_approval import protocol_for_config

    record = confirmed_protocol(method, reference)
    p = protocol_for_config(cfg)
    if not (
        p["Q"] == record["Q"]
        and p["N"] in record["N"]
        and p["B"] == record["B"]
        and p["k"] == 8 // p["N"]
        and p["search_seed"] in record["seeds"]
        and p["E_proxy"] == 100
        and p["retrain_passes"] == 25
        and p["fanout"] == [5, 5]
        and p["search_generator"] == "rl"
        and p["pool_mode"] == "independent"
        and p["topology"] == "native"
        and p["worker_counts"] == [p["N"], p["N"]]
    ):
        raise AssertionError("Products configuration differs from confirmed protocol")
    if any(v != "gpu" for v in p["executor"].values()):
        raise AssertionError("Products retrain requires gpu/gpu/gpu")
    if (
        not cfg.runtime.require_idle_preflight
        or not cfg.runtime.require_native_topology
    ):
        raise AssertionError("Products requires idle GPUs and native topology")
    if method == "SREGNAS" and (
        not cfg.runtime.require_fi_gpu or p["partition_strategy"] != "fi"
    ):
        raise AssertionError("SREGNAS requires FI at every cell")
    if hardware_path is not None:
        import yaml

        machine = yaml.safe_load(Path(hardware_path).read_text())
        if machine.get("name_contains") != "A6000":
            raise AssertionError("user directive covers idle A6000 only")
    return dict(
        method=method,
        authorization="recorded_user_directive",
        directive_path=str(Path(reference).resolve()),
        protocol=p,
        protocol_confirmed=True,
        readiness_checks_waived=record["supersedes"],
    )


def assert_native_products_config(config):
    confirmed_protocol("FastGNAS")
    t, b, nas = config["training"], config["fastgnas_backends"], config["nas"]
    expected = dict(
        batch_size=256,
        eval_batch_size=256,
        fan_out=[15, 10],
        epochs_per_round=5,
        steps_per_epoch=10,
        num_rounds=1,
        num_classes=47,
    )
    if any(t.get(k) != v for k, v in expected.items()):
        raise AssertionError("FastGNAS Products native candidate budget changed")
    if (
        nas.get("gnn_layers") != 2
        or b.get("learning_rate") != 0.01
        or b.get("dropout") != 0.6
    ):
        raise AssertionError("FastGNAS Products model/training definition changed")
    if b.get("partition_strategy") != "balanced_stratified_khop":
        raise AssertionError("FastGNAS comparison partition definition changed")
