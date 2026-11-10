

import hashlib
import json
from pathlib import Path

from sregnas.paths import PROJECT_ROOT as ROOT
METHODS = ("SREGNAS", "BaselineFair", "BaselineFull", "FastGNAS")
REQUIRED_CHECKS = (
    "deterministic_port_equivalence",
    "sampling_inputs",
    "calibration",
    "resident_probe",
    "entry_smoke",
    "cost_projection",
    "ab_reuse",
)


def protocol_for_config(cfg):
    from sregnas.execution.options import resolve_executor

    return dict(
        dataset=cfg.graph.dataset,
        Q=cfg.search.q,
        N=cfg.graph.num_gpus,
        B=cfg.fixed_x or cfg.graph.num_gpus * cfg.selection.k,
        k=cfg.selection.k,
        search_seed=cfg.search.base_seed,
        E_proxy=cfg.train.proxy_epochs,
        retrain_passes=cfg.train.retrain_epochs,
        fanout=[cfg.graph.fanout_a, cfg.graph.fanout_b],
        batch_size=4096,
        partition_strategy=cfg.graph.partition_strategy,
        partition_seed=cfg.graph.partition_seed,
        partition_label_scope=cfg.graph.partition_label_scope,
        executor=resolve_executor(cfg),
        search_generator=cfg.search.generator,
        pool_mode=cfg.search.pool_mode,
        graph_parameters=cfg.to_dict()["graph"],
        search_parameters=cfg.to_dict()["search"],
        train_parameters=cfg.to_dict()["train"],
        selection_parameters=cfg.to_dict()["selection"],
        full_budget=cfg.full_budget,
        worker_counts=[len(cfg.runtime.system_gpus), len(cfg.runtime.baseline_gpus)],
        topology=cfg.runtime.topology_mode,
        execution_mode=cfg.runtime.execution_mode,
        scheduler=cfg.runtime.shared_search_scheduler,
        execution_scope=cfg.runtime.execution_scope,
    )


def assert_products_method_approval(cfg, method=None, *, hardware_path=None):
    if cfg.graph.dataset != "ogbn-products":
        return None
    method = method or cfg.runtime.comparison_method or "SREGNAS"
    if method not in METHODS:
        raise RuntimeError("unknown Products comparison method")
    if method == "SREGNAS" and cfg.runtime.execution_scope != "system":
        raise RuntimeError(
            "Products requires a separate approved method entry; legacy two-side runs are not approved"
        )
    declared = cfg.runtime.products_verification_reference
    if not declared:
        raise RuntimeError(
            f"Products {method} is not approved: no per-method approval reference"
        )
    path = Path(declared)
    if not path.is_absolute():
        path = ROOT / path
    payload = json.loads(path.read_text())
    if payload.get("schema") == "icde27_quality_products_v1":
        from sregnas.icde27.quality.approval import assert_quality_products
        return assert_quality_products(cfg, method, path, hardware_path)
    if payload.get("schema") == "products_user_directive_v1":
        from .products_protocol import assert_user_directive

        return assert_user_directive(cfg, method, path, hardware_path=hardware_path)
    if payload.get("schema") != "products_method_approval_v1":
        raise RuntimeError(
            "legacy/draft Products gate is not campaign-ready: per-method approval is required"
        )
    row = payload.get("methods", {}).get(method, {})
    if row.get("status") != "APPROVED":
        raise RuntimeError(f"Products {method} is not approved")
    approval = row.get("approval", {})
    if not approval.get("user_instruction") or not approval.get("approved_at"):
        raise RuntimeError(
            "Products approval must cite the actual user instruction and date"
        )
    if set(row.get("checks", {})) != set(REQUIRED_CHECKS) or any(
        row["checks"][key] != "PASS" for key in REQUIRED_CHECKS
    ):
        raise RuntimeError("Products method has incomplete or failed readiness checks")
    requested = protocol_for_config(cfg)
    if requested not in row.get("allowed_cells", []):
        raise RuntimeError(
            "Products method approval does not cover this resolved Q/N/B/seed/protocol"
        )
    if method == "SREGNAS" and (
        not cfg.runtime.require_fi_gpu
        or any(v != "gpu" for v in requested["executor"].values())
    ):
        raise RuntimeError(
            "approved SREGNAS Products entry requires FI and gpu/gpu/gpu"
        )
    if hardware_path is not None:
        profile_sha = hashlib.sha256(Path(hardware_path).read_bytes()).hexdigest()
        if row.get("hardware_profile_sha256") != profile_sha:
            raise RuntimeError(
                "Products method approval does not cover this hardware profile"
            )
    evidence = row.get("evidence", [])
    if not evidence:
        raise RuntimeError("Products approval has no hashed evidence")
    for item in evidence:
        artifact = Path(item["path"])
        if not artifact.is_absolute():
            artifact = ROOT / artifact
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError("Products approval evidence changed: " + str(artifact))
    budget = row.get("cost", {})
    if (
        not 0
        < float(budget.get("projected_gpu_h", 0))
        <= float(budget.get("approved_gpu_h", 0))
    ):
        raise RuntimeError(
            "Products projected cost is outside the method-approved budget"
        )
    return dict(
        method=method,
        approval_path=str(path.resolve()),
        protocol=requested,
        approval=row["approval"],
        cost=budget,
        campaign_ready=True,
    )
