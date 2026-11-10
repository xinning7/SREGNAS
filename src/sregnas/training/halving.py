

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from sregnas.core import Architecture
from sregnas.models.canonical import assert_model
from sregnas.core.contracts import CellResult, Status
from sregnas.evaluation.inference import score_full_graph_inference
from sregnas.training.fullbatch import (
    CELL_TIMING_FIELDS,
    PersistentCellRuntime,
    _memory,
    _record_retrain_validation,
    _set_seed,
    fingerprint_seed,
    fingerprint_retrain_replica_seed,
)
from sregnas.evaluation.checkpoints import (
    assert_test_metric_isolated,
    save_retrain_checkpoint,
)

S1_STAGE_MODE = "retrain_s1_stage"
S1_EPOCHS = (50, 100, 200)
FULL_BATCH_DATASETS = {"cora", "citeseer", "pubmed", "flickr", "ogbn-arxiv"}
STATE_SCHEMA = "distnas_v2_s1_training_state_v1"


def protocol_identity(spec: dict) -> dict:
    pass
    dataset = str(spec["dataset"]).lower()
    if dataset not in FULL_BATCH_DATASETS:
        raise ValueError("S1 supports full-batch datasets only; products is unresolved")
    if int(spec["epochs"]) != 200:
        raise ValueError("S1 requires the unchanged 200-epoch protocol")
    if float(spec["lr"]) != 0.01 or float(spec["weight_decay"]) != 0.0005:
        raise ValueError("S1 requires Adam lr=.01, weight_decay=.0005")
    if Architecture(spec["architecture"]).fingerprint() != spec["fingerprint"]:
        raise ValueError("S1 architecture/fingerprint mismatch")
    policy = spec.get("retrain_seed_policy", "legacy_fingerprint")
    if "retrain_seed" in spec:
        seed, seed_policy = int(spec["retrain_seed"]), "explicit_retrain_seed"
    elif policy == "fingerprint_retrain_replica":
        seed = fingerprint_retrain_replica_seed(
            spec["fingerprint"], int(spec.get("replica", 0))
        )
        seed_policy = "H(fingerprint,retrain,replica);side_excluded"
    elif policy == "legacy_fingerprint":
        seed, seed_policy = fingerprint_seed(spec["fingerprint"]), "fingerprint"
    else:
        raise ValueError(f"unknown retrain seed policy: {policy}")
    graph = Path(spec["graph_path"]).resolve()
    stat = graph.stat()
    sources = [
        Path(__file__),
        Path(__file__).with_name("fullbatch.py"),
        Path(__file__).resolve().parents[1] / "evaluation/inference.py",
        Path(__file__).resolve().parents[1] / "models/gnn.py",
    ]
    return dict(
        dataset=dataset,
        architecture=spec["architecture"],
        fingerprint=spec["fingerprint"],
        epochs=200,
        lr=float(spec["lr"]),
        weight_decay=float(spec["weight_decay"]),
        dropout=float(spec["dropout"]),
        seed=seed,
        seed_policy=seed_policy,
        graph_path=str(graph),
        graph_size=stat.st_size,
        graph_mtime_ns=stat.st_mtime_ns,
        torch_version=str(torch.__version__),
        implementation_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
    )


def rank_stage(results: list[CellResult]) -> list[CellResult]:
    if not results or len({r.fingerprint for r in results}) != len(results):
        raise ValueError("stage needs nonempty, unique candidates")
    for row in results:
        if row.status != Status.OK or row.value is None or not math.isfinite(row.value):
            raise ValueError(
                f"S1 stage failed; do not silently drop {row.fingerprint}: {row.status}"
            )
        assert_test_metric_isolated(S1_STAGE_MODE, row.metrics)
    return sorted(results, key=lambda r: (-float(r.value), r.fingerprint))


def _cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_tree(v) for v in value)
    return value


def _rng_state(device):
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.random.get_rng_state(),
        cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    )


def _restore_rng(state, device):
    if (state["cuda"] is None) != (device.type != "cuda"):
        raise ValueError("S1 cannot migrate a training trajectory between CPU and CUDA")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def run_s1_stage(
    spec: dict, runtime: PersistentCellRuntime | None = None
) -> CellResult:
    pass
    if spec.get("mode") != S1_STAGE_MODE:
        raise ValueError("S1 stage requires explicit opt-in mode")
    identity = protocol_identity(spec)
    target = int(spec["s1_target_epoch"])
    if target not in S1_EPOCHS:
        raise ValueError("S1 targets are exactly 50,100,200")
    start = {50: 0, 100: 50, 200: 100}[target]
    resume = spec.get("s1_resume_path")
    if bool(resume) != bool(start):
        raise ValueError("S1 later stages require the preceding training state")
    entered = time.perf_counter()
    launch = float(spec.get("parent_launch_perf_s", entered))
    runtime = runtime if runtime is not None else PersistentCellRuntime()
    device = runtime.device
    timings = {k: 0.0 for k in CELL_TIMING_FIELDS}
    timings["cuda_init_s"] = runtime.claim_startup_cuda_init_s()
    _set_seed(identity["seed"])
    payload, load_wall, cache_hit = runtime.load_payload(spec["graph_path"], "retrain")
    timings["data_load_h2d_s"] = load_wall
    model = runtime.model_module.build_model(
        Architecture(spec["architecture"]),
        payload["num_features"],
        payload["num_classes"],
        dropout=identity["dropout"],
    ).to(device)
    assert_model(model, runtime.model_module)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=identity["lr"], weight_decay=identity["weight_decay"]
    )
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if (
            state["schema"] != STATE_SCHEMA
            or state["identity"] != identity
            or state["epoch"] != start
        ):
            raise ValueError("S1 resume protocol, identity, or epoch mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        modules = dict(model.named_modules())
        if set(modules) != set(state["training_modes"]):
            raise ValueError("S1 module topology changed")
        for name, training in state["training_modes"].items():
            modules[name].training = training
        _restore_rng(state["rng"], device)
        del state
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    compute = time.perf_counter()
    timings["process_spawn_s"] = max(
        0.0, compute - launch - timings["cuda_init_s"] - load_wall
    )
    for _ in range(start, target):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(payload["x"], payload["edge_index"])
        loss = F.cross_entropy(
            out[payload["train_mask"]], payload["labels"][payload["train_mask"]]
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite S1 full loss: {loss.item()}")
        loss.backward()
        optimizer.step()
    score_kwargs = dict(
        dataset=identity["dataset"],
        device=device,
        x=payload["x"],
        edge_index=payload["edge_index"],
        labels=payload["labels"],
        masks={"val": payload["val_mask"]},
    )
    if target < 200:
        value, inference_s = _record_retrain_validation(model, **score_kwargs)
        implementation = "existing_full_graph_validation_observer"
    else:
        inferred = score_full_graph_inference(model, splits=("val",), **score_kwargs)
        value, inference_s, implementation = (
            inferred.val_acc,
            inferred.inference_s,
            inferred.implementation,
        )
    if value is None or not math.isfinite(value):
        raise ValueError("S1 validation produced no finite score")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings["train_compute_s"] = time.perf_counter() - compute
    save_started = time.perf_counter()
    metrics = dict(
        val_acc=value,
        epochs_ran=target,
        stage_epochs_ran=target - start,
        start_epoch=start,
        retrain_schedule="S1",
        scorer=f"full_val@{target}",
        cross_shard_comparable=True,
        seed=identity["seed"],
        seed_policy=identity["seed_policy"],
        graph_cache_hit=cache_hit,
        exact_inference_s=inference_s,
        inference_implementation=implementation,
        test_isolation="RETRAIN_VAL_ONLY",
        selection_eligible=target == 200,
        train_wall_s=timings["train_compute_s"],
        s1_protocol_identity=identity,
    )
    if target < 200:
        path = Path(spec["s1_state_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        rng = _rng_state(device)
        torch.save(
            dict(
                schema=STATE_SCHEMA,
                identity=identity,
                epoch=target,
                model=_cpu_tree(model.state_dict()),
                optimizer=_cpu_tree(optimizer.state_dict()),
                training_modes={name: m.training for name, m in model.named_modules()},
                rng=rng,
            ),
            temp,
        )
        temp.replace(path)
        metrics["s1_state_path"] = str(path)
    else:
        path = save_retrain_checkpoint(
            spec["checkpoint_path"],
            model,
            fingerprint=spec["fingerprint"],
            architecture=spec["architecture"],
            dataset=identity["dataset"],
        )
        metrics["checkpoint_path"] = str(path)
    metrics["state_persistence_s"] = time.perf_counter() - save_started
    allocated, reserved = _memory(device)
    del model, optimizer, out, loss
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings["teardown_s"] = time.perf_counter() - save_started
    metrics.update(
        timings,
        cuda_boundary_synchronized=True,
        cuda_boundary_contract=(
            "torch.cuda.synchronize before stage return"
            if device.type == "cuda"
            else "cpu_no_cuda_work"
        ),
    )
    assert_test_metric_isolated(S1_STAGE_MODE, metrics)
    return CellResult(
        status=Status.OK,
        fingerprint=spec["fingerprint"],
        phase=spec["phase"],
        gpu=int(spec["gpu"]),
        round=int(spec["round"]),
        shard=int(spec["shard"]),
        architecture=spec["architecture"],
        value=value,
        wall_s=sum(timings.values()),
        peak_allocated_gb=allocated,
        peak_reserved_gb=reserved,
        metrics=metrics,
    )
