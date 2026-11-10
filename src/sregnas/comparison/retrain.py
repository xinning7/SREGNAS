

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time


def native_seed(architecture, replica):
    encoded = json.dumps(
        [architecture, "retrain", replica], sort_keys=True, separators=(",", ":")
    )
    return int.from_bytes(hashlib.sha256(encoded.encode()).digest()[:8], "big") % (
        2**31
    )


def native_cell(spec, device, *, runtime=None, payloads=None):
    import torch
    from sregnas.baselines.fastgnas.trainer.builder import ModelBuilder
    from sregnas.training.minibatch import (
        ProductsRuntimeCache,
        set_retrain_seed,
        train_products_minibatch,
        sampled_inference_batch_size,
    )
    from sregnas.training.native import FastPerLayerModel, exact_fastgnas_inference
    from sregnas.evaluation.checkpoints import (
        save_retrain_checkpoint,
        load_retrain_checkpoint,
    )
    from sregnas.execution.startup import assert_project_module_origins

    assert_project_module_origins()
    testing = spec["mode"] == "post_selection_test"
    seed = native_seed(spec["architecture"], spec["replica"])
    set_retrain_seed(seed)
    sampled = spec["dataset"] in {"reddit", "ogbn-products"}
    if sampled:
        if spec["epochs"] != 25:
            raise ValueError("sampled retrain is locked to 25 complete passes")
        cache = runtime or ProductsRuntimeCache()
        graph, sampler, _ = cache.get(
            spec["graph_path"],
            device=device,
            **{
                key: spec[key]
                for key in ("sampler_impl", "feature_residency", "inference_residency")
            },
        )
        features, classes = graph.num_features, graph.num_classes
    else:
        payloads = payloads if payloads is not None else {}
        if spec["graph_path"] not in payloads:
            raw = torch.load(spec["graph_path"], map_location="cpu", weights_only=False)
            payloads[spec["graph_path"]] = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in raw.items()
            }
        data = payloads[spec["graph_path"]]
        features, classes = data["num_features"], data["num_classes"]
    canonical = ModelBuilder.build_model(
        spec["architecture"], features, classes, dict(dropout=spec["dropout"])
    ).to(device)
    model = FastPerLayerModel(canonical).to(device) if sampled else canonical
    identity = dict(
        fingerprint=spec["fingerprint"],
        architecture=spec["architecture"],
        dataset=spec["dataset"],
    )
    if testing:
        load_retrain_checkpoint(spec["checkpoint_path"], model, **identity)
    else:
        opt = torch.optim.Adam(
            model.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"]
        )
        if sampled:
            from sregnas.training.sampled_clip import final_training_clip
            with final_training_clip(opt, dataset=spec["dataset"], mode=spec["mode"]) as clipping:
                trained = train_products_minibatch(
                    model,
                    graph,
                    sampler,
                    opt,
                    device=device,
                    root_seed=seed,
                    passes=spec["epochs"],
                )
            accounting = asdict(trained)
            accounting["sampled_grad_clip"] = clipping.accounting(trained.optimizer_steps)
        else:
            for _ in range(spec["epochs"]):
                model.train()
                opt.zero_grad(set_to_none=True)
                logits = model(data["x"], data["edge_index"])
                loss = torch.nn.functional.cross_entropy(
                    logits[data["train_mask"]], data["labels"][data["train_mask"]]
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite retrain loss")
                loss.backward()
                opt.step()
            accounting = dict(optimizer_steps=spec["epochs"])
        save_retrain_checkpoint(spec["checkpoint_path"], model, **identity)
    split = "test" if testing else "val"
    if sampled:
        score = exact_fastgnas_inference(
            canonical,
            graph,
            sampler,
            device=device,
            splits=(split,),
            batch_size=sampled_inference_batch_size(spec["dataset"]),
        )
        value = getattr(score, split + "_acc")
    else:
        model.eval()
        with torch.no_grad():
            mask = data[split + "_mask"]
            logits = model(data["x"], data["edge_index"])
            count = int(mask.sum())
            if not count:
                raise AssertionError("empty scoring mask")
            value = float(
                (logits[mask].argmax(1) == data["labels"][mask]).sum().item() / count
            )
    return dict(
        status="OK",
        value=value,
        metrics={
            split + "_acc": value,
            "checkpoint_path": spec["checkpoint_path"],
            "seed": seed,
            "test_evaluations": int(testing),
            **({} if testing else {"train_accounting": accounting}),
        },
    )


def run_job(job, device):
    from sregnas.execution.startup import assert_cell_start
    from sregnas.evaluation.checkpoints import assert_test_metric_isolated
    from sregnas.training.fullbatch import run_cell, PersistentCellRuntime
    from sregnas.training.minibatch import run_products_cell, ProductsRuntimeCache

    from sregnas.icde27.ledger import call as recorded_call

    results = []
    sampled_runtime = ProductsRuntimeCache()
    full_runtime = None
    payloads = {}
    for spec in job["specs"]:
        started = time.perf_counter()
        startup = assert_cell_start(spec)
        try:
            if job["method"] == "FastGNAS":
                result = recorded_call(
                    spec, native_cell, device, backend="fastgnas-native", runtime=sampled_runtime, payloads=payloads
                )
            elif spec["dataset"] in {"reddit", "ogbn-products"}:
                result = asdict(
                    recorded_call(spec, run_products_cell, device=device, runtime=sampled_runtime)
                )
            else:
                if full_runtime is None:
                    full_runtime = PersistentCellRuntime()
                result = asdict(recorded_call(spec, run_cell, runtime=full_runtime))
        except Exception as exc:
            import torch

            if isinstance(exc, torch.cuda.OutOfMemoryError):
                status = "OOM"
            elif isinstance(exc, FloatingPointError):
                status = "DIVERGE"
            else:
                raise
            result = dict(status=status, error=str(exc), value=None, metrics={})
        assert_test_metric_isolated(spec["mode"], result["metrics"])
        from sregnas.execution.provenance import commit_hash

        result.update(
            commit_hash=commit_hash(),
            id=spec["fingerprint"],
            architecture=spec["architecture"],
            wall_s=time.perf_counter() - started,
        )
        result["metrics"]["cell_start"] = startup
        results.append(result)
        Path(job["output"]).write_text(json.dumps(results, indent=2))
        import os

        if os.environ.get("SREGNAS_STOP_ON_INVALID") == "1" and result[
            "status"
        ] not in {"OK", "OOM"}:
            raise RuntimeError(f"invalid retrain result: {result}")
        import gc, torch

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args(argv)
    from sregnas.comparison.runtime import configure_native_client

    configure_native_client()
    import torch

    if os.environ.get('SREGNAS_X8_DETERMINISTIC_DIAGNOSTIC') == '1':
        torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    run_job(json.loads(args.job.read_text()), torch.device("cuda:0"))


if __name__ == "__main__":
    main()
