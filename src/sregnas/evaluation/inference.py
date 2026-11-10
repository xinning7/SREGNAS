

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import torch


Split = Literal["val", "test"]
DIRECT_FULL_GRAPH_IMPLEMENTATION = "canonical_single_full_graph_forward"
PRODUCTS_EXACT_IMPLEMENTATION = (
    "sregnas.training.minibatch.exact_layerwise_full_neighbor_inference"
)


@dataclass(frozen=True)
class FullGraphInferenceScore:
    val_acc: float | None
    test_acc: float | None
    inference_s: float
    layer_batches: int
    implementation: str
    logits: torch.Tensor | None = None


def score_full_graph_inference(
    canonical_model: torch.nn.Module,
    *,
    dataset: str,
    device: torch.device,
    splits: tuple[Split, ...],
    x: torch.Tensor | None = None,
    edge_index: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    masks: dict[Split, torch.Tensor] | None = None,
    products_graph: Any | None = None,
    products_sampler: Any | None = None,
    products_batch_size: int | None = None,
    collect_logits: bool = False,
) -> FullGraphInferenceScore:
    pass
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("inference splits must be non-empty and unique")
    if not set(splits) <= {"val", "test"}:
        raise ValueError(f"unsupported inference splits: {splits}")

    if dataset.lower() in {"ogbn-products", "reddit"} and products_graph is not None:
        if products_graph is None or products_sampler is None:
            raise ValueError("products scoring requires graph and sampler")


        from sregnas.training import minibatch as products_training

        kwargs: dict[str, Any] = {
            "device": device,
            "splits": splits,
            "collect_logits": collect_logits,
        }
        if products_batch_size is not None:
            kwargs["batch_size"] = int(products_batch_size)
        result = products_training.exact_layerwise_full_neighbor_inference(
            canonical_model,
            products_graph,
            products_sampler,
            **kwargs,
        )
        return FullGraphInferenceScore(
            val_acc=result.val_acc,
            test_acc=result.test_acc,
            inference_s=result.inference_s,
            layer_batches=result.layer_batches,
            implementation=PRODUCTS_EXACT_IMPLEMENTATION,
            logits=result.logits,
        )

    if x is None or edge_index is None or labels is None or masks is None:
        raise ValueError(
            "direct full-graph scoring requires x, edge_index, labels, and masks"
        )
    missing = [split for split in splits if split not in masks]
    if missing:
        raise ValueError(f"missing inference masks: {missing}")
    started = time.perf_counter()
    canonical_model.eval()
    with torch.no_grad():
        logits = canonical_model(x, edge_index)
        scores: dict[str, float] = {}
        for split in splits:
            mask = masks[split]
            if not int(mask.sum()):
                raise AssertionError(f"full-graph inference split is empty: {split}")
            predictions = logits[mask].argmax(dim=1)
            scores[split] = float((predictions == labels[mask]).float().mean().item())
    return FullGraphInferenceScore(
        val_acc=scores.get("val"),
        test_acc=scores.get("test"),
        inference_s=time.perf_counter() - started,
        layer_batches=0,
        implementation=DIRECT_FULL_GRAPH_IMPLEMENTATION,
        logits=logits.detach().cpu() if collect_logits else None,
    )
