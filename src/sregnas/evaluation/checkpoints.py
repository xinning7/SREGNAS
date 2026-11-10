

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


CHECKPOINT_SCHEMA = "distnas_v2_post_selection_checkpoint_v1"
POST_SELECTION_TEST_MODE = "post_selection_test"


def save_retrain_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    fingerprint: str,
    architecture: list[dict[str, Any]],
    dataset: str,
) -> Path:
    pass
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "fingerprint": fingerprint,
            "architecture": architecture,
            "dataset": dataset,
            "state_dict": state,
        },
        temporary,
    )
    temporary.replace(destination)
    return destination


def load_retrain_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    fingerprint: str,
    architecture: list[dict[str, Any]],
    dataset: str,
) -> None:
    pass
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported post-selection checkpoint schema")
    expected = {
        "fingerprint": fingerprint,
        "architecture": architecture,
        "dataset": dataset,
    }
    observed = {name: payload.get(name) for name in expected}
    if observed != expected:
        raise AssertionError(
            f"post-selection checkpoint identity mismatch: {observed} != {expected}"
        )
    model.load_state_dict(payload["state_dict"], strict=True)


def assert_test_metric_isolated(mode: str, metrics: dict[str, Any]) -> None:
    pass
    explicit_test_modes = {POST_SELECTION_TEST_MODE, "reference_control"}
    if mode not in explicit_test_modes and "test_acc" in metrics:
        raise AssertionError(
            f"mode={mode} computed test_acc outside an explicit test step"
        )
