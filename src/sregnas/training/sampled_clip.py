
from contextlib import contextmanager
from dataclasses import dataclass
import math
import os

import torch

FLAG = "SREGNAS_SAMPLED_GRAD_CLIP"
SCHEMA = "sampled_final_grad_clip_v1"


def max_norm():
    value = float(os.environ.get(FLAG, "1.0"))
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{FLAG} must be finite and nonnegative")
    return value


@dataclass
class ClipState:
    max_norm: float
    calls: int = 0
    last_norm: object = None

    def accounting(self, optimizer_steps):
        expected = optimizer_steps if self.max_norm else 0
        assert self.calls == expected, (self.calls, expected)
        return dict(schema=SCHEMA, enabled=bool(self.max_norm),
                    max_norm=self.max_norm, calls=self.calls,
                    optimizer_steps=optimizer_steps, once_per_step=True)


@contextmanager
def final_training_clip(optimizer, *, dataset, mode):
    pass
    value = max_norm() if dataset in {"reddit", "ogbn-products"} and mode == "retrain" else 0.0
    state = ClipState(value)
    if not value:
        yield state
        return
    if hasattr(optimizer, "_sampled_final_clip_state"):
        raise AssertionError("nested sampled-final clipping")

    def before_step(opt, args, kwargs):
        parameters = [p for group in opt.param_groups for p in group["params"]]
        state.last_norm = torch.nn.utils.clip_grad_norm_(parameters, value)
        state.calls += 1

    optimizer._sampled_final_clip_state = state
    handle = optimizer.register_step_pre_hook(before_step)
    try:
        yield state
    finally:
        handle.remove()
        del optimizer._sampled_final_clip_state
