

from types import SimpleNamespace

from sregnas.training.executor import DEFAULTS, apply_executor_arguments


def resolve_executor(cfg, args=None):
    spec = {
        key: getattr(cfg.runtime, key)
        for key in DEFAULTS
        if getattr(cfg.runtime, key) is not None
    }
    resolved = apply_executor_arguments(args or SimpleNamespace(), spec)
    for key, value in resolved.items():
        setattr(cfg.runtime, key, value)
    cfg.validate()
    return resolved
