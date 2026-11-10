

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict






_GNAS_FINGERPRINT_FIELD_ORDER = (
    "encoding_op",
    "aggregation_op",
    "activation",
    "num_heads",
    "hidden_dim",
)
_GNAS_FINGERPRINT_FIELD_RANK = {
    key: index for index, key in enumerate(_GNAS_FINGERPRINT_FIELD_ORDER)
}


def _fingerprint_key_order(key: Any) -> tuple[int, str, str, str]:
    pass

    return (
        _GNAS_FINGERPRINT_FIELD_RANK.get(key, len(_GNAS_FINGERPRINT_FIELD_RANK)),
        type(key).__module__,
        type(key).__qualname__,
        repr(key),
    )


def _normalize_for_fingerprint(value: Any) -> Any:
    pass

    if isinstance(value, Mapping):
        return {
            key: _normalize_for_fingerprint(value[key])
            for key in sorted(value, key=_fingerprint_key_order)
        }
    if isinstance(value, list):
        return [_normalize_for_fingerprint(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_for_fingerprint(item) for item in value)
    return value


@dataclass(frozen=True)
class Architecture:
    pass

    arch_description: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        pass
        content = repr(_normalize_for_fingerprint(self.arch_description))
        return hashlib.sha256(content.encode()).hexdigest()[:16]
