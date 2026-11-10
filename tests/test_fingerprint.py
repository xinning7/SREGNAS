import json

import pytest

from sregnas.core import Architecture


def _reordered_json_copy(architecture: Architecture):
    return json.loads(json.dumps(architecture.arch_description, sort_keys=True))


def _assert_mapping_order_independent(architecture: Architecture) -> None:
    assert (
        Architecture(_reordered_json_copy(architecture)).fingerprint()
        == architecture.fingerprint()
    )


@pytest.mark.unit
def test_json_round_trip_and_direct_construction_are_order_independent():
    canonical = [
        {
            "encoding_op": "gat",
            "aggregation_op": "mean",
            "activation": "elu",
            "num_heads": 8,
            "hidden_dim": 256,
        },
        {
            "encoding_op": "const",
            "aggregation_op": "sum",
            "activation": "none",
            "num_heads": 1,
            "hidden_dim": 64,
        },
    ]
    reordered = [dict(reversed(tuple(layer.items()))) for layer in canonical]
    expected = Architecture(canonical).fingerprint()
    assert Architecture(reordered).fingerprint() == expected
    assert Architecture(json.loads(json.dumps(canonical))).fingerprint() == expected
    assert (
        Architecture(json.loads(json.dumps(canonical, sort_keys=True))).fingerprint()
        == expected
    )


@pytest.mark.unit
def test_historical_canonical_digest_is_unchanged():
    architecture = [
        {
            "encoding_op": "const",
            "aggregation_op": "max",
            "activation": "none",
            "num_heads": 8,
            "hidden_dim": 64,
        },
        {
            "encoding_op": "gcn",
            "aggregation_op": "sum",
            "activation": "leaky_relu",
            "num_heads": 1,
            "hidden_dim": 128,
        },
    ]
    assert Architecture(architecture).fingerprint() == "56e56fd3777c6d3d"
