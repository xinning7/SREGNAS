
import os
import math


def partition_mode():
    mode = os.environ.get("SREGNAS_PARTITION_MODE", "legacy")
    if mode not in ("legacy", "proxy_aware"):
        raise ValueError("SREGNAS_PARTITION_MODE must be legacy or proxy_aware")
    return mode


def validate_choice(choice):
    pass
    if 'forced_quality_arm' in choice:
        from sregnas.icde27.x25.partition import validate_explicit_arm
        return validate_explicit_arm(choice)
    scores = choice["scores"]
    da = scores["da_random"]
    assert all(math.isfinite(da[k]) for k in ("SNR", "LA", "FI")), "nonfinite DA score"
    if da["SNR"] >= .9:
        selected = "da_random"
        assert "la_metis" not in scores, "METIS must be skipped when DA passes"
    else:
        la = scores["la_metis"]
        assert all(math.isfinite(la[k]) for k in ("SNR", "LA", "FI")), "nonfinite LA score"
        selected = "la_metis" if la["FI"] > da["FI"] else "da_random"
    assert selected == choice["selected"], "prepared proxy-aware choice violates selection rule"
    return selected
