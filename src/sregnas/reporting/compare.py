

import json
import math
from pathlib import Path


def _profile(path):
    path = Path(path)
    if path.is_file():
        record = json.loads(path.read_text())
        if "summary" in record:
            return record
        path = path.parent
    if (path / "summary.json").exists():
        return dict(
            summary=json.loads((path / "summary.json").read_text()),
            config=json.loads((path / "resolved_config.json").read_text()),
            source=str(path),
            audit_status="UNREVIEWED",
        )
    profiles = list((path / "profiles").glob("*/summary.json"))
    if len(profiles) != 1:
        raise ValueError("choose a directory containing exactly one completed profile")
    record = _profile(profiles[0].parent)
    for filename, key in (
        ("request.json", "request"),
        ("provenance.json", "provenance"),
    ):
        if (path / filename).exists():
            record[key] = json.loads((path / filename).read_text())
    return record


def compare(reference, candidate, out):
    old, new = _profile(reference), _profile(candidate)
    for record in (old, new):
        record.setdefault("config", {}).setdefault("runtime", {}).setdefault(
            "execution_scope", "both"
        )
        record.setdefault("config", {}).setdefault("runtime", {}).setdefault(
            "side_timer",
            (
                "continuous"
                if record["summary"].get("direct_continuous_side_e2e")
                else "phase_sum"
            ),
        )
    differences = []
    ignored_runtime = {
        "graph_cache_reference",
        "projection_reference_run",
        "balance_reference_run",
        "balance_reference_profile",
        "legacy_barrier_reference_run",
        "legacy_barrier_reference_profile",
        "oversubscribed_reference_run",
        "oversubscribed_reference_profile",
    }
    for group in ("graph", "search", "train", "selection", "runtime"):
        left, right = old.get("config", {}).get(group, {}), new.get("config", {}).get(
            group, {}
        )
        for key in sorted(left.keys() | right.keys()):
            if group == "runtime" and key in ignored_runtime:
                continue
            if left.get(key) != right.get(key):
                differences.append(
                    dict(
                        field=f"{group}.{key}",
                        reference=left.get(key),
                        candidate=right.get(key),
                    )
                )
    for key in ("fixed_x", "full_budget"):
        if old.get("config", {}).get(key) != new.get("config", {}).get(key):
            differences.append(
                dict(
                    field=key,
                    reference=old.get("config", {}).get(key),
                    candidate=new.get("config", {}).get(key),
                )
            )
    deltas = {}
    a, b = old["summary"], new["summary"]
    for key in ("test_isolation", "winner_selection_metric", "kpi2_report_metric"):
        if a.get(key) != b.get(key):
            differences.append(
                dict(field=f"summary.{key}", reference=a.get(key), candidate=b.get(key))
            )
    for key in (
        "system_e2e_s",
        "baseline_e2e_s",
        "system_search_s",
        "baseline_search_s",
        "speedup",
        "system_val",
        "baseline_val",
        "system_test",
        "baseline_test",
    ):
        x, y = a.get(key), b.get(key)
        if (
            isinstance(x, (int, float))
            and isinstance(y, (int, float))
            and math.isfinite(x)
            and math.isfinite(y)
        ):
            delta = y - x
            deltas[key] = dict(reference=x, candidate=y, delta=delta)
            if key.endswith(("_val", "_test")):
                deltas[key]["delta_percentage_points"] = 100 * delta
            elif x:
                deltas[key]["relative_change_percent"] = 100 * delta / x

    blockers = list(old.get("comparison_caveats", []))
    if new["summary"].get("execution_scope") == "system":
        blockers.append(
            "Candidate is system-only: no measured baseline or same-run speedup is available."
        )
    blockers.append(
        "GPU occupancy, hardware/software identity and active proposal gate require the recorded run audits; summary metrics alone do not establish equivalence."
    )
    result = dict(
        reference=str(reference),
        candidate=str(candidate),
        protocol_differences=differences,
        metric_deltas=deltas,
        comparison_caveats=blockers,
        conclusion="PROTOCOL_MISMATCH" if differences else "AUDIT_REQUIRED",
        performance_unchanged=None,
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result
