

import csv
import json
from pathlib import Path


def _number(value, *, percent=False):
    if value is None:
        return "N/A"
    return f"{100 * value:.3f}%" if percent else f"{value:.3f}"


def generate_report(run_dir):
    run_dir = Path(run_dir)
    summaries = [
        json.loads(p.read_text())
        for p in sorted((run_dir / "profiles").glob("*/summary.json"))
    ]
    fields = [
        "profile",
        "execution_scope",
        "n",
        "q",
        "topology_mode",
        "system_e2e_s",
        "baseline_e2e_s",
        "speedup",
        "system_search_s",
        "baseline_search_s",
        "system_test",
        "baseline_test",
    ]
    with (run_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    lines = [
        "# SREGNAS run",
        "",
        "Historical E2E fields exclude graph preparation. Cold-start preparation is recorded separately.",
        "",
        "| Profile | Shards | Physical GPUs | System s | Baseline s | Speedup | System test | Baseline test |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['profile']} | {s['n']} | {len(s['system_physical_gpus'])} | {s['system_e2e_s']:.3f} | "
            f"{_number(s.get('baseline_e2e_s'))} | {_number(s.get('speedup'))} | "
            f"{_number(s['system_test'], percent=True)} | {_number(s.get('baseline_test'), percent=True)} |"
        )
    if any(s.get("execution_scope") == "system" for s in summaries):
        lines.extend(
            [
                "",
                "System-only execution: baseline was not run; its metrics and same-run speedup are N/A.",
            ]
        )
    lines.extend(
        [
            "",
            "Raw task results, validation selection, OOM events and GPU monitor traces remain in this run directory.",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")
    return summaries
