

import json
from pathlib import Path
import time
from .provenance import commit_hash


def record(out, **fields):
    path = Path(out) / "campaign_timing.json"
    row = (
        json.loads(path.read_text())
        if path.exists()
        else dict(
            commit_hash=commit_hash(),
            timing_basis="search_start_to_validation_winner_selected",
            test_included=False,
            cpu_preparation_included=False,
        )
    )
    row.update(fields)
    if "winner_selected_perf_s" in row and "search_started_perf_s" in row:
        row["search_to_winner_s"] = (
            row["winner_selected_perf_s"] - row["search_started_perf_s"]
        )
    if "search_finished_perf_s" in row and "search_started_perf_s" in row:
        row["search_wall_s"] = (
            row["search_finished_perf_s"] - row["search_started_perf_s"]
        )
    path.write_text(json.dumps(row, indent=2))
    return row


def mark(out, event, **fields):
    return record(
        out,
        **{event + "_perf_s": time.perf_counter(), event + "_epoch_s": time.time()},
        **fields,
    )
