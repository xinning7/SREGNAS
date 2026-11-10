

import csv
import json
import os
import time
from pathlib import Path

from sregnas.search.controller import ShardController
from .common import candidate, token_space


class CentralController(ShardController):
    def __init__(self, config):


        from copy import deepcopy
        from sregnas.icde27.space import NAME
        config = deepcopy(config)
        config.token_subset = NAME
        super().__init__(config, config.base_seed + 10_000)


class RecordSink:
    def __init__(self, out, method, mode):
        self.out = Path(out)
        self.method = method
        self.mode = mode
        self.rows = []
        self.launches = {}

    def event(self, event, **fields):
        if os.environ.get('SREGNAS_X23_INCREMENTAL_RECORDS') == '1':
            from sregnas.icde27.x23.records import port_event
            port_event(self.out / "port_events.jsonl", dict(event=event, **fields))
            return
        with (self.out / "port_events.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(event=event, **fields)) + "\n")

    def start(self, spec):
        from sregnas.icde27 import ledger
        index = len(self.rows)
        common = dict(identity=f"{self.mode}:{index}:{spec['fingerprint']}",
                      fingerprint=spec["fingerprint"], architecture=spec["layers"],
                      backend="canonical", mode="port_proxy", method=self.method)
        attempt = ledger.launch("candidate", **common)
        self.launches[index] = attempt
        if not hasattr(self, "search_start_perf_s"):
            self.search_start_perf_s = attempt["perf_s"] if attempt else time.perf_counter()
        ledger.event("cell_start", **common, launch_id=attempt["launch_id"] if attempt else None)
        self.rows.append(dict(attempt=index, fingerprint=spec["fingerprint"],
                              architecture=spec["layers"], status="incomplete", value=None,
                              method=self.method, record_type=self.mode))
        self.event("candidate_launch", attempt=index, fingerprint=spec["fingerprint"])
        self.flush(index)
        return index

    def finish(self, index, row):
        from sregnas.icde27 import ledger
        self.rows[index].update(row)
        attempt = self.launches[index]
        ledger.event("cell_result", mode="port_proxy", fingerprint=row.get("fingerprint", self.rows[index]["fingerprint"]),
                     backend="canonical", result=self.rows[index])
        ledger.event("cell_end", mode="port_proxy", fingerprint=self.rows[index]["fingerprint"],
                     launch_id=attempt["launch_id"] if attempt else None)
        self.event("candidate_finish", attempt=index, status=row["status"])
        self.flush(index)

    def finalize_records(self):
        from sregnas.icde27.x23.records import finish
        finish(self)

    def flush(self, changed_index=None):
        if os.environ.get('SREGNAS_X23_INCREMENTAL_RECORDS') == '1' and not getattr(self, '_x23_sync_flush', False):
            from sregnas.icde27.x23.records import changed, finish
            if changed_index is not None:
                changed(self, changed_index)
            else:
                finish(self)
            return
        if os.environ.get('SREGNAS_X22_DEFER_RECORDS') == '1' and not getattr(self,'_x22_sync_flush',False) and not getattr(self,'_x23_sync_flush',False):
            from copy import copy, deepcopy
            from sregnas.icde27.x22.deferred import submit
            snapshot=copy(self);snapshot.rows=deepcopy(self.rows);snapshot._x22_sync_flush=True
            submit(type(self).flush,snapshot)
            return
        path = self.out / "port_candidates.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.rows, indent=2) + "\n")
        temporary.replace(path)
        fields = ["attempt", "method", "record_type", "fingerprint", "status", "value",
                  "wall_s", "optimizer_steps", "optimizer_steps_total", "seed_inputs",
                  "peak_allocated_bytes", "peak_reserved_bytes", "search_features_on_gpu",
                  "parameter_average_count", "parameter_average_bytes", "failure_stage", "error"]
        with (self.out / "candidates.tsv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            for row in self.rows:
                record = dict(row)
                if self.method != "GraphNAP":
                    record.update(parameter_average_count="N/A", parameter_average_bytes="N/A")
                writer.writerow(record)


def load_pool(path, limit=None):
    path = Path(path)
    if path.suffix == ".tsv":
        with path.open() as stream:
            rows = [dict(architecture=json.loads(r["architecture_json"]), fingerprint=r["fingerprint"])
                    for r in csv.DictReader(stream, delimiter="\t")]
    else:
        payload = json.loads(path.read_text())
        rows = payload if isinstance(payload, list) else payload["candidates"]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("fixed pool must contain candidates")
    return [candidate(row["architecture"], row.get("fingerprint", row.get("id")))
            for row in rows]


def central_search(config, executor, sink, *, max_search_candidates=None, max_search_seconds=None,
                   paspl_full_nas=False):
    from sregnas.icde27.x_authorization import seed_allowed
    seed_ok = seed_allowed(config.base_seed) or (
        paspl_full_nas and sink.method == "PAS-PL" and config.base_seed in (43, 44))
    if config.m != 16 or config.q % config.m or not seed_ok:
        raise ValueError("D10 requires M=16, Q divisible by 16, authorized root seed")
    budget = None
    if max_search_candidates is not None or max_search_seconds is not None:
        from .truncation import SearchBudget
        budget = SearchBudget(max_search_candidates, max_search_seconds)
        executor.search_budget = budget
    controller = CentralController(config)
    rounds = config.q // config.m
    for round_index in range(rounds):
        if budget and not budget.allowance(sink, 1):
            break
        proposals, observations = [], []
        for slot in range(config.m):
            for _ in range(config.max_resamples_per_slot):
                proposal = controller.propose()
                classification = controller.classify(proposal, static_gate=False)
                if classification == "ACCEPT":
                    proposals.append(proposal)
                    break
                sink.event("static_reject", round=round_index, slot=slot,
                           fingerprint=proposal.architecture.fingerprint(),
                           reason=classification, consumes_q=False)
                observations.append((proposal, config.floor_reward))
            else:
                raise RuntimeError("controller exhausted resamples before launch; Q incomplete")
        specs = [candidate(p.architecture.arch_description, p.architecture.fingerprint())
                 for p in proposals]
        rows = executor.evaluate(specs, sink)
        if len(rows) != len(proposals) and not (budget and budget.reason):
            raise AssertionError("one terminal result required per attempted candidate")
        observations.extend((p, row["value"] if row["status"] == "OK" else config.floor_reward)
                            for p, row in zip(proposals, rows))
        update = controller.observe(observations, launches_per_shard_round=16,
                                    flush=round_index == rounds - 1 or bool(budget and budget.reason))
        sink.event("controller_observe", round=round_index, **update)
        controller.save(sink.out / "controllers" / f"round_{round_index}.pt")
    if budget:
        budget.finish(sink, config.q)
    if len(sink.rows) != config.q and not (budget and budget.reason):
        raise AssertionError("attempt count differs from Q without an explicit search bound")
    return sink.rows
