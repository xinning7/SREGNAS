
from contextlib import contextmanager, nullcontext
from functools import wraps
import json
import os
from pathlib import Path
import time

ENV = "SREGNAS_EXP3_FINE_PROFILE"


def requested(cell):
    return (cell.get("experiment_id") == "3" and cell.get("record_type") == "profile"
            and cell.get("dataset") in ("arXiv", "Products"))


def enabled():
    return os.environ.get(ENV) == "1"


def append(kind, **fields):
    root = Path(os.environ["SREGNAS_ICDE27_RECORD_DIR"])
    row = dict(kind=kind, pid=os.getpid(), **fields)
    with (root/f"exp3_events_{os.getpid()}.jsonl").open("a") as stream:
        stream.write(json.dumps(row)+"\n")


@contextmanager
def interval(kind, **fields):
    started = time.perf_counter()
    append(kind+"_start", start_perf_s=started, **fields)
    status = "incomplete"
    try:
        yield
        status = "OK"
    finally:
        ended = time.perf_counter()
        append(kind, start_perf_s=started, end_perf_s=ended, status=status, **fields)


def scope(kind, **fields):
    return interval(kind, **fields) if enabled() else nullcontext()


def search_phase():
    try: return (Path(os.environ["TIMING_RUN_DIR"])/"phase.txt").read_text().strip() == "search"
    except (KeyError, FileNotFoundError): return False


def observe_orchestrator(orchestrator):
    if not enabled() or getattr(orchestrator, "_exp3_observed", False): return
    orchestrator._exp3_observed = True
    orchestrator._exp3_round = 0
    train = orchestrator.run_multiple_rounds
    evaluate = orchestrator.global_evaluate

    @wraps(train)
    def training(*args, **kwargs):
        if not search_phase(): return train(*args, **kwargs)
        orchestrator._exp3_round += 1
        with interval("native_schedule", round_index=orchestrator._exp3_round):
            return train(*args, **kwargs)

    @wraps(evaluate)
    def validation(*args, **kwargs):
        if not search_phase(): return evaluate(*args, **kwargs)
        arch_id = kwargs.get("arch_id", args[0] if args else None)
        with interval("native_round_validation", round_index=orchestrator._exp3_round, fingerprint=arch_id):
            return evaluate(*args, **kwargs)

    orchestrator.run_multiple_rounds = training
    orchestrator.global_evaluate = validation


def native_segment(function, worker, arch_spec, state_dict, config):
    pass
    audit = config["_timing_audit"]
    if not enabled() or audit["phase"] != "search":
        return function(worker, arch_spec, state_dict, config)
    fields = dict(segment_id=config.get("_icde27_segment_id"), fingerprint=audit["arch_id"],
                  worker=worker.worker_id, rotation_step=audit.get("rotation_step"))
    original_eval = worker._evaluate
    own_eval = "_evaluate" in vars(worker)

    @wraps(original_eval)
    def local_validation(*args, **kwargs):
        with interval("native_local_validation", **fields):
            return original_eval(*args, **kwargs)

    worker._evaluate = local_validation
    try:
        with interval("native_segment_execution", **fields):
            result = function(worker, arch_spec, state_dict, config)
        if '_icde27_exp3_training_end_perf_s' in config:
            append("native_segment_training", **fields, status="OK",
                   start_perf_s=config['_icde27_exp3_training_start_perf_s'],
                   end_perf_s=config['_icde27_exp3_training_end_perf_s'])
        append("native_segment_counters", **fields, counters=result[4])
        return result
    finally:
        if own_eval: worker._evaluate = original_eval
        else: del worker._evaluate
