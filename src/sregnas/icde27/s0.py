
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import os
import threading
import time

ENV = "SREGNAS_S0_PARALLEL"


def enabled():
    return os.environ.get(ENV) == "1"


@contextmanager
def span(step, **fields):
    if not enabled():
        yield
        return
    from .ledger import event
    start = time.perf_counter()
    event("prep_step_start", step=step, **fields)
    try:
        yield
    finally:
        event("prep_step_end", step=step, start_perf_s=start,
              wall_s=time.perf_counter()-start, **fields)


def ordered_map(function, items, *, workers=2):
    items = list(items)
    if not enabled():
        return [function(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(workers, len(items)) or 1) as pool:
        return list(pool.map(function, items))


def parallel_probe_and_refill(nominees, ranked, probe, gpu_for):
    pass
    from sregnas.selection.nomination import probe_and_refill
    if not enabled():
        return probe_and_refill(nominees, ranked, probe)
    locks = {gpu_for(row): threading.Lock() for row in nominees}
    def run(row):
        with locks[gpu_for(row)]:
            return probe(row)
    with ThreadPoolExecutor(max_workers=len(locks) or 1) as pool:
        futures = {r["fingerprint"]: pool.submit(run, r) for r in nominees}

        cached = {fp: future.result() for fp, future in futures.items()}
    def ordered_probe(row):
        fp = row["fingerprint"]
        return cached[fp] if fp in cached else probe(row)
    return probe_and_refill(nominees, ranked, ordered_probe)


def observe_cache_upload(runtime):
    pass
    if not enabled(): return
    from pathlib import Path
    original=runtime.load_payload
    def load(graph_path,mode):
        key=(str(Path(graph_path).resolve()),'proxy')
        if mode!='proxy' or key in runtime._payloads:
            return original(graph_path,mode)
        with span('shard_cache_load_h2d',graph_path=str(graph_path),
                  physical_gpu=os.environ.get('SREGNAS_PHYSICAL_GPU')):
            return original(graph_path,mode)
    runtime.load_payload=load
