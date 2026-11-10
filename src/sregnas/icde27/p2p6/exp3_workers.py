
import time
from .port_workers import APWorker


class Exp3APWorker(APWorker):
    def run_candidate(self, spec):
        started = time.perf_counter()
        row = super().run_candidate(spec)
        row.update(exp3_start_perf_s=started, exp3_end_perf_s=time.perf_counter())
        return row
