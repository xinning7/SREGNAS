
import json
import time
from sregnas.execution.telemetry import gpu_idle_preflight


def wait_for_gpu_release(gpus, *, owner_pid, verified_owned, baseline, out,
                         poll_interval_s=.5, required_idle=2, max_idle_drift_wait_s=None):
    pass
    memory = {r['gpu']: r['memory_used_mib'] for r in baseline['snapshots']}
    started = time.perf_counter()
    consecutive = 0
    samples = 0
    foreign_seen = False
    while True:
        sample = gpu_idle_preflight(gpus, owner_pid=owner_pid, verified_owned=verified_owned)
        released = (set(memory) == {r['gpu'] for r in sample['snapshots']} and
                    all(r['memory_used_mib'] <= memory[r['gpu']] for r in sample['snapshots']))
        foreign_seen |= bool(sample['foreign_processes'])
        consecutive = consecutive + 1 if sample['idle'] and released else 0
        samples += 1
        sample.update(memory_released=released, baseline_memory_mib=memory,
                      consecutive_idle_samples=consecutive, release_wait_s=time.perf_counter()-started)
        with (out/'resources_release_wait.jsonl').open('a') as stream:
            stream.write(json.dumps(sample)+'\n')
        if consecutive >= required_idle:
            return dict(sample, idle=True, settling_decision='PASS_AFTER_MEMORY_RELEASE',
                        release_samples=samples, foreign_process_observed=foreign_seen)
        if (max_idle_drift_wait_s is not None and sample['idle'] and not released
                and time.perf_counter()-started>=max_idle_drift_wait_s):


            return dict(sample,idle=False,actual_compute_idle=True,
                        settling_decision='FAIL_IDLE_MEMORY_BASELINE_DRIFT',release_samples=samples,
                        foreign_process_observed=foreign_seen)
        time.sleep(poll_interval_s)
