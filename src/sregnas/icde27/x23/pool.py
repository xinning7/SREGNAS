
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def active():
    return any(os.environ.get(k) == '1' for k in ('SREGNAS_X23_POOL_READY', 'SREGNAS_X23_POOL_PROGRESS'))


def worker_ready(job):
    if not active(): return


    print('X23_POOL_READY', flush=True)
    if sys.stdin.readline().strip() != 'START':
        raise RuntimeError('X23 pool parent disappeared before START')


def worker_done(job, result, code):
    from sregnas.icde27.recorder import dump
    if active():
        print('X23_POOL_DONE', flush=True)
        if sys.stdin.readline().strip() != 'FINALIZE':
            raise RuntimeError('X23 pool parent disappeared before FINALIZE')
    if active() or os.environ.get('SREGNAS_X23_POOL_PROGRESS') == '1':
        dump(Path(job['output']), result)
    return code


def execute(out, cell, groups):
    from sregnas.icde27 import ledger
    from sregnas.icde27.recorder import dump
    from sregnas.icde27.p2p6.resources import mark_stage
    processes = []
    receipt = []
    def mark_start():
        mark_stage(out, 'pool/proxy' if cell['kind'] == 'replay' else 'pool/retrain_and_validation')
        ledger.event('pool_start')
        return time.perf_counter()
    started = None
    if os.environ.get('SREGNAS_X23_POOL_READY') != '1': started = mark_start()
    def pump(proc, log, ready, done, record):
        for line in proc.stdout:
            if line.strip() == 'X23_POOL_READY':
                record['ready_perf_s'] = time.perf_counter(); ready.set()
            elif line.strip() == 'X23_POOL_DONE':
                record['done_perf_s'] = time.perf_counter(); done.set()
            else:
                log.write(line)
        log.flush()
    try:
        for gpu, specs in groups.items():
            if not specs: continue
            job = out / f'worker_{gpu}.job.json'; output = out / f'worker_{gpu}.json'
            dump(job, dict(cell=cell, gpu=gpu, specs=specs, output=str(output)))
            log = (out / f'worker_{gpu}.log').open('w')
            proc = subprocess.Popen([sys.executable, '-m', 'sregnas.icde27.p2p6.canonical', '--job', str(job)],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            ready, done = threading.Event(), threading.Event()
            record = dict(gpu=gpu, pid=proc.pid)
            thread = threading.Thread(target=pump, args=(proc,log,ready,done,record), daemon=True); thread.start()
            processes.append((proc,log,ready,done,record,output,thread)); receipt.append(record)
        def wait_for(which):
            limit = time.monotonic() + 600
            for item in processes:
                proc, event = item[0], item[which]
                while not event.wait(.01):
                    if proc.poll() is not None: raise RuntimeError(f'worker {proc.pid} exited before barrier: {proc.returncode}')
                    if time.monotonic() > limit: raise TimeoutError('X23 fixed-pool barrier')
        wait_for(2)
        if started is None: started = mark_start()
        for proc,*_ in processes: proc.stdin.write('START\n'); proc.stdin.flush()
        wait_for(3)
        ended = time.perf_counter()
        ledger.event('pool_end'); ledger.stop_clock()

        for proc,*_ in processes: proc.stdin.write('FINALIZE\n'); proc.stdin.flush()
        output = []
        for proc,log,ready,done,record,path,thread in processes:
            code = proc.wait(timeout=120); thread.join(timeout=5)
            value = json.loads(path.read_text())
            if code: value['process_failed'] = True
            output.append(value)
        dump(out / 'x23_pool_boundary.json', dict(workers=receipt, pool_start_perf_s=started,
            pool_end_perf_s=ended, workers_ready_before_start=all(r['ready_perf_s'] <= started for r in receipt),
            first_payload_H2D='inside candidate, unchanged', result_serialization='after pool_end'))
        return output, started, ended
    finally:
        for proc,log,*_ in processes:
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            log.close()
