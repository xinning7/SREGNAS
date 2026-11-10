
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from .recorder import dump


def dispatch(method, specs, out, label):
    groups = {}
    for spec in specs:
        groups.setdefault(spec['gpu'], []).append(spec)
    def worker(gpu, batch):
        job = out/f'{label}_gpu{gpu}.job.json'
        output = out/f'{label}_gpu{gpu}.json'
        dump(job, dict(method=method, specs=batch, output=str(output)))
        with (out/f'{label}_gpu{gpu}.log').open('w') as log:
            result = subprocess.run([sys.executable, '-m', 'sregnas.icde27.x9_retrain', '--job', str(job)],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'cell worker failed: {job}; exit={result.returncode}')
        return json.loads(output.read_text())
    with ThreadPoolExecutor(max_workers=len(groups) or 1) as pool:
        futures = [pool.submit(worker, gpu, batch) for gpu, batch in groups.items()]
        results = [r for future in futures for r in future.result()]
    by_fp = {r.get('fingerprint', r.get('id')): r for r in results}
    return [by_fp[s['fingerprint']] for s in specs]


def main():
    from .quality import post, entry
    post.dispatch = dispatch
    rc = entry.main()
    out = Path(sys.argv[sys.argv.index('--out') + 1])
    job = json.loads((out/'job.json').read_text())
    path = out/'measurement.json'
    if path.exists():
        result = json.loads(path.read_text())
        counters = [json.loads(p.read_text()) for p in out.glob('*.nf_calls.json')]
        calls = {name: sum(c[name] for c in counters) for name in ('enabled', 'attach_degrees', 'forward')}
        result.update(nf_runtime_calls=calls, nf_instrumented_processes=len(counters))
        if job.get('runtime_assert'):
            result['assertions'].update(nf_observer_executed=bool(counters),
                nf_no_fullbatch_calls=bool(counters) and all(v == 0 for v in calls.values()))
            if not all(result['assertions'].values()):
                rc = 2
        dump(path, result)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
