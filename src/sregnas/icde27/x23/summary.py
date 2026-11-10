
import csv
from pathlib import Path
from .common import OUT, read, write, now

PHASES = ['prep', 'search', 'nomination+probe+refill', 'retrain+validation', 'winner-test', 'other']
def summarize():
    rows = []
    for path in sorted((OUT / 'runs').glob('*/run.json')):
        try: info = read(path)
        except (ValueError, OSError): continue
        result_path = path.parent / 'pipeline/result.json'
        result = read(result_path) if result_path.exists() else {}
        if not result and info.get('method') == 'SREGNAS':


            summaries = sorted((path.parent / 'pipeline/profiles').glob('*/summary.json'))
            if len(summaries) == 1:
                profile = read(summaries[0])
                result = dict(val_acc=profile.get('system_val'), test_acc=profile.get('system_test'))
        val = result.get('val_acc', info.get('val_acc')); test = result.get('test_acc', info.get('test_acc'))
        if val is None: val = info.get('winner', {}).get('val_acc')
        if test is None:

            test = info.get('winner_test_acc')
        row = dict(run_id=info.get('run_id', path.parent.name), source=str(path),
            status=info.get('status'), timing_valid=info.get('timing_valid'), method=info.get('method'),
            dataset=info.get('dataset'), record_type=info.get('record_type'), commit=info.get('commit'),
            Q_attempt=info.get('Q_attempt'), E2E_s=info.get('direct_e2e_wall_s', info.get('e2e_s')),
            pool_s=info.get('pool_s'), val_pct=100*val if isinstance(val,(float,int)) else None,
            test_pct=100*test if isinstance(test,(float,int)) else None)
        phases = {p['phase']: p.get('wall_s') for p in info.get('phases', [])}
        row.update({p+'_s': phases.get(p) for p in PHASES})
        if not phases and info.get('prep_s') is not None: row['prep_s'] = info['prep_s']
        rows.append(row)
    fields = list(dict.fromkeys(k for r in rows for k in r)) or ['run_id','status']
    dest = OUT / 'all_runs.csv'; tmp = dest.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    tmp.replace(dest);write(OUT/'summary_status.json',dict(at=now(),runs=len(rows)))
    return rows

if __name__ == '__main__': summarize()
