
import csv
from pathlib import Path
from .recorder import read_rows


def completed_accounting(path, info):
    path = Path(path)
    with (path / 'launch_ledger.tsv').open() as stream:
        ledger = list(csv.DictReader(stream, delimiter='\t'))
    candidates = [r for r in ledger if r['stage'] == 'search' and r['kind'] == 'candidate']
    good = [r for r in candidates if r['status'] == 'OK']
    events = read_rows(path / 'events.jsonl')
    trained = [e for e in events if e.get('event') == 'cell_result'
               and e.get('mode') == 'retrain' and e['result']['status'] == 'OK']
    row = dict(Q_candidate=len(candidates), Q_success=len(good),
               U_unique=len({r['fingerprint'] for r in candidates}),
               U_unique_success=len({r['fingerprint'] for r in good}),
               K_successful_retrains=len({e['fingerprint'] for e in trained}))
    phases = {p['phase']: p for p in info.get('phases', [])}
    resources = read_rows(path / 'resources_during.jsonl')
    for name, mode, phase in [('search', 'proxy', 'search'), ('retrain', 'retrain', 'retrain+validation')]:
        bound = phases.get(phase)
        values = [p['memory_mib'] / 1024 for s in resources
                  if bound and bound['start_perf_s'] <= s['perf_s'] <= bound['end_perf_s']
                  for p in s['processes'] if p['owned'] and p.get('memory_mib') is not None]
        row[name + '_peak_process_GiB'] = max(values) if values else ''
        for metric in ('allocated', 'reserved'):
            values = [e['result']['peak_' + metric + '_gb'] for e in events
                      if e.get('event') == 'cell_result' and e.get('mode') == mode
                      and e['result'].get('peak_' + metric + '_gb') is not None]
            row[name + '_peak_' + metric + '_GiB'] = max(values) if values else ''
    return row
