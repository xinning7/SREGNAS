
import json
from pathlib import Path
import sqlite3
from sregnas.icde27.recorder import read_rows


def source_records(source):
    source = Path(source)
    info = json.loads((source/'run.json').read_text())
    if info['status'] != 'OK' or not all(info['assertions'].values()):
        raise ValueError('source run is not a complete validated pipeline')
    specs = [json.loads(p.read_text()) for p in source.glob('pipeline/**/specs/*retrain*.json')]
    for p in source.glob('pipeline/*retrain*.job.json'):
        specs.extend(json.loads(p.read_text()).get('specs', []))
    specs = [s for s in specs if s.get('mode') == 'retrain']
    if not specs: raise ValueError('source retraining specifications missing')
    if info['method'] == 'SREGNAS':
        conn = sqlite3.connect('file:'+str(source/'pipeline/queue.sqlite')+'?mode=ro',uri=True)
        conn.row_factory = sqlite3.Row
        try:
            raw = conn.execute("SELECT shard,fingerprint,architecture_json,status,value,round FROM tasks WHERE phase LIKE '%:proxy' ORDER BY round,shard,task_key").fetchall()
        finally: conn.close()
        rows = [dict(shard=r['shard'],fingerprint=r['fingerprint'],architecture=json.loads(r['architecture_json']),
                     status=r['status'],value=r['value'],round=r['round']) for r in raw]
        if not rows: raise ValueError('empty SREGNAS search trace')
    else:
        ranked = json.loads((source/'pipeline/ranked_candidates.json').read_text())
        rows = [dict(r,shard=0,fingerprint=r['id'],status='OK') for r in ranked]
    return info, rows, specs


def plan1_statistics(source):
    pass
    info,rows,specs=source_records(source)
    source=Path(source)
    with sqlite3.connect('file:'+str(source/'pipeline/queue.sqlite')+'?mode=ro',uri=True) as conn:
        entries=conn.execute("SELECT payload_json FROM phase_events WHERE event='PROBE_COMPLETE' ORDER BY rowid").fetchall()
    audit=json.loads(entries[-1][0])['audit'] if entries else None
    return dict(K_actual=info.get('K_actual',len(specs)),covered_shards=len({s['shard'] for s in specs}),
        duplicate_count=len(rows)-len({r['fingerprint'] for r in rows}),probe_count=info.get('Q_probe'),
        probe_decision_count=len(audit) if audit is not None else None,probe_reused=0,
        refill_count=sum('refill_fingerprint' in r for r in audit) if audit is not None else None,
        newly_trained=0,source_run_id=info['run_id'],source_counts_reused=True)
