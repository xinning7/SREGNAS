
from collections import defaultdict
import json
from pathlib import Path
from .common import *

def jsonlines(path):
    if not Path(path).exists():return []
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

def sources():
    main=[dict(r,scope='search') for r in rows(ROOT/'results/icde27/x25/table_iii.csv')]
    pool=[dict(r,scope='pool') for r in rows(ROOT/'results/icde27/x25/table_v_fixed_pool.csv')]
    return main+pool

def events(r):return jsonlines(Path(r['source']).parent/'events.jsonl')

def cells(r):
    ee=sorted(events(r),key=lambda e:e['perf_s']);starts={};current={};out=[]
    for e in ee:
        if e['event']=='cell_start':
            c=dict(e,result={});starts[e['launch_id']]=c
            current[e.get('pid'),e.get('mode'),e.get('fingerprint')]=c
        elif e['event']=='cell_result':
            c=current.get((e.get('pid'),e.get('mode'),e.get('fingerprint')))
            if c is not None:c['result']=e['result'];c['result_perf_s']=e['perf_s']
        elif e['event']=='cell_end' and e.get('launch_id') in starts:
            c=starts.pop(e['launch_id']);c['end_perf_s']=e['perf_s'];c['service_s']=e['perf_s']-c['perf_s']
            rr=c['result'];c['gpu']=c.get('gpu',rr.get('worker_id'))
            if c['gpu'] is None and rr.get('cuda_visible_devices') is not None:
                try:c['gpu']=int(str(rr['cuda_visible_devices']).split(',')[0])
                except ValueError:pass
            c['status']=rr.get('status','missing');out.append(c)
    return sorted(out,key=lambda c:c['perf_s'])

def proxy_cells(r):return [c for c in cells(r) if c.get('mode') in ('proxy','port_proxy')]

def ident(r):return {k:r.get(k) for k in ('run_id','dataset','method','seed','scope','source')}
