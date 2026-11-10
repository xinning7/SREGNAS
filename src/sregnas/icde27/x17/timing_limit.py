
import json
import os
from pathlib import Path
import time

FLAG='SREGNAS_X17_E2E_CAP'


def progress(out):
    start=None;complete=None;ended=None
    path=Path(out)/'events.jsonl'
    if path.exists():
        with path.open() as stream:
            for line in stream:
                try:r=json.loads(line)
                except json.JSONDecodeError:continue
                if r.get('event')=='e2e_start' and start is None:start=r['perf_s']
                if r.get('event')=='direct_e2e_wall':complete=r['wall_s'];ended=r['perf_s']
    return start,complete,ended


def reached(out,started,limit,*,now_perf=None):
    now=time.perf_counter() if now_perf is None else now_perf
    if os.environ.get(FLAG,'0')!='1':return now-started>=limit
    start,complete,ended=progress(out)
    if complete is not None:return complete>limit or now-ended>=120
    if start is None:return now-started>=120
    return now-start>=limit
