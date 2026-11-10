
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time
import uuid

ENV = "SREGNAS_ICDE27_RECORD_DIR"
_DIRECT_START = None


def enabled():
    return bool(os.environ.get(ENV))


def append(filename, row):
    if not enabled():
        return None
    path=Path(os.environ[ENV])/filename
    row=dict(row, perf_s=time.perf_counter(), epoch_s=time.time(), pid=os.getpid())
    if os.environ.get('SREGNAS_X22_DEFER_RECORDS') == '1':
        from sregnas.icde27.x22.deferred import append as deferred_append
        deferred_append(path, row)
        return row
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n')
        f.flush()
    return row


def event(name, **fields):
    return append('events.jsonl',dict(event=name,**fields))


def start_clock(source):
    global _DIRECT_START
    if enabled():
        _DIRECT_START=time.perf_counter()
        event('e2e_start',source=source)


def stop_clock():
    if enabled() and _DIRECT_START is not None:

        event('direct_e2e_wall',wall_s=time.perf_counter()-_DIRECT_START)


def rows(filename):
    if os.environ.get('SREGNAS_X22_DEFER_RECORDS') == '1':
        from sregnas.icde27.x22.deferred import flush
        flush()
    path=Path(os.environ[ENV])/filename
    if not path.exists():return []
    with path.open() as f:
        fcntl.flock(f,fcntl.LOCK_SH)
        return [json.loads(line) for line in f if line.strip()]


def launch(kind, *, identity, stage='search', **fields):
    if not enabled():
        return None
    if kind not in {'candidate','retry','resume','warmup','probe'}:
        raise ValueError(kind)
    path=Path(os.environ[ENV])/'launch_ledger.jsonl'
    with path.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        if os.environ.get('SREGNAS_X22_DEFER_RECORDS') == '1':
            from sregnas.icde27.x22.ledger_index import previous_identities
            repeated = (identity,stage) in previous_identities(path,f)
        else:
            f.seek(0)
            previous=[json.loads(line) for line in f if line.strip()]
            repeated = any(r['identity']==identity and r['stage']==stage for r in previous)
        if kind=='candidate' and repeated:
            kind='resume' if os.environ.get('SREGNAS_ICDE27_RESUME')=='1' else 'retry'
        row=dict(kind=kind,stage=stage,identity=identity,launch_id=uuid.uuid4().hex,
            perf_s=time.perf_counter(),epoch_s=time.time(),pid=os.getpid(),**fields)
        f.seek(0,2); f.write(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n'); f.flush()
    return row


@contextmanager
def cell(spec, *, backend='canonical'):
    if not enabled():
        yield
        return
    mode=spec['mode']
    identity=':'.join(str(spec.get(k,'')) for k in ('phase','round','shard','fingerprint'))
    common=dict(identity=identity,mode=mode,backend=backend,fingerprint=spec['fingerprint'],
        architecture=spec['architecture'],gpu=spec.get('gpu'),phase=spec.get('phase'),
        checkpoint_path=spec.get('checkpoint_path'),epochs=spec.get('epochs'))
    attempt=None
    checkpoint_before=None
    if mode=='post_selection_test' and spec.get('checkpoint_path'):
        from .recorder import sha
        checkpoint_before=sha(spec['checkpoint_path'])
    if os.environ.get('SREGNAS_ICDE27_PROFILE')=='1' and mode=='retrain':
        attempt=launch('probe',stage='profile',**common)
    elif mode in {'proxy','full'}:
        attempt=launch('candidate',**common)
    elif mode=='probe':
        attempt=launch('probe',stage='nomination',**common)
    event('cell_start',**common,launch_id=attempt['launch_id'] if attempt else None,
          checkpoint_before_sha256=checkpoint_before)
    try:
        yield
    except BaseException as exc:
        event('cell_error',**common,error=repr(exc),launch_id=attempt['launch_id'] if attempt else None)
        raise
    finally:
        event('cell_end',**common,launch_id=attempt['launch_id'] if attempt else None,
              checkpoint_after_sha256=sha(spec['checkpoint_path']) if checkpoint_before else None)
        if mode=='post_selection_test':
            event('e2e_end',source='existing checkpoint test returned')


def call(spec, function, *args, backend='canonical', **kwargs):
    from sregnas.training.fixed_proxy import apply_options
    spec = apply_options(spec)
    with cell(spec,backend=backend):
        result=function(spec,*args,**kwargs)
        if enabled():
            from dataclasses import asdict, is_dataclass
            row=asdict(result) if is_dataclass(result) else result
            event('cell_result',mode=spec['mode'],fingerprint=spec['fingerprint'],
                  backend=backend,result=row)
        return result
