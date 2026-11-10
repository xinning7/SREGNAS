
import atexit
from copy import deepcopy
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import queue
import threading

_queue=queue.Queue()
_thread=None
_start_lock=threading.Lock()
_error=None
_audit_logs=[]

def audit_log(prefix,value):
    pass
    _audit_logs.append((prefix,deepcopy(value)))

def enabled():return os.environ.get('SREGNAS_X22_DEFER_RECORDS')=='1'

def _work():
    global _error
    while True:
        fn,args=_queue.get()
        try:fn(*args)
        except BaseException as exc:_error=exc
        finally:_queue.task_done()

def submit(fn,*args):
    global _thread
    if _error is not None:raise RuntimeError('X22 record writer failed') from _error
    with _start_lock:
        if _thread is None:
            _thread=threading.Thread(target=_work,name='x22-framework-writer',daemon=True);_thread.start()
    _queue.put((fn,args))

def flush():
    _queue.join()
    if _error is not None:
        root=os.environ.get('SREGNAS_ICDE27_RECORD_DIR')
        if root:
            dest=Path(root)/'x22_framework';dest.mkdir(parents=True,exist_ok=True)
            (dest/f'writer_error_{os.getpid()}.json').write_text(json.dumps(dict(error=repr(_error))))
        raise RuntimeError('X22 record writer failed') from _error


    while _audit_logs:
        prefix,value=_audit_logs.pop(0)
        print(prefix+json.dumps(value,sort_keys=True),flush=False)

def _json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True));tmp.replace(path)

def atomic_json(path,value):submit(_json,Path(path),deepcopy(value))

def worker_snapshot(result):
    pass
    if result.commit_hash is None:
        from sregnas.execution.provenance import commit_hash
        result.commit_hash=commit_hash()
    return asdict(result)

def _append(path,row):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n');f.flush()

def append(path,row):submit(_append,Path(path),deepcopy(row))

def assertions(out):
    return {'x22_record_writer_complete':not any((Path(out)/'x22_framework').glob('writer_error_*.json'))}

atexit.register(flush)
