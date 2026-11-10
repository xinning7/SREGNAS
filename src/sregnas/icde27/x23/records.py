
from copy import deepcopy
import os
import json
from sregnas.icde27.x22.deferred import submit, _append, flush

ENV = 'SREGNAS_X23_INCREMENTAL_RECORDS'

def enabled():
    return os.environ.get(ENV) == '1'

def changed(sink, index):

    row = deepcopy(sink.rows[index])
    submit(_append, sink.out / 'port_candidate_updates.jsonl', dict(index=index, row=row))

def finish(sink):
    if not enabled() or getattr(sink, '_x23_finished', False):
        return
    flush()
    sink._x23_sync_flush = True
    try:
        sink.flush()
        sink._x23_finished = True
    finally:
        sink._x23_sync_flush = False


def _port_event(path, row):
    with path.open('a') as stream:
        stream.write(json.dumps(row) + '\n')

def port_event(path, row):
    submit(_port_event, path, deepcopy(row))
