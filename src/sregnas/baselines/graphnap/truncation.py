
import hashlib
import json
from pathlib import Path
import time


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SearchBudget:
    def __init__(self, max_candidates=None, max_seconds=None):
        if max_candidates is not None and max_candidates <= 0:
            raise ValueError('max search candidates must be positive')
        if max_seconds is not None and max_seconds <= 0:
            raise ValueError('max search seconds must be positive')
        self.max_candidates, self.max_seconds = max_candidates, max_seconds
        self.reason = None

    def allowance(self, sink, requested):
        completed = sum(row['status'] != 'incomplete' for row in sink.rows)
        if self.max_candidates is not None and completed >= self.max_candidates:
            self.reason = 'max_search_candidates'
            return 0
        start = getattr(sink, 'search_start_perf_s', None)
        if self.max_seconds is not None and start is not None and time.perf_counter()-start >= self.max_seconds:
            self.reason = 'max_search_seconds'
            return 0
        return min(requested, self.max_candidates-completed) if self.max_candidates is not None else requested

    def finish(self, sink, nominal_q):
        self.allowance(sink, 1)
        info = dict(nominal_Q=nominal_q, max_search_candidates=self.max_candidates,
                    max_search_seconds=self.max_seconds, actual_attempts=len(sink.rows),
                    completed=sum(row['status'] != 'incomplete' for row in sink.rows),
                    successful=sum(row['status'] == 'OK' for row in sink.rows), stop_reason=self.reason,
                    search_start_perf_s=getattr(sink, 'search_start_perf_s', None),
                    search_return_perf_s=time.perf_counter(),
                    semantics='stop dispatch at the bound; already dispatched candidates finish')
        (sink.out/'search_budget.json').write_text(json.dumps(info, indent=2)+'\n')
        return info


def select_search_record(source, limit, cfg, method):
    pass
    source = Path(source).resolve()
    if limit <= 0:
        raise ValueError('positive source search limit required')
    info = json.loads((source/'run.json').read_text())
    if info['method'] != method or info['backend'] != 'canonical' or info['config'] != cfg.to_dict():
        raise ValueError('source method/backend/frozen config mismatch')
    if not info['timing_valid']:
        raise ValueError('source search record has invalid timing')
    launches = [r for r in read_rows(source/'launch_ledger.jsonl') if r['kind'] == 'candidate' and r['stage'] == 'search']
    events = read_rows(source/'events.jsonl')
    completed = []
    for launch in launches:
        ends = [e for e in events if e['event'] == 'cell_end' and e.get('launch_id') == launch['launch_id']]
        if len(ends) != 1:
            continue
        results = [e for e in events if e['event'] == 'cell_result' and e.get('mode') == launch['mode']
                   and e.get('fingerprint') == launch['fingerprint'] and e['pid'] == launch['pid']
                   and launch['perf_s'] <= e['perf_s'] <= ends[0]['perf_s']]
        if len(results) == 1 and results[0]['result']['status'] == 'OK':
            row = dict(results[0]['result'])
            if row.get('optimizer_steps') != 50:
                raise ValueError('source proxy budget mismatch')
            completed.append(dict(candidate=row, launch_id=launch['launch_id'], start_perf_s=launch['perf_s'],
                                  end_perf_s=ends[0]['perf_s']))
    completed.sort(key=lambda x: (x['end_perf_s'], x['launch_id']))
    if len(completed) < limit:
        raise ValueError('source has fewer completed candidates than requested')
    selected = completed[:limit]
    search_start = min(r['perf_s'] for r in launches)
    clock_start = min(e['perf_s'] for e in events if e['event'] == 'e2e_start')
    return dict(source_run_id=info['run_id'], source_dir=str(source), source_commit=info['commit'],
                source_timing_valid=info['timing_valid'], selected=selected, search_limit=limit,
                source_prep_s=search_start-clock_start, source_search_start_perf_s=search_start,
                source_graph_hash=info['graph_hash'], source_split_hash=info['split_hash'], source_space_hash=info['space_hash'],
                source_artifact_sha256={name:digest(source/name) for name in ['run.json','launch_ledger.jsonl','events.jsonl','pipeline/port_candidates.json']})


def run_from_search_record(cfg, method, source, limit, out, shortlist_size=None):
    from sregnas.icde27 import ledger
    from sregnas.comparison.search import ranked_candidates
    from sregnas.comparison.launch import common_retrain
    ledger.start_clock('P1-G nomination from an existing search record; no new search')
    ledger.event('nomination_started', source=str(source))
    selected = select_search_record(source, limit, cfg, method)
    (out/'source_selection.json').write_text(json.dumps(selected, indent=2)+'\n')
    candidates = [r['candidate'] for r in selected['selected']]
    (out/'port_candidates.json').write_text(json.dumps(candidates, indent=2)+'\n')
    setup = json.loads((Path(source)/'pipeline/port_setup.json').read_text())
    setup.update(reused_search_record=str(source), new_search_launches=0)
    (out/'port_setup.json').write_text(json.dumps(setup, indent=2)+'\n')
    graph_file = Path(source)/'pipeline/retrain_graph.json'
    if graph_file.exists():
        graph = json.loads(graph_file.read_text())['graph_path']
    else:
        from sregnas.paths import DATA_DIR
        relative = {'reddit':'Reddit/sregnas_cache_v1/full/manifest.json',
                    'ogbn-products':'ogbn_products/sregnas_comparison_v1/full/manifest.json'}
        graph = str(DATA_DIR/relative[cfg.graph.dataset])
    if not Path(graph).is_file():
        raise FileNotFoundError(graph)
    (out/'retrain_graph.json').write_text(json.dumps(dict(graph_path=graph)))
    ranked = ranked_candidates([dict(row, architecture_json=json.dumps(row['architecture'])) for row in candidates], method)
    (out/'ranked_candidates.json').write_text(json.dumps(ranked, indent=2))
    return common_retrain(cfg, method, ranked, graph, out, shortlist_size=shortlist_size)


def host_rss_bytes(pid='self'):
    try:
        return int(next(line.split()[1] for line in Path(f'/proc/{pid}/status').read_text().splitlines()
                        if line.startswith('VmRSS:')))*1024
    except (OSError, StopIteration):
        return None


def completed_timing(source, limit=None):
    pass
    launches = [r for r in read_rows(source/'launch_ledger.jsonl') if r['kind'] == 'candidate']
    events = read_rows(source/'events.jsonl')
    ends = {e['launch_id']:e['perf_s'] for e in events if e['event'] == 'cell_end' and e.get('launch_id')}
    good = {r['fingerprint'] for r in json.loads((source/'pipeline/port_candidates.json').read_text()) if r['status'] == 'OK'}
    rows = [dict(launch_id=r['launch_id'], fingerprint=r['fingerprint'], start_perf_s=r['perf_s'],end_perf_s=ends[r['launch_id']])
            for r in launches if r['launch_id'] in ends and r['fingerprint'] in good]
    rows.sort(key=lambda r:(r['end_perf_s'],r['launch_id']))
    return rows[:limit] if limit else rows
