
from copy import deepcopy
import json
from pathlib import Path

from sregnas.icde27.recorder import dump, read_rows, sha, write_table
from .post import dispatch
from .records import source_records
from .tables import final_accuracy, primary


def objects(root):
    pass
    state = json.loads((root/'results/icde27/quality_queue_state.json').read_text())
    selected = []
    for dataset in ('Flickr', 'arXiv', 'Reddit', 'Products'):
        for method in ('SREGNAS', 'FastGNAS', 'GraphNAP', 'AP'):
            path, info = primary(root, dataset, method, state)
            if path is None:
                raise ValueError(f'ambiguous/missing primary: {dataset} {method}')
            if method == 'SREGNAS' and not info.get('s0_parallel'):
                raise ValueError('Exp1R requires the S0 primary')
            selected.append((f'Exp1_{dataset}_{method}', path))
    metis = [p.parent for p in (root/'results/icde27/EXP8').glob('*/run.json')
             if (d := json.loads(p.read_text()))['dataset'] == 'Products'
             and d['method'] == 'SREGNAS' and d['status'] == 'OK'
             and d['config']['graph']['partition_strategy'] == 'metis']
    if len(metis) != 1:
        raise ValueError('ambiguous/missing Exp8 Products METIS winner')
    selected.append(('Exp8_METIS_Products_SREGNAS', metis[0]))
    result = []
    for object_id, path in selected:
        info, _, specs = source_records(path)
        val, test, fp = final_accuracy(path, info)
        matching = [s for s in specs if s['fingerprint'] == fp]
        if not matching or any(s != matching[0] for s in matching):
            raise ValueError(f'ambiguous winner training spec: {path}')
        spec = matching[0]
        assert spec['replica'] == 0 and 'retrain_seed' not in spec
        assert spec['retrain_seed_policy'] == 'fingerprint_retrain_replica'
        assert val is not None and test is not None
        assert Path(spec['graph_path']).is_file() and Path(spec['checkpoint_path']).is_file()
        if spec.get('partition_receipt'):
            receipt = spec['partition_receipt']
            assert sha(receipt['path']) == receipt['sha256']
        cells = [r for r in read_rows(path/'events.jsonl') if r['event'] == 'cell_result'
                 and r['fingerprint'] == fp and r['mode'] in ('retrain','post_selection_test')]
        observed_wall = sum(r['result'].get('wall_s', 0) for r in cells)
        result.append(dict(object_id=object_id, dataset=info['dataset'], method=info['method'],
            fingerprint=fp, source=str(path), source_run_id=info['run_id'],
            source_record_type=info['record_type'], source_Q_attempt=info['Q_attempt'],
            source_Q_trunc=info.get('Q_trunc'), source_run_sha256=sha(path/'run.json'),
            backend=info['backend'], template=spec, val_pct=100*val, test_pct=100*test,
            estimated_wall_s=max(300, observed_wall+60)))
    return result


def replica_spec(template, replica, gpu, out):
    if replica not in (1, 2):
        raise ValueError('only the two authorized new replicas may train')
    if template.get('replica') != 0 or 'retrain_seed' in template:
        raise ValueError('expected the frozen derived-seed replica-zero template')
    spec = deepcopy(template)
    spec.update(replica=replica, gpu=gpu, phase=f'exp1r:replica{replica}:retrain',
                checkpoint_path=str(out/'checkpoints'/(spec['fingerprint']+'.pt')))
    return spec


def derived_seed(method, spec):
    if method == 'FastGNAS':
        from sregnas.comparison.retrain import native_seed
        return native_seed(spec['architecture'], spec['replica'])
    if spec['dataset'] in ('reddit', 'ogbn-products'):
        from sregnas.training.minibatch import retrain_root_seed
        return retrain_root_seed(spec['fingerprint'], spec['replica'])
    from sregnas.training.fullbatch import fingerprint_retrain_replica_seed
    return fingerprint_retrain_replica_seed(spec['fingerprint'], spec['replica'])


def test_accounting(out, fingerprint):
    pass
    rows=[r for r in read_rows(out/'events.jsonl')
          if r.get('mode')=='post_selection_test' and r.get('fingerprint')==fingerprint]
    starts=[r for r in rows if r['event']=='cell_start']
    ends=[r for r in rows if r['event']=='cell_end']
    results=[r for r in rows if r['event']=='cell_result']
    successes=[r for r in results if r['result']['status']=='OK']
    one=len(starts)==len(ends)==len(results)==len(successes)==1
    unchanged=one and starts[0]['checkpoint_before_sha256']==ends[0]['checkpoint_after_sha256']==sha(starts[0]['checkpoint_path'])
    return dict(test_evaluations=len(successes),one_test=one,test_checkpoint_unchanged=unchanged)


def execute(job, out, info):
    source = Path(job['source'])
    assert sha(source/'run.json') == job['source_run_sha256']
    spec = replica_spec(job['template'], job['replica'], job['gpus'][0], out)
    seed = derived_seed(job['method'], spec)
    dump(out/'source_retrain_spec.json', job['template'])
    dump(out/'retrain_spec.json', spec)
    result = dict(object_id=job['object_id'], source_run_id=job['source_run_id'],
        source_record_type=job['source_record_type'], source_Q_attempt=job['source_Q_attempt'],
        source_Q_trunc=job.get('source_Q_trunc'), winner=spec['fingerprint'], replica=job['replica'],
        derived_seed=seed, seed_rule=dict(retrain=(
            'comparison.retrain.native_seed(architecture, replica)' if job['method']=='FastGNAS'
            else 'H(fingerprint, retrain, replica); existing production seed function')),
        search_executed=False, test_evaluations=0, assertions={})
    training = dispatch(job['method'], [spec], out, 'retrain')[0]
    dump(out/'partial_retrains.json', [training])
    result.update(retrain_result=training, status=training['status'])
    write_table(out/'retrain.tsv', [dict(run_id=out.name,fingerprint=spec['fingerprint'],
        replica=job['replica'],status=training['status'],val_acc=training.get('value'),
        checkpoint_path=spec['checkpoint_path'],derived_seed=seed)],
        ['run_id','fingerprint','replica','status','val_acc','checkpoint_path','derived_seed'])
    result['assertions'].update(retrain_success=training['status']=='OK',
        original_record_unchanged=sha(source/'run.json')==job['source_run_sha256'])
    if training['status'] != 'OK':
        return result
    result['assertions']['training_seed_matches'] = training['metrics']['seed'] == seed
    assert result['assertions']['training_seed_matches']
    test_spec = dict(spec,mode='post_selection_test',phase=f"exp1r:replica{job['replica']}:test")
    dump(out/'test_spec.json', test_spec)
    tested = dispatch(job['method'], [test_spec], out, 'test')[0]
    accounting=test_accounting(out,spec['fingerprint'])
    result.update(status=tested['status'],val_acc=training['metrics']['val_acc'],test_result=tested,
        test_acc=tested.get('metrics',{}).get('test_acc'),
        test_evaluations=accounting['test_evaluations'],
        checkpoint_path=spec['checkpoint_path'],checkpoint_sha256=sha(spec['checkpoint_path']))
    result['assertions'].update(test_success=tested['status']=='OK',
        test_seed_matches=tested.get('metrics',{}).get('seed')==seed,
        one_test=accounting['one_test'],test_checkpoint_unchanged=accounting['test_checkpoint_unchanged'],
        optional_backend_counter_matches=('test_evaluations' not in tested.get('metrics',{})
            or tested['metrics']['test_evaluations']==accounting['test_evaluations']),
        checkpoint_exists=Path(spec['checkpoint_path']).is_file())
    return result
