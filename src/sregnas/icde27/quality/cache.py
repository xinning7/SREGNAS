
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
from sregnas.icde27.recorder import read_rows, sha, dump
from sregnas.paths import PROJECT_ROOT as ROOT


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


@lru_cache(None)
def training_sources(commit):


    names = ['training/fullbatch.py','training/minibatch.py','training/native.py',
             'training/executor.py','models/gnn.py','models/canonical.py','models/codegen.py',
             'data/products.py','data/sampling.py','evaluation/inference.py',
             'evaluation/checkpoints.py','comparison/retrain.py',
             'baselines/fastgnas/trainer/builder.py']
    return {name: hashlib.sha256(subprocess.check_output(
        ['git','show',f'{commit}:src/sregnas/{name}'],cwd=ROOT)).hexdigest() for name in names}


@lru_cache(256)
def artifact_sha(path, size, mtime_ns):
    return sha(path)


def training_key(spec, info, *, sources=None):
    required = ('architecture','fingerprint','dataset','epochs','lr','weight_decay','dropout',
                'sampler_impl','feature_residency','inference_residency','replica','retrain_seed_policy')
    missing = [k for k in required if k not in spec]
    if missing:
        raise ValueError(f'incomplete training key: {missing}')


    graph = str(Path(spec['graph_path']).resolve())
    stat = Path(graph).stat()
    key = dict(schema='quality_retrain_key_v1', backend=info['backend'],
               graph_hash=info['graph_hash'], split_hash=info['split_hash'],
               graph_manifest_or_tensor_sha256=artifact_sha(graph,stat.st_size,stat.st_mtime_ns),
               workload={k:spec[k] for k in required},
               explicit_seed=spec.get('retrain_seed'),
               checkpoint_rule='final_epoch_or_final_pass; validation_only; no early stopping',
               sampling='G3_25_complete_passes_5_5' if spec['dataset'] in ('reddit','ogbn-products') else 'full_batch',
               training_sources=sources if sources is not None else training_sources(info['commit']))
    if info.get('normalization_fix'):
        key['normalization_fix'] = dict(schema=info['normalization_fix_schema'],
                                        source_sha256=info['normalization_fix_sha256'])
    if info.get('deterministic_diagnostic'):
        key['deterministic_diagnostic'] = dict(torch_deterministic_algorithms=True,cublas_workspace=':4096:8')
    if info.get('sampled_grad_clip'):
        key['sampled_grad_clip'] = dict(max_norm=info['sampled_grad_clip'],
            schema=info['sampled_grad_clip_schema'],source_sha256=info['sampled_grad_clip_sha256'])
    return dict(key=key, key_hash=digest(key), graph_source_path=graph)


class TrainingCache:
    def __init__(self, directory):
        self.directory = Path(directory); self.directory.mkdir(parents=True,exist_ok=True)

    def get(self, key):
        path = self.directory/(key['key_hash']+'.json')
        if not path.exists(): return None
        row = json.loads(path.read_text())
        if row['key'] != key['key']: raise AssertionError('training cache key collision')
        checkpoint = Path(row['result']['metrics']['checkpoint_path'])
        if not checkpoint.is_file() or sha(checkpoint) != row['checkpoint_sha256']:
            return None
        return row

    def put(self, key, result, source_run_id, test=None):
        if result['status'] != 'OK': return
        cp = result['metrics']['checkpoint_path']
        if not Path(cp).is_file(): raise FileNotFoundError(cp)
        path = self.directory/(key['key_hash']+'.json')
        current = self.get(key)
        if current:

            if test and current['checkpoint_sha256'] == sha(cp):
                current['test'] = test; dump(path,current)
            return
        dump(path,dict(**key,result=result,source_run_id=source_run_id,
                       checkpoint_sha256=sha(cp),test=test))

    def import_run(self, source):
        source = Path(source); info = json.loads((source/'run.json').read_text())
        events = read_rows(source/'events.jsonl')
        specs = [json.loads(p.read_text()) for p in source.glob('pipeline/**/specs/*.json')]
        for p in source.glob('pipeline/*.job.json'):
            specs.extend(json.loads(p.read_text()).get('specs',[]))
        results = {r['fingerprint']:r['result'] for r in events
                   if r['event']=='cell_result' and r['mode']=='retrain' and r['result']['status']=='OK'}
        tests = {r['fingerprint']:r['result'] for r in events
                 if r['event']=='cell_result' and r['mode']=='post_selection_test' and r['result']['status']=='OK'}
        imported = []
        for spec in specs:
            if spec.get('mode') != 'retrain' or spec['fingerprint'] not in results: continue
            result = results[spec['fingerprint']]
            key = training_key(spec,info)
            self.put(key,result,info['run_id'],tests.get(spec['fingerprint']))
            imported.append(dict(fingerprint=spec['fingerprint'],key_hash=key['key_hash'],source_run_id=info['run_id']))
        return imported
