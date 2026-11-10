
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from .recorder import dump, sha, write_table
from .space import canonical_layers, canonical_fingerprint, validate_pair


def reference_tokens(name):
    enc, agg = ('gcn', 'sum') if name == 'GCN256' else ('const', 'mean')
    if name not in ('GCN256', 'SAGEmean256'): raise ValueError(name)
    return [dict(encoding_op=enc, aggregation_op=agg, activation='relu', num_heads=1, hidden_dim=256) for _ in range(2)]


def objects(root):
    from .quality.replicas import objects as original_objects
    chosen = original_objects(root)
    unique = {}
    for obj in chosen:
        layers = canonical_layers(obj['template']['architecture'])
        validate_pair(layers, obj['template']['dataset'])
        fp = canonical_fingerprint(layers)
        key = obj['dataset'], fp
        if key in unique:
            unique[key]['aliases'].append(obj['object_id'])
            unique[key]['methods'].append(obj['method'])
            unique[key]['source_run_ids'].append(obj['source_run_id'])
            continue
        unique[key] = dict(dataset=obj['dataset'], object=obj['object_id'], fingerprint=fp,
            architecture=layers, aliases=[obj['object_id']], methods=[obj['method']],
            source_run_ids=[obj['source_run_id']], source_fingerprint=obj['fingerprint'],
            source_backend=obj['backend'], source=obj['source'], source_run_sha256=obj['source_run_sha256'],
            template=deepcopy(obj['template']), protocols=['standard'], backend='canonical')
    for dataset in ('Flickr', 'arXiv', 'Reddit', 'Products'):
        source = next(o for o in chosen if o['dataset']==dataset and o['method']=='SREGNAS')
        for name in ('GCN256', 'SAGEmean256'):
            layers = reference_tokens(name); validate_pair(layers, source['template']['dataset'])
            fp = canonical_fingerprint(layers)
            assert (dataset, fp) not in unique
            unique[dataset, fp] = dict(dataset=dataset, object='Manual_'+dataset+'_'+name,
                fingerprint=fp, architecture=layers, aliases=[], methods=[name],
                source_run_ids=[source['source_run_id']], source_fingerprint='', source_backend='canonical',
                source=source['source'], source_run_sha256=source['source_run_sha256'],
                template=deepcopy(source['template']), protocols=['standard', 'main'], backend='canonical')
    assert len(unique)==23
    return list(unique.values())


def validation_points(sampled):
    return [5, 10, 15, 20] if sampled else list(range(1, 501))


class BestCheckpoint:
    def __init__(self, path, model, identity):
        self.path, self.model, self.identity = path, model, identity
        self.best = float('-inf'); self.step = None; self.history = []

    def observe(self, step, value):
        import math
        from sregnas.evaluation.checkpoints import save_retrain_checkpoint
        if not math.isfinite(value): raise FloatingPointError('nonfinite validation accuracy')
        self.history.append(dict(step=step, val_acc=value))
        if value > self.best:
            self.best, self.step = value, step
            save_retrain_checkpoint(self.path, self.model, **self.identity)


def standard(spec, out):
    import torch
    import torch.nn.functional as F
    from sregnas.core import Architecture
    from sregnas.models.canonical import load_model_module, assert_model
    from sregnas.training.fullbatch import PerLayerCanonicalModel
    from sregnas.training.minibatch import (retrain_root_seed, set_retrain_seed, ProductsRuntimeCache,
        train_products_minibatch, exact_layerwise_full_neighbor_inference, sampled_inference_batch_size)
    from sregnas.evaluation.checkpoints import load_retrain_checkpoint
    device = torch.device('cuda:'+str(spec['gpu']))
    torch.cuda.set_device(device)
    sampled = spec['dataset'] in ('reddit', 'ogbn-products')
    seed = retrain_root_seed(spec['fingerprint'], spec['replica'])
    set_retrain_seed(seed)
    module = load_model_module(force_native_scatter=True)
    if sampled:
        graph, sampler, _ = ProductsRuntimeCache().get(spec['graph_path'], device=device,
            sampler_impl='gpu', feature_residency='gpu', inference_residency='gpu')

        sampler.fanout = (15, 10); sampler.batch_size = 1024
        features, classes = graph.num_features, graph.num_classes
    else:
        raw = torch.load(spec['graph_path'], map_location='cpu', weights_only=False)
        data = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in raw.items()}
        features, classes = data['num_features'], data['num_classes']
    canonical = module.build_model(Architecture(spec['architecture']), features, classes, dropout=0.5).to(device)
    model = PerLayerCanonicalModel(canonical).to(device) if sampled else canonical
    assert_model(model,module)
    identity = {k:spec[k] for k in ('fingerprint', 'architecture', 'dataset')}
    best = BestCheckpoint(spec['checkpoint_path'],model,identity)
    opt = torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=5e-4)

    def score(split):
        if sampled:
            result=exact_layerwise_full_neighbor_inference(canonical,graph,sampler,device=device,
                splits=(split,),batch_size=sampled_inference_batch_size(spec['dataset']))
            return getattr(result,split+'_acc')
        model.eval()
        with torch.no_grad():
            logits=model(data['x'],data['edge_index']); mask=data[split+'_mask']
            assert int(mask.sum())>0
            return float((logits[mask].argmax(1)==data['labels'][mask]).float().mean().item())

    def observe(step):
        best.observe(step,score('val'))
        dump(out/'validation_history.json',best.history)
        print(json.dumps(dict(step=step,val_acc=best.history[-1]['val_acc'],best_step=best.step)),flush=True)

    if sampled:
        def callback(row):
            step=row['pass_index']+1
            if step in validation_points(True):observe(step)
        trained=train_products_minibatch(model,graph,sampler,opt,device=device,root_seed=seed,
            passes=20,pass_callback=callback)
        accounting=asdict(trained)
        assert trained.passes==20 and trained.train_seed_nodes==20*int(graph.train_mask.sum())
    else:
        for epoch in range(1,501):
            model.train();opt.zero_grad(set_to_none=True)
            logits=model(data['x'],data['edge_index'])
            loss=F.cross_entropy(logits[data['train_mask']],data['labels'][data['train_mask']])
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite full-batch loss')
            loss.backward();opt.step();observe(epoch)
        accounting=dict(epochs=500,optimizer_steps=500)
    assert [r['step'] for r in best.history]==validation_points(sampled)
    before=sha(spec['checkpoint_path'])
    dump(out/'checkpoint_selection.json',dict(best_step=best.step,best_val_acc=best.best,
        checkpoint_sha256=before,selection='strict greater validation; earliest tie',test_evaluations=0))
    load_retrain_checkpoint(spec['checkpoint_path'],model,**identity)
    tested=score('test')
    assert sha(spec['checkpoint_path'])==before
    return dict(status='OK',val_acc=best.best,test_acc=tested,winner=spec['fingerprint'],
        derived_seed=seed,seed_rule=dict(retrain='H(canonical_fingerprint,retrain,replica)'),
        best_step=best.step,validation_evaluations=len(best.history),test_evaluations=1,
        train_accounting=accounting,checkpoint_path=spec['checkpoint_path'],checkpoint_sha256=before,
        assertions=dict(validation_schedule_exact=True,best_checkpoint_selected=True,
            one_test=True,test_checkpoint_unchanged=True,canonical_model=True))


def execute(job,out,info):
    assert sha(Path(job['source'])/'run.json')==job['source_run_sha256']
    spec=deepcopy(job['template'])
    spec.update(architecture=job['architecture'],fingerprint=job['fingerprint'],replica=job['replica'],
        gpu=job['gpus'][0],mode='retrain',phase='x3:'+job['protocol']+':retrain',
        checkpoint_path=str(out/'checkpoints'/f"{job['fingerprint']}.pt"),
        comparison_method=None,dropout=0.5 if job['protocol']=='standard' else 0.0)
    spec['epochs']=(20 if spec['dataset'] in ('reddit','ogbn-products') else 500) if job['protocol']=='standard' else (25 if spec['dataset'] in ('reddit','ogbn-products') else 200)
    dump(out/'retrain_spec.json',spec)
    if job['protocol']=='standard':
        result=standard(spec,out)
    else:
        from .quality.post import dispatch
        from .quality.replicas import test_accounting
        training=dispatch('SREGNAS',[spec],out,'retrain')[0]
        dump(out/'partial_retrains.json',[training])
        result=dict(status=training['status'],assertions=dict(retrain_success=training['status']=='OK'))
        if training['status']=='OK':
            tested=dispatch('SREGNAS',[dict(spec,mode='post_selection_test',phase='x3:main:test')],out,'test')[0]
            result.update(val_acc=training['value'],test_acc=tested.get('metrics',{}).get('test_acc'),
                status=tested['status'],winner=spec['fingerprint'],derived_seed=training['metrics']['seed'],
                checkpoint_path=spec['checkpoint_path'],checkpoint_sha256=sha(spec['checkpoint_path']))
            accounting=test_accounting(out,spec['fingerprint']);result['test_evaluations']=accounting['test_evaluations']
            result['assertions'].update({k:v for k,v in accounting.items() if k!='test_evaluations'})
            result['assertions']['test_success']=tested['status']=='OK'
    result.update(backend='canonical',object=job['object'],report_method=job['report_method'],
        protocol=job['protocol'],replica=job['replica'],source_run_ids=job['source_run_ids'])
    write_table(out/'retrain.tsv',[dict(run_id=out.name,fingerprint=spec['fingerprint'],replica=spec['replica'],
        status=result['status'],val_acc=result.get('val_acc',''),checkpoint_path=spec['checkpoint_path'])],
        ['run_id','fingerprint','replica','status','val_acc','checkpoint_path'])
    return result
