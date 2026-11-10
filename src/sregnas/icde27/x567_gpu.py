
import json
from pathlib import Path

from .recorder import dump, sha


def execute(job, out, info):
    import torch
    from sregnas.core import Architecture
    from sregnas.models.canonical import load_model_module, assert_model
    from sregnas.training.fullbatch import PerLayerCanonicalModel
    from sregnas.training.minibatch import (ProductsRuntimeCache, retrain_root_seed,
        set_retrain_seed, exact_layerwise_full_neighbor_inference, sampled_inference_batch_size)
    from sregnas.training.executor import training_sample, training_inputs
    from sregnas.evaluation.checkpoints import load_retrain_checkpoint
    source = Path(job['source'])
    source_info = json.loads((source/'run.json').read_text())
    source_job = json.loads((source/'job.json').read_text())
    spec = json.loads((source/'retrain_spec.json').read_text())
    assert sha(source/'run.json') == job['source_run_sha256']
    assert source_info['status'] == 'OK' and all(source_info['assertions'].values())
    assert source_job['protocol'] == 'main' and spec['replica'] == 0
    assert spec['epochs'] == 25 and spec['dropout'] == 0.0
    assert spec['lr'] == .01 and spec['weight_decay'] == .0005
    assert all(spec[k] == 'gpu' for k in ('sampler_impl','feature_residency','inference_residency'))
    from .x3 import reference_tokens
    assert spec['architecture'] == reference_tokens('GCN256')
    checkpoint_hash = sha(spec['checkpoint_path'])
    assert checkpoint_hash == source_info['checkpoint_sha256']
    from .quality.cache import training_key
    original_key=training_key(spec,source_info)
    current_key=training_key(spec,info)
    assert original_key['key']==current_key['key'], 'complete training reuse key changed'
    device = torch.device('cuda:'+str(job['gpus'][0])); torch.cuda.set_device(device)
    seed = retrain_root_seed(spec['fingerprint'], 0); set_retrain_seed(seed)
    graph, sampler, _ = ProductsRuntimeCache().get(spec['graph_path'], device=device,
        sampler_impl='gpu', feature_residency='gpu', inference_residency='gpu')
    assert tuple(sampler.fanout) == (5,5) and sampler.batch_size == 4096
    module = load_model_module(force_native_scatter=True)
    canonical = module.build_model(Architecture(spec['architecture']),graph.num_features,
        graph.num_classes,dropout=0.0).to(device)
    model = PerLayerCanonicalModel(canonical).to(device); assert_model(model,module)
    identity = {k:spec[k] for k in ('fingerprint','architecture','dataset')}
    load_retrain_checkpoint(spec['checkpoint_path'],model,**identity)
    dump(out/'source_retrain_spec.json',spec)
    dump(out/'reuse_key.json',dict(source_run_id=source.name,source_run_sha256=job['source_run_sha256'],
        source_config=source_info['config'],source_graph_hash=source_info['graph_hash'],
        source_split_hash=source_info['split_hash'],spec=spec,backend='canonical',
        checkpoint_sha256=checkpoint_hash,derived_seed=seed,selection='final_pass',
        original_training_key=original_key,current_training_key=current_key,
        optimizer='Adam',passes=25,fanout=[5,5],batch_size=4096,weights_updated=0))
    exact = exact_layerwise_full_neighbor_inference(canonical,graph,sampler,device=device,
        splits=('val','test'),batch_size=sampled_inference_batch_size(spec['dataset']))
    rows = [dict(inference='exact',val_pct=100*exact.val_acc,test_pct=100*exact.test_acc)]
    sampled = {}; counts = {}; model.eval()
    with torch.no_grad():
        for split in ('val','test'):
            correct = count = 0


            for index, seeds in enumerate(sampler.seed_batches(split,seed=seed,shuffle=False)):
                batch = training_sample(sampler,seeds,root_seed=seed,pass_index=0,batch_index=index)
                x, edges, labels = training_inputs(graph,sampler,batch,device=device)
                prediction = model(x,edges)[:batch.batch_size].argmax(1)
                correct += int((prediction==labels).sum()); count += len(labels)
            assert count == int(getattr(graph,split+'_mask').sum())
            sampled[split+'_pct'] = 100*correct/count; counts[split] = dict(correct=correct,total=count)
    rows.append(dict(inference='sampled_5_5',**sampled))
    dump(out/'diagnostic_scores.json',rows)
    return dict(status='OK',backend='canonical',fingerprint=spec['fingerprint'],replica=0,
        checkpoint_path=spec['checkpoint_path'],checkpoint_sha256=checkpoint_hash,
        source_run_id=source.name,source_run_sha256=job['source_run_sha256'],derived_seed=seed,
        seed_rule=dict(retrain='H(fingerprint,retrain,0)',sampling='H(retrain_seed,neighbor_sample,0,split_local_batch_index)'),
        diagnostic_scores=rows,sampled_counts=counts,fanout=[5,5],batch_size=4096,
        Q_attempt=0,Q_search=0,training_launches=0,test_evaluations=2,
        assertions=dict(checkpoint_unchanged=sha(spec['checkpoint_path'])==checkpoint_hash,
            exact_val_reproduced=abs(exact.val_acc-source_info['val_acc'])<1e-6,
            exact_test_reproduced=abs(exact.test_acc-source_info['test_acc'])<1e-6,
            source_unchanged=sha(source/'run.json')==job['source_run_sha256'],
            matched_main_recipe=True,sampled_split_coverage=True))
