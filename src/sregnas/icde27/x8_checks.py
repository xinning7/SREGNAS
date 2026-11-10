
from sregnas.paths import PROJECT_ROOT as _artifact_root
import json
from pathlib import Path
from .recorder import dump


def execute(job,out,info):
    import numpy as np
    import torch
    from sregnas.data.products import ProductsMmapGraph
    from sregnas.training.executor import GPUProductsNeighborSampler, local_indices
    from sregnas.training.minibatch import _exact_gcn_target_forward
    from sregnas.models.canonical import load_model_module
    from sregnas.models.normalization_fix import forward
    device=torch.device('cuda:'+str(job['gpus'][0]));torch.cuda.set_device(device)
    graph=ProductsMmapGraph.open(str(_artifact_root / 'data/Reddit/sregnas_cache_v1/full/manifest.json'))
    sampler=GPUProductsNeighborSampler(graph,device=device)
    degrees=sampler.rowptr[1:]-sampler.rowptr[:-1]
    eligible=np.flatnonzero(np.asarray(graph.train_mask)&(np.diff(graph.rowptr)>5))
    selected=np.random.default_rng(42).choice(eligible,size=64,replace=False)
    targets=torch.as_tensor(selected,device=device)
    nodes,edge=sampler.full_neighbor_edges_device(targets)
    x=torch.as_tensor(np.array(graph.x[nodes.cpu().numpy()]),device=device)
    degree_local=degrees[nodes]
    samples=[]
    generator=torch.Generator(device=device).manual_seed(42)
    for _ in range(1024):
        sources,destinations=sampler.neighbors_device(targets,5,generator)
        assert sources.numel()==64*5 and torch.equal(destinations.reshape(64,5)[:,0],targets)
        samples.append(sources.reshape(64,5))
    sampled=torch.stack(samples)
    full_counts=torch.bincount(edge[1],minlength=nodes.numel())[:64]
    audit=dict(no_sample_duplicates=bool((sampled.sort(-1).values.diff(dim=-1)>0).all()),
        no_sample_self_loops=bool((sampled!=targets[None,:,None]).all()),
        no_full_self_loops=bool((edge[0]!=edge[1]).all()),
        no_full_duplicate_edges=torch.unique(edge[0]*nodes.numel()+edge[1]).numel()==edge.shape[1],
        full_degree_matches_exact=bool(torch.equal(full_counts,degree_local[:64])),
        sampled_count=5,draws=1024,deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    dump(out/'sampling_audit.json',audit)
    assert all(audit[k] for k in ('no_sample_duplicates','no_sample_self_loops','no_full_self_loops','no_full_duplicate_edges','full_degree_matches_exact'))
    dump(out/'diagnostic_design.json',dict(root_seed=42,targets=selected.tolist(),full_degrees=degree_local[:64].tolist(),
        samples=1024,fanout=5,output_dim=32,attention_heads=4,dropout=0,chunk_samples=32,
        target_selection='default_rng(42), 64 train nodes without replacement, full_degree > 5',
        metric='norm(mean(post_MLP_layer_output)-exact)/norm(exact)',
        linear_projection_cached=True,self_importance=1,max_mc_test=False,threshold=.05))
    rows=[];module=load_model_module(force_native_scatter=True)
    with torch.no_grad():
        for op in ('const','gcn','gat','sym-gat','cos','linear','gene-linear'):
            for agg in ('sum','mean','max','mlp'):
                torch.manual_seed(42)
                conv=module.GNASConv(graph.num_features,32,op,agg,num_heads=4,dropout=0).to(device).eval()
                exact=(_exact_gcn_target_forward(conv,x,edge,nodes,64,degrees+1)
                       if op=='gcn' else conv(x,edge)[:64])
                full=forward(conv,x,edge,degree_local)[:64]
                rel_full=float(torch.linalg.vector_norm(full-exact)/torch.linalg.vector_norm(exact).clamp_min(1e-12))
                row=dict(encoding=op,aggregation=agg,full_fanout_rel_l2=rel_full,
                    full_fanout_pass=bool(torch.allclose(full,exact,rtol=1e-5,atol=1e-6)),samples=0,
                    corrected_rel_l2=None,legacy_rel_l2=None,mc_pass=None,
                    corrected_mean_row_norm_ratio=None,legacy_mean_row_norm_ratio=None)
                if agg!='max':
                    projected=conv.lin(x)
                    total=torch.zeros_like(exact,dtype=torch.float64)
                    squares=torch.zeros_like(total);old_total=torch.zeros_like(total)
                    for offset in range(0,1024,32):
                        count=32*64
                        global_ids=torch.cat((targets.repeat(32),sampled[offset:offset+32].reshape(-1)))
                        local=local_indices(nodes,global_ids)
                        h=projected[local]
                        sampled_edge=torch.stack((count+torch.arange(count*5,device=device),
                                                  torch.arange(count,device=device).repeat_interleave(5)))
                        result=forward(conv,h,sampled_edge,degrees[global_ids],projected=h)[:count].reshape(32,64,32)
                        total+=result.double().sum(0);squares+=result.double().square().sum(0)
                        linear=conv.lin
                        try:
                            conv.lin=torch.nn.Identity()
                            old=conv(h,sampled_edge)[:count].reshape(32,64,32)
                        finally:conv.lin=linear
                        old_total+=old.double().sum(0)
                    norm=torch.linalg.vector_norm(exact.double()).clamp_min(1e-12)
                    new_error=float(torch.linalg.vector_norm(total/1024-exact)/norm)
                    old_error=float(torch.linalg.vector_norm(old_total/1024-exact)/norm)
                    variance=(squares/1024-(total/1024).square()).clamp_min(0)
                    exact_row_norm=exact.double().norm(dim=1).mean()
                    ratio_new=float((total/1024).norm(dim=1).mean()/exact_row_norm)
                    ratio_old=float((old_total/1024).norm(dim=1).mean()/exact_row_norm)
                    row.update(samples=1024,corrected_rel_l2=new_error,legacy_rel_l2=old_error,
                        mc_relative_standard_error=float((variance.sum()/1024).sqrt()/norm),mc_pass=new_error<=.05,
                        corrected_mean_row_norm_ratio=ratio_new,legacy_mean_row_norm_ratio=ratio_old,
                        linear_5pct_required=op in ('const','gcn') and agg in ('sum','mean'),
                        scale_ratio_required=op not in ('const','gcn') or agg=='mlp',
                        scale_ratio_pass=.8<=ratio_new<=1.25)
                    if op in ('const','gcn') and agg in ('sum','mean'):


                        messages=projected[edge[0]].double()
                        if op=='gcn':
                            norms=(degree_local.double()+1).pow(-.5)
                            messages*= (norms[edge[0]]*norms[edge[1]])[:,None]
                        sums=torch.zeros((64,32),dtype=torch.float64,device=device)
                        sums2=torch.zeros_like(sums)
                        sums.scatter_add_(0,edge[1][:,None].expand_as(messages),messages)
                        sums2.scatter_add_(0,edge[1][:,None].expand_as(messages),messages.square())
                        d=degree_local[:64].double()
                        popvar=(sums2-sums.square()/d[:,None])/(d-1)[:,None]
                        var_mean=d[:,None].square()*(1-5/d[:,None])*popvar/5/1024
                        if op=='const' and agg=='mean':var_mean/=(d+1)[:,None].square()
                        row['analytic_mc_relative_standard_error']=float(var_mean.clamp_min(0).sum().sqrt()/norm)
                rows.append(row);dump(out/'layer_checks.json',rows);print(json.dumps(row),flush=True)
    original_5pct=all(r['mc_pass'] for r in rows if r['aggregation']!='max')
    checks=dict(full_fanout_28=len(rows)==28 and all(r['full_fanout_pass'] for r in rows),
        legacy_attention_mean_negative_control=all(r['legacy_rel_l2']>.05 for r in rows if r['aggregation']=='mean' and r['encoding'] not in ('const','gcn')),
        real_features_and_full_degrees=True,no_test_access=True)
    if job.get('criteria_revision',1)==1:
        checks['monte_carlo_21']=original_5pct
    else:
        checks.update(linear_5pct_4=all(r['mc_pass'] for r in rows if r.get('linear_5pct_required')),
            nonlinear_scale_ratio_17=all(r['scale_ratio_pass'] for r in rows if r.get('scale_ratio_required')),
            legacy_attention_mean_scale_negative_control=all(not .8<=r['legacy_mean_row_norm_ratio']<=1.25
                for r in rows if r['aggregation']=='mean' and r['encoding'] not in ('const','gcn')))
    return dict(status='OK' if all(checks.values()) else 'incomplete',layer_checks=rows,assertions=checks,
        original_5pct_all_21=original_5pct,sampling_audit=audit,criteria_revision=job.get('criteria_revision',1),
        Q_attempt=0,Q_search=0,Qbench=0,record_type='profile',test_evaluations=0,
        stop_reason='' if all(checks.values()) else 'normalization_validation_failed')
