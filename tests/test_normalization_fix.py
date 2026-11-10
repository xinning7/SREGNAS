import itertools
import torch
import pytest
from sregnas.models.canonical import load_model_module
from sregnas.models.normalization_fix import forward, DegreeEdges, attach_degrees

OPS=('const','gcn','gat','sym-gat','cos','linear','gene-linear')
AGG=('sum','mean','max','mlp')


@pytest.mark.parametrize('op,agg',itertools.product(OPS,AGG))
def test_full_fanout_matches_legacy_exact_28(op,agg):
    module=load_model_module(force_native_scatter=True)
    torch.manual_seed(42)

    edges=[(i,j) for i in range(1,17) for j in range(i+1,17) if (i*j+i+j)%4==0]
    edge=torch.tensor(edges+[(j,i) for i,j in edges],dtype=torch.long).T
    degree=torch.bincount(edge[1],minlength=17)
    x=torch.randn(17,11,dtype=torch.float64)
    conv=module.GNASConv(11,8,op,agg,num_heads=2,dropout=0).double().eval()
    expected=conv(x,edge)
    actual=forward(conv,x,edge,degree)
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)


def test_ht_const_mean_exact_expectation_and_unscaled_self():
    module=load_model_module(force_native_scatter=True)
    conv=module.GNASConv(1,1,'const','mean',dropout=0).double().eval()
    with torch.no_grad():conv.lin.weight.fill_(1)
    x=torch.tensor([[7.],[1.],[2.],[12.]],dtype=torch.float64)
    full_degree=torch.tensor([3,1,1,1])
    estimates=[forward(conv,x,torch.tensor([[j],[0]]),full_degree)[0] for j in (1,2,3)]
    torch.testing.assert_close(torch.stack(estimates).mean(0),torch.tensor([5.5],dtype=torch.float64))


@pytest.mark.parametrize('op,agg',itertools.product(('const','gcn'),('sum','mean')))
def test_linear_estimators_exact_expectation_over_all_subsets(op,agg):
    module=load_model_module(force_native_scatter=True);torch.manual_seed(42)
    conv=module.GNASConv(3,4,op,agg,dropout=0).double().eval()
    x=torch.randn(6,3,dtype=torch.float64)
    edges=[(0,j) for j in range(1,6)]+[(1,2),(2,3),(3,4)]
    edge=torch.tensor(edges+[(j,i) for i,j in edges]).T
    degree=torch.bincount(edge[1],minlength=6)
    exact=conv(x,edge)[0]
    outputs=[forward(conv,x,torch.tensor([list(pair),[0,0]]),degree)[0]
             for pair in itertools.combinations(range(1,6),2)]
    torch.testing.assert_close(torch.stack(outputs).mean(0),exact,rtol=1e-12,atol=1e-12)


def test_off_context_preserves_original_edges_and_no_rng(monkeypatch):
    monkeypatch.setenv('SREGNAS_CANONICAL_NORMALIZATION_FIX','0')
    edges=[torch.tensor([[0,1],[1,0]])]*2
    before=torch.random.get_rng_state().clone()
    assert attach_degrees(None,None,None,edges,'cpu') is edges
    assert torch.equal(torch.random.get_rng_state(),before)


def test_full_attention_gradient_matches_exact():
    module=load_model_module(force_native_scatter=True);torch.manual_seed(42)
    conv=module.GNASConv(3,4,'gat','mean',num_heads=2).double()
    edge=torch.tensor([[0,1,1,2],[1,0,2,1]])
    x=torch.randn(3,3,dtype=torch.float64,requires_grad=True)
    expected=conv(x,edge).square().sum();expected.backward();grad=x.grad.clone();x.grad=None
    actual=forward(conv,x,edge,torch.tensor([1,2,1])).square().sum();actual.backward()
    torch.testing.assert_close(x.grad,grad,rtol=1e-10,atol=1e-10)


def test_missing_full_graph_degree_rejected():
    module=load_model_module(force_native_scatter=True)
    conv=module.GNASConv(1,1,'gcn','sum')
    with pytest.raises(ValueError,match='exceeds'):
        forward(conv,torch.ones(2,1),torch.tensor([[0],[1]]),torch.zeros(2))


def test_degree_context_uses_full_graph_not_sample(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    monkeypatch.setenv('SREGNAS_CANONICAL_NORMALIZATION_FIX','1')
    graph=SimpleNamespace(rowptr=np.array([0,1,4,6]))
    sampler=SimpleNamespace();batch=SimpleNamespace(node_ids=np.array([2,0]))
    edges=[torch.tensor([[0],[1]])]*2
    attached=attach_degrees(graph,sampler,batch,edges,'cpu')
    assert torch.equal(attached.nf_full_degrees,torch.tensor([2,1]))
    assert attached[0] is edges[0]


def test_cache_distinguishes_enabled_normalization(tmp_path):
    from sregnas.icde27.quality.cache import training_key
    graph=tmp_path/'manifest.json';graph.write_text('{}')
    spec=dict(architecture=[],fingerprint='fp',dataset='reddit',epochs=25,lr=.01,weight_decay=.0005,
        dropout=0,sampler_impl='gpu',feature_residency='gpu',inference_residency='gpu',replica=0,
        retrain_seed_policy='fingerprint_retrain_replica',graph_path=str(graph))
    info=dict(backend='canonical',graph_hash='graph',split_hash='split')
    old=training_key(spec,info,sources={})
    new=training_key(spec,dict(info,normalization_fix=True,normalization_fix_schema='nf',normalization_fix_sha256='source'),sources={})
    assert old['key_hash']!=new['key_hash']
    assert training_key(spec,dict(info,normalization_fix=False),sources={})==old
    diagnostic=training_key(spec,dict(info,deterministic_diagnostic=True),sources={})
    assert diagnostic['key_hash']!=old['key_hash']
