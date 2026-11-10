import json


import pytest


import torch


from sregnas.training.sampled_clip import FLAG, final_training_clip


from sregnas.core import Architecture


@pytest.mark.parametrize('dataset,mode', [('flickr','retrain'),('ogbn-arxiv','retrain'),('reddit','proxy'),('ogbn-products','proxy'),('ogbn-products','full'),('reddit','cost_calibration')])
def test_fullbatch_and_proxy_scope_never_clips(monkeypatch,dataset,mode):
    monkeypatch.setenv(FLAG,'1.0')
    monkeypatch.setattr(torch.nn.utils,'clip_grad_norm_',lambda *a,**k:pytest.fail('forbidden clip'))
    model=torch.nn.Linear(2,2);opt=torch.optim.Adam(model.parameters())
    with final_training_clip(opt,dataset=dataset,mode=mode) as state:
        model(torch.ones(2,2)).sum().backward();opt.step()
    assert state.accounting(1)['calls']==0


def test_explicit_off_preserves_weights_and_rng(monkeypatch):
    monkeypatch.setenv(FLAG,'0')
    torch.manual_seed(42);a=torch.nn.Linear(2,2);b=torch.nn.Linear(2,2);b.load_state_dict(a.state_dict())
    rng=torch.random.get_rng_state().clone()
    for model in (a,b):
        opt=torch.optim.Adam(model.parameters())
        if model is b:
            with final_training_clip(opt,dataset='reddit',mode='retrain') as state:
                model(torch.ones(2,2)).sum().backward();opt.step()
            assert state.accounting(1)['calls']==0
        else:model(torch.ones(2,2)).sum().backward();opt.step()
    for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert torch.equal(rng,torch.random.get_rng_state())


@pytest.mark.parametrize('dataset',['reddit','ogbn-products'])
def test_default_one_matches_explicit_one_and_bounds_gradient(monkeypatch,dataset):
    outcomes=[]
    for setting in (None,'1.0'):
        if setting is None:monkeypatch.delenv(FLAG,raising=False)
        else:monkeypatch.setenv(FLAG,setting)
        torch.manual_seed(42);model=torch.nn.Linear(2,2);opt=torch.optim.Adam(model.parameters())
        with final_training_clip(opt,dataset=dataset,mode='retrain') as state:
            for _ in range(3):
                opt.zero_grad();(100*model(torch.ones(2,2)).sum()).backward();opt.step()
                norm=torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in model.parameters()]))
                assert norm<=1.000001
        assert state.accounting(3)['calls']==3 and state.max_norm==1.0
        outcomes.append(([p.detach().clone() for p in model.parameters()],torch.random.get_rng_state().clone()))
    for x,y in zip(outcomes[0][0],outcomes[1][0]):torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert torch.equal(outcomes[0][1],outcomes[1][1])

