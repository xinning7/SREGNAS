


import numpy as np


import pytest


import scipy.sparse as sp


from sregnas.data import proxy_partition as p


def graph(n=800, density=.015, seed=91):
    rng = np.random.default_rng(seed)
    a = sp.random(n, n, density=density, random_state=rng, format="csr")
    a = ((a+a.T) > 0).astype(np.int8)
    a.setdiag(0)
    a.eliminate_zeros()
    a.sort_indices()
    split = rng.integers(0, 3, size=n)
    y = np.arange(n) % 4
    y[split == 2] = -1
    return a, y, split


def test_thresholds():
    assert [p.d_star(n) for n in (2,4,8,16)] == [11,23,48,96]


def test_twenty_seed_class_mean():
    a,y,split = graph(density=.12)
    counts = np.zeros((20,8,4))
    for seed in range(20):
        part = p.da_random(a,y,split,8,seed=seed)
        for c in range(4):
            counts[seed,:,c] = np.bincount(part[y == c],minlength=8)
    target = np.bincount(y[y >= 0],minlength=4)/8
    assert np.max(np.abs(counts.mean(0)-target)) < 1.0


@pytest.mark.parametrize("builder", [p.da_random,p.la_metis,p.select_partition])
def test_test_labels_rejected_at_entry(builder):
    a,y,split = graph()
    y[split == 2] = 999
    with pytest.raises(AssertionError,match="test labels"):
        builder(a,y,split,8)


def test_no_metis_when_da_snr_passes(monkeypatch):
    a,y,split = graph(density=.3)
    def forbidden(*a,**k):
        pytest.fail("METIS ran despite passing DA-Random SNR")
    monkeypatch.setattr(p,"la_metis",forbidden)
    candidates,info = p.select_partition(a,y,split,8,seed=42)
    assert info["selected"] == "da_random"
    assert set(candidates) == {"da_random"}


def test_selection_fi_and_da_tie(monkeypatch):
    a,y,split = graph()
    da = np.arange(len(y)) % 8
    la = (da+1) % 8
    monkeypatch.setattr(p,"da_random",lambda *a,**k:(da,{}))
    monkeypatch.setattr(p,"la_metis",lambda *a,**k:(la,{}))
    scores = {id(da):dict(SNR=.8,FI=.6),id(la):dict(SNR=.8,FI=.6)}
    monkeypatch.setattr(p,"fi_metrics",lambda a,y,s,part,*x:scores[id(part)])
    assert p.select_partition(a,y,split,8)[1]["selected"] == "da_random"
    scores[id(la)]["FI"] = .7
    assert p.select_partition(a,y,split,8)[1]["selected"] == "la_metis"


def test_optin_masks_before_boundary(monkeypatch):
    from sregnas.config import GraphConfig
    from sregnas.data.partition_selection import select_partition
    from sregnas.icde27.x18.cold_prep import partitions
    a,y,split = graph()
    y[split == 2] = 999
    cfg = GraphConfig(partition_strategy="fi",num_gpus=8)
    observed = []
    def boundary(config,rowptr,col,labels,splits):
        assert np.all(labels[splits == 2] == -1)
        observed.append(True)
        return {"da_random":np.arange(len(y))%8},{"selected":"da_random"}
    monkeypatch.setenv("SREGNAS_PARTITION_MODE","proxy_aware")
    monkeypatch.setattr(p,"select_from_config",boundary)
    select_partition(cfg,a.indptr,a.indices,y,split==0,split==1)
    partitions(cfg,a.indptr,a.indices,dict(labels=y,train_mask=split==0,val_mask=split==1))
    assert len(observed) == 2


def test_legacy_default_and_explicit_identical(monkeypatch):
    from sregnas.config import GraphConfig
    from sregnas.data.partition_selection import select_partition
    from sregnas.icde27.x18.cold_prep import partitions
    a,y,split = graph()
    cfg = GraphConfig(partition_strategy="fi",num_gpus=8)
    monkeypatch.setattr("sregnas.data.sampling.metis_partition",lambda *a:np.arange(len(y))%8)
    raw = dict(labels=y,train_mask=split==0,val_mask=split==1)
    monkeypatch.delenv("SREGNAS_PARTITION_MODE",raising=False)
    default = select_partition(cfg,a.indptr,a.indices,y,split==0,split==1)
    cold_default = partitions(cfg,a.indptr,a.indices,raw)
    monkeypatch.setenv("SREGNAS_PARTITION_MODE","legacy")
    explicit = select_partition(cfg,a.indptr,a.indices,y,split==0,split==1)
    cold_explicit = partitions(cfg,a.indptr,a.indices,raw)
    np.testing.assert_array_equal(default[0],explicit[0])
    assert default[1] == explicit[1]
    assert cold_default[1] == cold_explicit[1]
    for key in cold_default[0]:
        np.testing.assert_array_equal(cold_default[0][key],cold_explicit[0][key])


def test_partition_cache_modes_are_separated(monkeypatch):
    from sregnas.config import ExperimentConfig,GraphConfig
    from sregnas.execution.orchestrator import _graph_cache_key,_validate_partition_cache
    cfg = ExperimentConfig(graph=GraphConfig(partition_strategy="fi"))
    manifest = dict(vars(cfg.graph),num_shards=cfg.graph.num_gpus,
        partition_selection=dict(requested_strategy="fi",random_seed=42,label_scope="train_val"))
    monkeypatch.delenv("SREGNAS_PARTITION_MODE",raising=False)
    legacy = _graph_cache_key(cfg)
    _validate_partition_cache(cfg,manifest)
    monkeypatch.setenv("SREGNAS_PARTITION_MODE","proxy_aware")
    assert _graph_cache_key(cfg) != legacy
    with pytest.raises(ValueError,match="mode"):
        _validate_partition_cache(cfg,manifest)
    manifest["partition_selection"]["partition_mode"] = "proxy_aware"
    _validate_partition_cache(cfg,manifest)

