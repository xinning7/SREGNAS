
import json
from pathlib import Path


def assert_quality_products(cfg,method,path,hardware_path=None):
    record=json.loads(Path(path).read_text())
    assert record['schema']=='icde27_quality_products_v1' and record['user_instruction']
    candidate=dict(method=method,P=len(cfg.runtime.system_gpus),S=cfg.graph.num_gpus,
                   E=cfg.train.proxy_epochs,partition=cfg.graph.partition_strategy)
    assert candidate in record['allowed'],'Products quality variant outside user request'
    assert cfg.search.q==400 and cfg.fixed_x==8 and cfg.search.base_seed==42
    assert cfg.search.generator=='rl' and cfg.search.pool_mode=='independent' and cfg.search.m==16
    assert cfg.train.retrain_epochs==25 and cfg.train.lr==.01 and cfg.train.weight_decay==.0005
    assert cfg.train.dropout==0 and cfg.train.retrain_seed_policy=='fingerprint_retrain_replica'
    assert cfg.train.retrain_replica==0 and cfg.graph.permutation_seed==42 and cfg.graph.partition_seed==42
    assert cfg.graph.partition_label_scope=='train_val' and cfg.graph.fanout_a==cfg.graph.fanout_b==5
    assert cfg.runtime.require_idle_preflight
    assert cfg.runtime.require_fi_gpu==(candidate['partition']=='fi')
    assert all(getattr(cfg.runtime,k)=='gpu' for k in ('sampler_impl','feature_residency','inference_residency'))
    cfg.validate()
    if hardware_path:
        import yaml
        assert yaml.safe_load(Path(hardware_path).read_text())['name_contains']=='A6000'
    return dict(authorization='explicit_quality_user_task',directive_path=str(path),protocol=candidate)
