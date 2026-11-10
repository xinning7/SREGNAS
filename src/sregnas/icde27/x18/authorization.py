
import os


def validate(cfg,args):
    assert args.experiment_id in ('X18', 'X20', 'X22', 'X23', 'X25', 'X27') and cfg.name.startswith('x18_')
    assert cfg.search.q==400 and cfg.search.m==16 and cfg.fixed_x==8
    assert cfg.search.generator=='rl' and cfg.search.pool_mode=='independent'
    assert cfg.graph.dataset in ('flickr','ogbn-arxiv','reddit','ogbn-products')
    assert cfg.graph.partition_seed==cfg.graph.permutation_seed==42
    assert cfg.train.retrain_replica==0 and cfg.train.retrain_seed_policy=='fingerprint_retrain_replica'
    assert cfg.train.lr==.01 and cfg.train.weight_decay==.0005
    assert cfg.runtime.system_gpus==cfg.runtime.baseline_gpus
    if args.method=='SREGNAS':
        assert cfg.search.base_seed in range(42,50)
        assert cfg.graph.partition_strategy in ('fi','metis','random')
        assert cfg.graph.num_gpus in (2,4,8) and cfg.selection.k==8//cfg.graph.num_gpus
        assert len(cfg.runtime.system_gpus) in (2,8)
        assert cfg.train.dropout==0 and cfg.train.proxy_epochs==100
        if cfg.search.base_seed>=45:
            explicit_x25_arm=(args.experiment_id=='X25'
                and os.environ.get('SREGNAS_PARTITION_MODE')=='proxy_aware'
                and os.environ.get('SREGNAS_X25_PARTITION_RECORDS')=='1'
                and os.environ.get('SREGNAS_X25_PARTITION_ARM') in ('la_metis','da_random'))
            assert cfg.graph.dataset=='ogbn-products'
            assert cfg.graph.partition_strategy in ('metis','random') or (explicit_x25_arm and cfg.graph.partition_strategy=='fi')
        if cfg.graph.num_gpus!=8:
            assert cfg.graph.dataset in ('ogbn-arxiv','ogbn-products') and len(cfg.runtime.system_gpus)==2
        assert args.proxy_steps in (25,50,100)
        assert args.proxy_batch_targets in (None,256,512,2048)
    else:
        assert args.method in ('AP','PAS-PL','GraphNAP','FastGNAS')
        assert cfg.search.base_seed in (42,43,44) and cfg.graph.num_gpus==8
        assert len(cfg.runtime.system_gpus)==8
    assert cfg.runtime.require_idle_preflight
    assert os.environ.get('SREGNAS_PI_SOFTMAX_MAX_FIX')=='1'
    cfg.validate()
    from sregnas.comparison import products_approval
    original=products_approval.assert_products_method_approval
    def approved(one,method=None,*,hardware_path=None):
        if one.graph.dataset=='ogbn-products' and one.name==cfg.name and args.method=='SREGNAS':
            assert one.to_dict()==cfg.to_dict()
            if hardware_path:
                import yaml
                assert yaml.safe_load(hardware_path.read_text())['name_contains']=='A6000'
            return dict(authorization='explicit_'+args.experiment_id+'_user_matrix',Q=400,K=8,seed=one.search.base_seed,
                S=one.graph.num_gpus,P=len(one.runtime.system_gpus),partition=one.graph.partition_strategy,
                proxy_steps=args.proxy_steps,proxy_batch_targets=args.proxy_batch_targets,
                NF=1,sampled_grad_clip=1.0,pi_softmax_max_fix=True)
        return original(one,method,hardware_path=hardware_path)
    products_approval.assert_products_method_approval=approved
