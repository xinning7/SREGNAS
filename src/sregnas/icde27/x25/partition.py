
import hashlib
import os
import time

KERNELS=('_weights','_edge_statistics','_bfs','_followers','_quotient','_recombine')

def validate_explicit_arm(choice):
    pass
    import math
    arm=os.environ.get('SREGNAS_X25_PARTITION_ARM')
    assert os.environ.get('SREGNAS_X25_PARTITION_RECORDS')=='1', 'explicit arm recording flag required'
    assert os.environ.get('SREGNAS_PARTITION_MODE')=='proxy_aware', 'explicit arm requires proxy_aware'
    assert arm in ('da_random','la_metis'), 'explicit arm must be authorized in the process environment'
    assert choice['forced_quality_arm']==choice['selected']==arm, 'explicit arm receipt mismatch'
    assert choice['test_labels_used'] is False and choice['label_scope']=='train_val'
    assert set(choice['scores'])==set(choice['candidate_membership_sha256'])=={arm}
    score=choice['scores'][arm]
    assert all(math.isfinite(score[k]) for k in ('SNR','LA','FI')), 'nonfinite explicit arm score'
    assert choice['x25_preparation']['label_boundary_assertions_pass'] is True
    return arm

def cache_counters(module):
    return {name:{'hits':sum(getattr(module,name)._cache_hits.values()),
                  'misses':sum(getattr(module,name)._cache_misses.values()),
                  'signatures':[str(s) for s in getattr(module,name).signatures]} for name in KERNELS}

def recorded_selection(config,rowptr,col,y,split):
    import numpy as np
    from scipy.sparse import csr_matrix
    from sregnas.data import proxy_partition as pp
    from sregnas.icde27 import ledger
    pp.strata(y,split)
    assert np.all(y[split>=2]==-1)
    times={};originals={};calls={};before=cache_counters(pp)
    def measured(name,fn):
        def call(*args,**kwargs):
            start=time.perf_counter()
            try:return fn(*args,**kwargs)
            finally:
                times[name]=times.get(name,0.)+time.perf_counter()-start
                calls[name]=calls.get(name,0)+1
        return call
    for name in ('da_random','la_metis','fi_metrics'):
        originals[name]=getattr(pp,name);setattr(pp,name,measured(name,originals[name]))
    start=time.perf_counter();forced=os.environ.get('SREGNAS_X25_PARTITION_ARM')
    try:
        if forced:
            if forced not in ('la_metis','da_random'):raise ValueError('unknown X25 partition arm')
            A=csr_matrix((np.ones(len(col),dtype=np.int8),col,rowptr),shape=(len(y),len(y)))
            part,details=getattr(pp,forced)(A,y,split,config.num_gpus,seed=config.partition_seed,return_info=True)
            score=pp.fi_metrics(A,y,split,part,config.num_gpus,config.fanout_a)
            candidates={forced:part}
            info=dict(schema='partition_proxy_aware_v1',partition_mode='proxy_aware',requested_strategy='fi',
                selected=forced,forced_quality_arm=forced,random_seed=config.partition_seed,label_scope='train_val',
                test_labels_used=False,scores={forced:score},criterion='explicit X25 Table VI(a) arm; not the automatic selection rule',
                candidate_membership_sha256={forced:hashlib.sha256(part.astype(np.int64).tobytes()).hexdigest()},
                **({'LA':details} if forced=='la_metis' else {'DA':details}))
        else:
            candidates,info=pp.select_from_config(config,rowptr,col,y,split)
    finally:
        for name,fn in originals.items():setattr(pp,name,fn)
    total=time.perf_counter()-start;after=cache_counters(pp)
    receipt=dict(component_s=times,component_calls=calls,selection_total_s=total,
        partition_s=sum(times.get(n,0.) for n in ('da_random','la_metis')),
        FI_s=times.get('fi_metrics',0.),
        label_boundary_assertions_pass=True,test_or_unassigned_label_count=int((split>=2).sum()),
        test_labels_all_minus_one=True,train_val_labels_available=True,
        kernel_cache_before=before,kernel_cache_after=after,forced_arm=forced,
        numba_cache_dir=os.environ.get('NUMBA_CACHE_DIR'))
    info['x25_preparation']=receipt
    ledger.event('x25_partition_preparation',selected=info['selected'],
                 candidate_membership_sha256=info['candidate_membership_sha256'],**receipt)
    return candidates,info
