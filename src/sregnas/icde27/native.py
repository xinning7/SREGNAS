
import fcntl
from functools import wraps
import json
import os
from pathlib import Path
import time
import uuid
from . import ledger
from .space import canonical_fingerprint


def fingerprint(arch):
    try:return canonical_fingerprint(arch)
    except KeyError:

        from sregnas.core.architecture import Architecture
        return Architecture(arch).fingerprint()


def reset_rng(worker, arch, config, *, initialization):
    if os.environ.get('SREGNAS_ICDE27_NATIVE_RNG')!='1':return
    if os.environ.get('SREGNAS_X22_QUALIFY_NUMERICS')=='1':
        from sregnas.icde27.x22.qualify_numerics import enable
        enable()
    from sregnas.training.minibatch import _hash_seed,set_retrain_seed
    fp=fingerprint(arch)
    partition=int(worker.worker_id)%int(worker.base_config['logical_partitions'])
    inputs=[fp,'search',0] if initialization else [fp,'search',0,partition]
    seed=_hash_seed(*inputs);set_retrain_seed(seed)
    import hashlib
    import torch
    ledger.event('native_rng_reset',segment_id=config.get('_icde27_segment_id'),
        scope='initialization' if initialization else 'partition_training',
        fingerprint=fp,partition_id=partition,seed=seed,seed_inputs=inputs,
        seed_function='sregnas.training.minibatch._hash_seed',
        torch_cpu_state_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        torch_cuda_state_sha256=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
            if torch.cuda.is_available() else None)
    if not initialization:
        ledger.event('native_training_start',segment_id=config.get('_icde27_segment_id'),
            fingerprint=fp,partition_id=partition,seed=seed)


def parent_launch(arch, audit, partition):
    pass
    identity=f"native:{audit['phase']}:{audit['arch_id']}"
    root=Path(os.environ[ledger.ENV])
    with (root/'native_launch.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        indexed = os.environ.get('SREGNAS_X23_LEDGER_INDEX') == '1'
        if indexed:
            from .x23.ledger_index import lookup
            existing = lookup('launch_ledger.jsonl', identity)
        else:
            existing=[r for r in ledger.rows('launch_ledger.jsonl') if r['identity']==identity]
        if existing:
            parent=existing[0]
            previous = lookup('native_segments.jsonl', (parent['launch_id'], partition)) if indexed else [
                r for r in ledger.rows('native_segments.jsonl')
                if r['parent_candidate']==parent['launch_id'] and r['partition_id']==partition]
            if previous:

                retry=ledger.launch('retry',identity=identity,stage=audit['phase'],
                    parent_candidate=parent['launch_id'],fingerprint=fingerprint(arch),backend='fastgnas-native')
                return parent,retry
            return parent,None
        kind='candidate' if audit['phase']=='search' else 'probe' if audit['phase']=='capacity_probe' else 'warmup'
        parent=ledger.launch(kind,identity=identity,stage=audit['phase'],
            fingerprint=fingerprint(arch),architecture=arch,backend='fastgnas-native',arch_id=audit['arch_id'])
        return parent,None


def training_segment(function):
    @wraps(function)
    def wrapped(worker,arch_spec,state_dict,config):
        if not ledger.enabled() and not worker.base_config.get('icde27_gnn_hparams'):
            return function(worker,arch_spec,state_dict,config)
        config=dict(config)

        config.update(worker.base_config.get('icde27_gnn_hparams',{}))
        if not ledger.enabled():return function(worker,arch_spec,state_dict,config)
        audit=config['_timing_audit'];partition=worker.worker_id%worker.base_config['logical_partitions']
        parent,retry=parent_launch(arch_spec,audit,partition)
        sid=uuid.uuid4().hex;config['_icde27_segment_id']=sid
        fields=dict(segment_id=sid,parent_candidate=parent['launch_id'],
            attempt_id=retry['launch_id'] if retry else parent['launch_id'],
            arch_id=audit['arch_id'],fingerprint=fingerprint(arch_spec),phase=audit['phase'],
            partition_id=partition,worker=worker.worker_id,physical_gpu=worker.gpu_id,
            rotation_step=audit.get('rotation_step'),
            expected_optimizer_steps=int(config['steps_per_epoch'])*int(config['epochs_per_rotation']),
            nominal_scheduler_budget=audit.get('nominal_scheduler_budget'),
            migration_bytes_in=sum(t.numel()*t.element_size() for t in (state_dict or {}).values()),
            dropout=config.get('dropout'),lr=config.get('lr'),weight_decay=config.get('weight_decay'))
        started=ledger.append('native_segments.jsonl',dict(event='segment_start',**fields))
        status='incomplete';result=None
        try:

            if os.environ.get('SREGNAS_X18_COST_PROFILE') in ('sync', 'control', 'trace') and audit['phase'] == 'search':
                from .x18.cost_hooks import native_call
                result=native_call(function,worker,arch_spec,state_dict,config)
            elif os.environ.get('SREGNAS_P2P6_PROFILE') == '1':
                from .p2p6.profiling import capture
                with capture(f"native_worker{worker.worker_id}_{fields['fingerprint']}_{partition}"):
                    if os.environ.get('SREGNAS_EXP3_FINE_PROFILE') == '1':
                        from .p2p6.exp3_hooks import native_segment
                        result=native_segment(function,worker,arch_spec,state_dict,config)
                    else:
                        result=function(worker,arch_spec,state_dict,config)
            else:
                result=function(worker,arch_spec,state_dict,config)
            if os.environ.get('SREGNAS_X22_INCREMENTAL_AUDIT')=='1':
                from .x22.origins import audit
                audit()
            status='OK'
            return result
        except BaseException as exc:
            status='OOM' if 'out of memory' in str(exc).lower() else 'incomplete'
            raise
        finally:

            ledger.append('native_segments.jsonl',dict(event='segment_end',**fields,status=status,
                start_perf_s=started['perf_s'],optimizer_steps=config.get('_icde27_optimizer_steps'),
                migration_bytes_out=sum(t.numel()*t.element_size() for t in result[1].values()) if result else None))
    return wrapped
