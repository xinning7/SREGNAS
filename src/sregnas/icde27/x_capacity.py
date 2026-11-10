
import json
import os
from pathlib import Path
import tempfile

from .recorder import dump, read_rows


def execute(job,out,info):
    import ray
    from sregnas.config import load_config
    from sregnas.paths import PROJECT_ROOT as ROOT
    from sregnas.comparison.search import native_config
    from sregnas.baselines.fastgnas.search import merge_dataset_config,set_random_seed
    from sregnas.baselines.fastgnas.data.partition import PartitionGraph
    from sregnas.baselines.fastgnas.trainer.worker import TrainerWorker
    from sregnas.baselines.fastgnas.core.scheduling.rotation_scheduler import RotationScheduler
    from .p2p6.resources import mark_stage
    cfg=load_config(job['config'])
    config,name=native_config(cfg,ROOT/'configs/icde27/fastgnas_native.yaml',out)
    config=merge_dataset_config(config,name)
    training=config['training'];training['logical_partitions']=8
    if job['dataset']=='Products':
        from sregnas.comparison.products_protocol import DIRECTIVE,assert_native_products_config
        os.environ['SREGNAS_PRODUCTS_PROTOCOL']=str(DIRECTIVE)
        assert_native_products_config(config)
    os.environ.update(RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES='1',TIMING_RUN_DIR=str(out),
        TIMING_RAY_TMP=tempfile.mkdtemp(prefix='x-capacity-',dir='/tmp'),SREGNAS_ICDE27_NATIVE_RNG='1')

    set_random_seed(42)
    dump(out/'native_config.json',config)
    ray.init(num_gpus=2,num_cpus=40,resources={'pilot_gpu_0':4,'pilot_gpu_1':4},
        include_dashboard=False,_temp_dir=os.environ['TIMING_RAY_TMP'])
    try:
        mark_stage(out,'prep/partition')
        parts=PartitionGraph.get_partitions(dataset_name=name,num_partitions=8,max_hops=2,
            strategy=config['fastgnas_backends']['partition_strategy'])
        training['num_classes']=max(int(p.y.max())+1 for p in parts)
        mark_stage(out,'prep/worker_init')
        workers=[TrainerWorker.options(num_gpus=0,num_cpus=4,resources={f'pilot_gpu_{i%2}':1},
            runtime_env={'env_vars':{'CUDA_VISIBLE_DEVICES':str(i%2),'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES':'1'}}).remote(
                worker_id=i,gpu_id=i%2,data_partition=parts[i],config=training,collector_ref=None) for i in range(8)]
        row=job['architecture'];fp=row['fingerprint'];arch=json.loads(row['native_architecture_json'])
        scheduler=RotationScheduler(8)
        scheduler.initialize([dict(arch_id=fp,arch_spec=arch)],training)
        tasks=scheduler.schedule_next()
        assert len(tasks)==8 and {t.arch_id for t in tasks}=={fp}
        dump(out/'capacity_plan.json',dict(unique_architectures=1,S=8,P=2,
            policy='one simultaneous capacity wave; identical heaviest architecture on eight logical workers; no controller, warmup or accuracy selection',
            tasks=[dict(worker=t.worker_id,gpu=t.worker_id%2,config=t.metadata['config_overrides']) for t in tasks]))

        futures=[]
        for task in tasks:
            task_cfg=dict(task.metadata['config_overrides'])
            task_cfg['_timing_audit']=dict(phase='capacity_probe',arch_id=fp,worker_id=task.worker_id,
                rotation_step=0,expected_steps=task_cfg['steps_per_epoch'],nominal_scheduler_budget=scheduler.total_batches_per_arch)
            futures.append(workers[task.worker_id].train_model_on_partition.remote(
                arch_spec=arch,state_dict=None,config=task_cfg))
        ray.get(futures)
        segments=[s for s in read_rows(out/'native_segments.jsonl') if s['event']=='segment_end']
        checks=dict(single_unique_architecture={s['fingerprint'] for s in segments}=={fp},
            eight_partitions_complete=len(segments)==8 and {s['partition_id'] for s in segments}==set(range(8)),
            all_segments_ok=all(s['status']=='OK' for s in segments),
            scheduler_steps_exact=all(s['optimizer_steps']==s['expected_optimizer_steps'] for s in segments))
        return dict(status='OK',backend='fastgnas-native',capacity_status='measured_fit',
            capacity_scope='native S8/P2 single-heaviest simultaneous capacity wave',fingerprint=fp,
            Q_attempt=0,Q_search=0,Q_probe=1,K=0,completed_candidates=1,assertions=checks)
    finally:
        ray.shutdown()
