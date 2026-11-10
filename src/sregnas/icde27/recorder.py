
from sregnas.paths import GPU_LOCK as _artifact_lock
import csv
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from sregnas.paths import PROJECT_ROOT as ROOT
from .space import space_hash

LOCK = Path(str(_artifact_lock))
BUDGET = ROOT/'results/icde27/gpu_budget_p0c.json'
DATASETS = {'flickr':'Flickr','ogbn-arxiv':'arXiv','reddit':'Reddit','ogbn-products':'Products'}
PHASES = ['prep','search','nomination+probe+refill','retrain+validation','winner-test','other']


def dump(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def write_table(path, rows, fields):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,delimiter='\t',lineterminator='\n',extrasaction='ignore')
        writer.writeheader()
        writer.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v
                         for k,v in row.items()} for row in rows)


def snapshot(gpus, owner, verified_owned=None):
    from sregnas.execution.telemetry import _owned, classify_gpu_process
    verified_owned = verified_owned if verified_owned is not None else {}
    def command(argv):
        p=subprocess.run(argv,text=True,capture_output=True)
        return dict(argv=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    gpu=command(['nvidia-smi','-i',','.join(map(str,gpus))])
    compute=command(['nvidia-smi','-i',','.join(map(str,gpus)),
        '--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'])
    processes=[]
    for row in csv.reader(compute['stdout'].splitlines()):
        if len(row)<3: continue
        try: pid=int(row[0]); memory=int(row[2])
        except ValueError: continue
        name=row[1].strip()
        baseline=Path(name).name in {'Xorg','X','gnome-shell','kwin_x11','kwin_wayland'}
        processes.append(dict(pid=pid,name=name,memory_mib=memory,baseline=baseline,
            **classify_gpu_process(pid,name,owner,verified_owned)))
    cpu=command(['ps','-eo','pid,ppid,pcpu,pmem,comm','--sort=-pcpu'])
    load=list(os.getloadavg())
    foreign_cpu_cores=0.
    for line in cpu['stdout'].splitlines()[1:]:
        fields=line.split()
        if len(fields)<5:continue
        if fields[4] in {'Xorg','X','gnome-shell','kwin_x11','kwin_wayland'}:continue
        if not _owned(int(fields[0]),owner):foreign_cpu_cores+=float(fields[2])/100
    cpu_capacity=len(os.sched_getaffinity(0))
    return dict(epoch_s=time.time(),perf_s=time.perf_counter(),gpu=gpu,compute=compute,
        processes=processes,baseline_processes=[p for p in processes if p['baseline']],
        ownership_events=[p['ownership_event'] for p in processes if p['ownership_event']],
        desktop_baseline_rows=[line for line in gpu['stdout'].splitlines() if any(x in line for x in ['Xorg','gnome-shell','kwin'])],
        foreign_processes=[p for p in processes if not p['baseline'] and not p['owned']],
        cpu=cpu,cpu_load=load,cpu_capacity=cpu_capacity,foreign_cpu_cores=foreign_cpu_cores,
        cpu_contention_observed=load[0]>cpu_capacity and foreign_cpu_cores>.5,
        cpu_contention_rule='1m load > affinity CPU count and foreign non-desktop CPU > 0.5 core')


def metadata(cfg, method, out, experiment_id, record_type, argv):
    dataset=DATASETS[cfg.graph.dataset]
    hashes=json.loads((ROOT/f'docs/audit/dataset_{dataset}.json').read_text())
    gpus=cfg.runtime.system_gpus
    from sregnas.execution.provenance import implementation_sha256
    from sregnas.models.normalization_fix import enabled,SCHEMA,softmax_max_enabled
    from sregnas.training.sampled_clip import max_norm, SCHEMA as CLIP_SCHEMA
    clip=max_norm() if dataset in ('Reddit','Products') else 0
    nf=enabled() and method!='FastGNAS' and dataset in ('Reddit','Products')
    from sregnas.training.fixed_proxy import options
    proxy = options() if method == 'SREGNAS' else {}
    return dict(run_id=out.name,experiment_id=experiment_id,method=method,dataset=dataset,
        **(dict(version=os.environ['SREGNAS_X18_VERSION']) if os.environ.get('SREGNAS_X18_VERSION') and experiment_id=='X18' else {}),
        environment='E1',backend='fastgnas-native' if method=='FastGNAS' else 'canonical',
        commit=(__import__('sregnas.execution.provenance', fromlist=['commit_hash']).commit_hash() or 'source-archive'),
        implementation_sha256=implementation_sha256(),
        normalization_fix=nf,normalization_fix_schema=SCHEMA if nf else 'legacy',
        normalization_fix_sha256=sha(ROOT/'src/sregnas/models/normalization_fix.py'),
        **(dict(pi_softmax_max_fix='pi_softmax_max_v1') if nf and softmax_max_enabled() else {}),
        **(dict(wall_limit_scope='cold_e2e',process_setup_grace_s=120,process_cleanup_grace_s=120)
           if os.environ.get('SREGNAS_X17_E2E_CAP','0')=='1' else {}),
        **(dict(sampled_grad_clip=clip,sampled_grad_clip_schema=CLIP_SCHEMA,
                 sampled_grad_clip_sha256=sha(ROOT/'src/sregnas/training/sampled_clip.py')) if clip else {}),
        **(dict(proxy_options=proxy,proxy_schema='fixed_layer_target_batch_v1') if proxy else {}),
        **(dict(preparation_version='cold_memory_v2',cold_graph_preparation=True,
                graph_storage='RAM_only_shared_arrays',graph_cache_reads=False)
            if method=='SREGNAS' and os.environ.get('SREGNAS_X18_MEMORY_PREP')=='1' else {}),
        **(dict(x18_preparation={k:os.environ.get(env)=='1' for k,env in (
            ('parallel_permutations','SREGNAS_X18_PARALLEL_PERMUTATIONS'),('stable_integer_csr','SREGNAS_X18_FAST_CSR'))})
            if method=='SREGNAS' and any(os.environ.get(env)=='1' for env in
                ('SREGNAS_X18_PARALLEL_PERMUTATIONS','SREGNAS_X18_FAST_CSR')) else {}),
        config_hash=hashlib.sha256(json.dumps(dict(config=cfg.to_dict(),proxy_options=proxy) if proxy else cfg.to_dict(),sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        graph_hash=hashes['graph_hash'],split_hash=hashes['split_hash'],space_hash=space_hash(),
        P=len(gpus),S=cfg.graph.num_gpus,resource_class='exclusive',
        seed_rule=dict(controller=f'{cfg.search.base_seed} + shard (native manager {cfg.search.base_seed})',
            canonical_search='training.fullbatch.fingerprint_seed(fp); existing rule',
            canonical_retrain='H(fp,retrain,0)',native_retrain='H(architecture,retrain,0)',
            native_search_init='training.minibatch._hash_seed(canonical_fp,search,0) when flag enabled',
            native_search_segment='training.minibatch._hash_seed(canonical_fp,search,0,partition_id) when flag enabled'),
        status='incomplete',record_type=record_type,timing_valid=False,
        alignment_status='unaligned',argv=argv,Q_search=0,Q_attempt=0,Q_trunc=None,Qbench=None,
        Q_requested=cfg.search.q,K=cfg.fixed_x or cfg.graph.num_gpus*cfg.selection.k,
        native_rng_reset=os.environ.get('SREGNAS_ICDE27_NATIVE_RNG')=='1',
        canonical_winner_tie=os.environ.get('SREGNAS_ICDE27_TIE_RULE')=='1',
        tie_occurred=None,config=cfg.to_dict(),phases=[],assertions={})


def recording_limit_reached(out,started,timeout):
    if os.environ.get('SREGNAS_X17_E2E_CAP','0')=='1':
        from sregnas.icde27.x17.timing_limit import reached
        return reached(out,started,timeout)
    return time.perf_counter()-started>=timeout


def run_recorded(*, out, cfg, method, argv, timeout, experiment_id='P0C',record_type='profile',profile=False, gpu_lock=None, quality_only=False, allow_quality_sharing=False):
    pass
    out=Path(out).resolve()
    if out.exists(): raise FileExistsError(out)
    with Path(gpu_lock or LOCK).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        p0c=experiment_id=='P0C'
        budget=(json.loads(BUDGET.read_text()) if BUDGET.exists() else dict(limit_gpu_s=10800,used_gpu_s=0,runs=[])) if p0c else None
        gpus=cfg.runtime.system_gpus
        if p0c:
            allowance=(budget['limit_gpu_s']-budget['used_gpu_s']-120)/len(gpus)
            if allowance<=0: raise RuntimeError('P0-C total GPU budget exhausted')
            timeout=min(timeout,allowance)
        out.mkdir(parents=True)
        info=metadata(cfg,method,out,experiment_id,record_type,argv)
        if profile:
            info.update(Q_requested=0,K=1,search_executed=False,profile_epochs_or_passes=cfg.train.retrain_epochs)
        info['max_wall_seconds']=timeout
        info.update(timing_scope='quality_only' if quality_only else 'exclusive_primary',
                    gpu_lock=str(gpu_lock or LOCK), s0_parallel=os.environ.get('SREGNAS_S0_PARALLEL')=='1')
        dump(out/'run.json',info)
        verified_owned={}
        def observe():
            observation=snapshot(gpus,os.getpid(),verified_owned)
            for row in observation['ownership_events']:
                with (out/'ownership_events.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(row,epoch_s=observation['epoch_s'],
                        perf_s=observation['perf_s']))+'\n')
            return observation
        before=observe(); dump(out/'resources_before.json',before)
        shared_quality = allow_quality_sharing and quality_only
        if (before['foreign_processes'] and not shared_quality) or (before['cpu_contention_observed'] and not quality_only):
            info.update(status='contaminated',blocking_reason='foreign compute process at preflight')
            dump(out/'run.json',info)
            return 2
        from .x_authorization import active as x_active
        if x_active():
            from sregnas.execution.telemetry import gpu_idle_preflight
            release_baseline=gpu_idle_preflight(gpus,owner_pid=os.getpid(),verified_owned=verified_owned)
            dump(out/'release_baseline.json',release_baseline)
        env=dict(os.environ,SREGNAS_ICDE27_RECORD_DIR=str(out),
            SREGNAS_ALLOW_GPU='1',SREGNAS_APPROVED_GPUS=','.join(map(str,gpus)),
            PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
            OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        if os.environ.get('SREGNAS_CPU_THREADS'):
            env.update({key:os.environ['SREGNAS_CPU_THREADS'] for key in
                        ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
        if profile: env['SREGNAS_ICDE27_PROFILE']='1'
        env.pop('CUDA_VISIBLE_DEVICES',None)
        started=time.perf_counter(); timeout_hit=False; observations=[]
        with (out/'raw.log').open('w') as log:
            child=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            from sregnas.execution.telemetry import _start_ticks
            info['pipeline_process']=dict(pid=child.pid,pgid=child.pid,start_ticks=_start_ticks(child.pid),
                recorder_pid=os.getpid(),recorder_pgid=os.getpgrp())
            dump(out/'run.json',info)
            try:
                while child.poll() is None:
                    observation=observe()
                    observations.append(observation)
                    with (out/'resources_during.jsonl').open('a') as f: f.write(json.dumps(observation)+'\n')
                    if timeout is not None and recording_limit_reached(out,started,timeout):
                        timeout_hit=True
                        os.killpg(child.pid,signal.SIGTERM)
                        try: child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid,signal.SIGKILL); child.wait()
                        break
                    time.sleep(.5)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM); child.wait(timeout=10)
        wall=time.perf_counter()-started
        if x_active():
            from .p2p6.cleanup import wait_for_gpu_release
            settled=wait_for_gpu_release(gpus,owner_pid=os.getpid(),verified_owned=verified_owned,
                                        baseline=release_baseline,out=out,max_idle_drift_wait_s=120)
            dump(out/'release_settled.json',settled)
        after=observe();dump(out/'resources_after.json',after)
        charged=wall*len(gpus)
        if p0c:
            budget['used_gpu_s']+=charged
            budget['runs'].append(dict(run_id=out.name,P=len(gpus),wall_s=wall,gpu_s=charged))
            BUDGET.parent.mkdir(parents=True,exist_ok=True);dump(BUDGET,budget)
        info.update(process_wall_s=wall,reserved_gpu_hours=charged/3600,returncode=child.returncode,
            status='timeout' if timeout_hit else 'OK' if child.returncode==0 else 'incomplete',
            timing_valid=not any((x['foreign_processes'] and not shared_quality) or (x['cpu_contention_observed'] and not quality_only) for x in [before,*observations,after]),
            peak_process_memory_GiB=max([p['memory_mib']/1024 for o in observations for p in o['processes'] if p['owned']]+[0]),
            gpu_budget_used_hours=budget['used_gpu_s']/3600 if p0c else None)
        if not info['timing_valid'] and info['status']=='OK':info['status']='contaminated'
        preflight_path=out/'gpu_idle_preflight.json'
        if preflight_path.exists():
            preflight=json.loads(preflight_path.read_text())
            info.update(gpu_idle_wait_seconds=preflight['wait_seconds'],
                gpu_idle_preflight_attempts=preflight['attempts'],gpu_idle_preflight_passed=preflight['idle'])
        decision_path=out/'watchdog_decision.json'
        if decision_path.exists():
            decision=json.loads(decision_path.read_text())
            if (decision.get('decision')=='terminate' and decision.get('termination_requested')
                and decision.get('run_id')==info['run_id']
                and decision.get('pipeline_process')==info['pipeline_process']
                and child.returncode!=0):
                info.update(status='incomplete',stop_reason='projected_over_threshold')
        if profile:
            result=json.loads((out/'profile_result.json').read_text()) if (out/'profile_result.json').exists() else {}
            if result: info.update(profile_result=result,status=result['status'] if not timeout_hit else 'timeout')
            info['assertions']={'complete_retrain_and_validation':info['status']=='OK'}
            if result:
                write_table(out/'retrain.tsv',[dict(run_id=out.name,**result)],['run_id','fingerprint','backend',
                    'status','profile','epochs_or_passes','wall_s','peak_allocated_GiB','peak_reserved_GiB','value','checkpoint_path'])
        else:
            finalize_record(out,info)
        if any(os.environ.get(k)=='1' for k in ('SREGNAS_X22_INCREMENTAL_AUDIT','SREGNAS_X22_DEFER_RECORDS')):
            from .x22.checks import assertions as framework_assertions
            info['assertions'].update(framework_assertions(out))
            if info['status']=='OK' and not all(info['assertions'].values()):info['status']='incomplete'
        if x_active():
            info['assertions']['gpu_released_before_unlock']=settled['idle']
            if settled.get('settling_decision')=='FAIL_IDLE_MEMORY_BASELINE_DRIFT':
                info.update(status='incomplete',stop_reason='tool_cleanup_idle_memory_baseline_drift')
        if quality_only:
            if info['status']=='timeout': info.update(status='incomplete',stop_reason='over_twice_estimate')
            info.update(timing_valid=False, resource_class='quality_gpu_group')
        dump(out/'run.json',info)
        passed=info['status']=='OK' and all(info['assertions'].values())
        (out/'summary.md').write_text(f"run_id: {out.name}\n\nrecord_type: {record_type}\n\nstatus: {info['status']}\n\n"
            f"ASSERTION_PASS: {str(passed).lower()}\n\n"+'\n'.join(f'- {key}: {str(value).lower()}' for key,value in info['assertions'].items())+'\n')
        for table in ['candidates','shortlist','retrain']:
            if not (out/f'{table}.tsv').exists():write_table(out/f'{table}.tsv',[],['run_id','fingerprint','status'])
        print(json.dumps(dict(run_id=out.name,status=info['status'],assertion_pass=passed,wall_s=wall,
            gpu_budget_used_hours=budget['used_gpu_s']/3600 if p0c else None)),flush=True)
        return 0 if passed else 2


def finalize_record(out,info, *, expected_search_count=None, post_search_only=False):
    events=read_rows(out/'events.jsonl');ledger=read_rows(out/'launch_ledger.jsonl')
    search=[r for r in ledger if r['stage']=='search' and r['kind'] in {'candidate','retry','resume'}]
    segments=read_rows(out/'native_segments.jsonl')
    finished_segments=[r for r in segments if r['event']=='segment_end']
    if info['method']=='FastGNAS':
        budgets=[]
        candidate_path=out/'pipeline/search_candidates.json'
        final_candidates=json.loads(candidate_path.read_text()) if candidate_path.exists() else {}
        statuses={r['original_arch_id']:r.get('status','incomplete') for r in final_candidates.values()}
        for row in search:
            parts=[s for s in finished_segments if s['parent_candidate']==row['launch_id']]
            if parts:
                row.update(start_perf_s=min(s['start_perf_s'] for s in parts),
                    end_perf_s=max(s['perf_s'] for s in parts),
                    status=statuses.get(row.get('arch_id'),'incomplete'))
                row['perf_s']=row['start_perf_s']
            budgets.append(dict(parent_candidate=row['launch_id'],fingerprint=row['fingerprint'],
                segment_count=len(parts),partitions=sorted(s['partition_id'] for s in parts),
                assigned_optimizer_steps=sum(s['expected_optimizer_steps'] for s in parts),
                actual_optimizer_steps=sum(s['optimizer_steps'] or 0 for s in parts),
                nominal_scheduler_budget=parts[0]['nominal_scheduler_budget'] if parts else None))
        info['candidate_step_budgets']=budgets
        info['assertions']['native_N_segments']=bool(budgets) and all(
            b['segment_count']==info['S'] and b['partitions']==list(range(info['S'])) for b in budgets)
        info['assertions']['native_rotation_step_budget']=bool(budgets) and all(
            b['assigned_optimizer_steps']==b['actual_optimizer_steps'] for b in budgets)
        write_table(out/'ledger_segments.tsv',finished_segments,['parent_candidate','attempt_id','segment_id',
            'fingerprint','phase','partition_id','worker','physical_gpu','rotation_step','start_perf_s','perf_s',
            'status','expected_optimizer_steps','optimizer_steps','nominal_scheduler_budget',
            'migration_bytes_in','migration_bytes_out'])
    info.update(Q_search=len(search),Q_attempt=len(search),Q_warmup=sum(r['kind']=='warmup' for r in ledger),
        Q_probe=sum(r['kind']=='probe' for r in ledger),Q_retry=sum(r['kind']=='retry' for r in ledger),
        Q_resume=sum(r['kind']=='resume' for r in ledger))
    expected = info['Q_requested'] if expected_search_count is None else expected_search_count
    info['assertions']['candidate_launch_count']=sum(r['kind']=='candidate' for r in ledger)==expected
    info['assertions']['search_attempt_budget']=len(search)==expected
    cells=[r for r in events if r['event']=='cell_result']
    retrained=[r for r in cells if r['mode']=='retrain']
    tested=[r for r in cells if r['mode']=='post_selection_test']
    winners=[r for r in events if r['event']=='winner_selected']
    good=[r for r in retrained if str(r['result']['status'])=='OK']
    for row in ledger:
        if row.get('backend')=='fastgnas-native':
            parts=[s for s in finished_segments if s['parent_candidate']==row.get('parent_candidate',row['launch_id'])]
            if parts:
                row.setdefault('start_perf_s',min(s['start_perf_s'] for s in parts))
                row.setdefault('end_perf_s',max(s['perf_s'] for s in parts))
                row.setdefault('status','OK' if all(s['status']=='OK' for s in parts) else 'incomplete')
        else:
            matches=[e for e in events if e['event']=='cell_end' and e.get('launch_id')==row['launch_id']]
            row['start_perf_s']=row['perf_s']
            row['end_perf_s']=max((e['perf_s'] for e in matches),default=None)
            result=next((c['result'] for c in cells if c['fingerprint']==row.get('fingerprint')
                         and c['mode']==row.get('mode')),None)
            row['status']=str(result['status']) if result else 'incomplete'
    from .space import canonical_fingerprint
    if winners and good:
        winner=winners[-1];info['tie_occurred']=winner['tie_occurred']
        expected=max(good,key=lambda r:(float(r['result']['value']),canonical_fingerprint(r['result'].get('architecture') or
            next(e['architecture'] for e in events if e['event']=='cell_start' and e['fingerprint']==r['fingerprint']))))
        info['assertions']['validation_winner']=winner['winner']==expected['fingerprint']
        info['winner']=winner
        info['assertions']['winner_checkpoint_test']=len(tested)==1 and tested[0]['fingerprint']==winner['winner'] and \
            tested[0]['result']['metrics'].get('checkpoint_path')==winner['checkpoint_path'] and Path(winner['checkpoint_path']).is_file()
        test_start=[e for e in events if e['event']=='cell_start' and e['mode']=='post_selection_test']
        test_end=[e for e in events if e['event']=='cell_end' and e['mode']=='post_selection_test']
        info['assertions']['checkpoint_preserved']=len(test_start)==len(test_end)==1 and bool(test_start[0]['checkpoint_before_sha256']) and \
            test_start[0]['checkpoint_before_sha256']==test_end[0]['checkpoint_after_sha256']
    else:
        info['assertions'].update(validation_winner=False,winner_checkpoint_test=False)
    info['assertions']['K_unique_successful_retrains']=len({r['fingerprint'] for r in good})==info['K']
    def first(name):return min(e['perf_s'] for e in events if e['event']==name)
    search_ids={r['launch_id'] for r in search}
    ends=[e['perf_s'] for e in events if e['event'] in {'cell_end','native_candidate_end'} and e.get('launch_id') in search_ids]
    if info['method']=='FastGNAS':
        ends=[s['perf_s'] for s in finished_segments if s['parent_candidate'] in search_ids]
    try:
        start=first('e2e_start')
        ss,se=(start,start) if post_search_only else (min(r['perf_s'] for r in search),max(ends))
        rs=first('retrain_started');ws=first('winner_selected')
        ts=min(e['perf_s'] for e in events if e['event']=='cell_start' and e['mode']=='post_selection_test')
        end=max(e['perf_s'] for e in events if e['event']=='e2e_end')
        bounds=[start,ss,se,rs,ws,ts,end]
        assert bounds==sorted(bounds)
        segments=([('nomination+probe+refill',start,rs),('retrain+validation',rs,ts),('winner-test',ts,end)]
                  if post_search_only else
                  [('prep',start,ss),('search',ss,se),('nomination+probe+refill',se,rs),
                   ('retrain+validation',rs,ws),('other',ws,ts),('winner-test',ts,end)])
        info['phases']=[dict(phase=name,start_perf_s=a,end_perf_s=b,wall_s=b-a,
                            physical_gpus=info['config']['runtime']['system_gpus']) for name,a,b in segments]
        info['cold_e2e_wall_s']=end-start
        info['phase_sum_s']=sum(p['wall_s'] for p in info['phases'])
        info['direct_e2e_wall_s']=next(e['wall_s'] for e in events if e['event']=='direct_e2e_wall')
        agreement_key='three_phase_wall_agreement' if post_search_only else 'six_phase_wall_agreement'
        info['assertions'][agreement_key]=abs(info['phase_sum_s']-info['direct_e2e_wall_s'])<=.02*info['direct_e2e_wall_s']
        info['phase_wall_discrepancy_fraction']=abs(info['phase_sum_s']-info['direct_e2e_wall_s'])/info['direct_e2e_wall_s']
        if post_search_only:
            info['direct_post_search_wall_s']=info.pop('direct_e2e_wall_s')
            info.pop('cold_e2e_wall_s',None)
    except (ValueError,AssertionError,StopIteration):
        info['assertions']['three_phase_wall_agreement' if post_search_only else 'six_phase_wall_agreement']=False
    if info['method']=='FastGNAS':
        starts=[e for e in events if e['event']=='native_training_start']
        rng=[e for e in events if e['event']=='native_rng_reset' and e['scope']=='partition_training']
        info['assertions']['native_rng_before_every_training']=bool(starts) and len(starts)==len(rng)==len(finished_segments) and all(
            any(r['segment_id']==s['segment_id'] and r['perf_s']<=s['perf_s'] for r in rng) for s in starts)
        from sregnas.training.minibatch import _hash_seed
        resets=[e for e in events if e['event']=='native_rng_reset']
        info['assertions']['native_rng_inputs_match']=bool(resets) and all(
            r['seed']==_hash_seed(*r['seed_inputs']) and r['seed_inputs']==
            ([r['fingerprint'],'search',0] if r['scope']=='initialization' else [r['fingerprint'],'search',0,r['partition_id']])
            for r in resets)
    required='run_id experiment_id method dataset environment backend commit config_hash graph_hash split_hash space_hash P S resource_class seed_rule status record_type timing_valid'.split()
    info['assertions']['required_run_fields']=all(k in info and info[k] is not None for k in required)
    for name,entries in [('candidates',search),('shortlist',good),('retrain',retrained)]:
        rows=[dict(run_id=out.name,**r) for r in entries]
        write_table(out/f'{name}.tsv',rows,['run_id','fingerprint','kind','backend','architecture','result','perf_s','launch_id'])
    write_table(out/'launch_ledger.tsv',ledger,['launch_id','parent_candidate','kind','stage','identity','fingerprint',
        'backend','perf_s','start_perf_s','end_perf_s','status'])
    if not all(info['assertions'].values()) and info['status']=='OK':info['status']='incomplete'


def record_comparison(args,cfg,original_argv):
    args_list=list(sys.argv[1:] if original_argv is None else original_argv)
    args_list.remove('--icde27-record')
    if '--allow-quality-sharing' in args_list: args_list.remove('--allow-quality-sharing')
    index=args_list.index('--out');args_list[index+1]=str(args.out.resolve()/'pipeline')
    argv=[sys.executable,'-m','sregnas.comparison',*args_list]
    return run_recorded(out=args.out,cfg=cfg,method=args.method,argv=argv,timeout=args.max_wall_seconds,
        experiment_id=args.experiment_id,record_type=args.record_type,
        gpu_lock=args.gpu_lock, quality_only=args.quality_only,
        allow_quality_sharing=getattr(args,'allow_quality_sharing',False))
