
from sregnas.paths import GPU_LOCK as _artifact_lock
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from sregnas.paths import PROJECT_ROOT as ROOT
from . import recorder as rec
from .p2p6.entry import terminate, final_record

QUEUE = ROOT / 'results/icde27/x10_queue'
ANCHOR = 1793017785.0
DEADLINE = ANCHOR + 9000


def save(state):
    rec.dump(QUEUE/'state.json', state)


def configuration(job):
    if job['kind'] == 'reference':
        from .quality.entry import configuration as original
        return original(job)
    from .p2p6.entry import configuration as original
    cfg = original(job)
    if job.get('degree'):
        cfg.graph.partition_strategy = 'random'
        cfg.runtime.require_fi_gpu = False
        if os.environ.get('SREGNAS_X25_DEGREE_PARTITION')=='1':
            from sregnas.icde27.x25.degree_configuration import apply
            apply(job,cfg)
    cfg.validate()
    return cfg


def record(job, out, timeout):
    pass
    from sregnas.execution.telemetry import gpu_idle_preflight, _start_ticks
    from .p2p6.cleanup import wait_for_gpu_release
    assert os.environ.get('SREGNAS_X10_LOCKED') == '1'
    out.mkdir(parents=True, exist_ok=False)
    rec.dump(out/'job.json', job)
    cfg = configuration(job)
    cmd = [sys.executable, '-m', 'sregnas.icde27.x10_worker', '--job', str(out/'job.json'), '--out', str(out)]
    info = rec.metadata(cfg, job['method'], out, 'X10', job['record_type'], cmd)
    info.update(job=job, cell=job, timing_scope='exclusive_primary', resource_class='exclusive_A6000_8_reserved',
                gpu_lock=str(_artifact_lock), global_lock_owner_pid=os.getpid(),
                reserved_gpus=list(range(8)), normalization_fix=job.get('canonical_nf', True),
                max_wall_seconds=timeout, Q_requested=32 if job['kind']=='replay' else 0,
                K=0 if job['kind']=='replay' else 1, native_rng_reset=True)
    if job.get('degree'):
        graph = json.loads(Path(job['graph_manifest']).read_text())
        info.update(graph_hash=graph['graph_sha256'], degree=job['degree'], graph_manifest=job['graph_manifest'],
                    membership_sha256=job['membership_sha256'])
    info['seed_rule']['controller'] = 'none: fixed candidates'
    if job['method']=='AP':
        info['S']='N/A'
        info['seed_rule']['canonical_search']='training.minibatch._hash_seed(fingerprint,search,0)'
    identities = {}
    observations = []
    memory = []
    before = rec.snapshot(list(range(8)), os.getpid(), identities)
    rec.dump(out/'resources_before.json', before)
    baseline = gpu_idle_preflight(list(range(8)), owner_pid=os.getpid(), verified_owned=identities)
    if not baseline['idle'] and not baseline['foreign_processes']:
        from sregnas.comparison.launch import wait_gpu_idle_preflight
        rec.dump(out/'initial_idle_preflight.json',baseline)
        baseline=wait_gpu_idle_preflight(list(range(8)),timeout_s=min(60,timeout),poll_s=2)
    rec.dump(out/'release_baseline.json', baseline)
    rec.dump(out/'run.json', info)
    if before['foreign_processes'] or before['cpu_contention_observed'] or not baseline['idle']:
        raise RuntimeError('X10 exclusive preflight failed')
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), SREGNAS_ICDE27_RECORD_DIR=str(out),
        SREGNAS_ALLOW_GPU='1', SREGNAS_APPROVED_GPUS='0,1,2,3,4,5,6,7',
        SREGNAS_DATA_DIR=os.environ.get('SREGNAS_DATA_DIR', str(ROOT / 'data')), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1', SREGNAS_ICDE27_NATIVE_RNG='1',
        SREGNAS_CANONICAL_NORMALIZATION_FIX='1' if job.get('canonical_nf', True) else '0')
    env.pop('CUDA_VISIBLE_DEVICES', None)
    started = time.time(); hit = False
    info['process_started_epoch_s'] = started
    with (out/'raw.log').open('w') as log:
        child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        info['pipeline_process'] = dict(pid=child.pid, pgid=child.pid, start_ticks=_start_ticks(child.pid), recorder_pid=os.getpid())
        rec.dump(out/'run.json', info)
        try:
            while child.poll() is None:
                observation = rec.snapshot(list(range(8)), os.getpid(), identities)
                observations.append(observation)
                with (out/'resources_during.jsonl').open('a') as f:
                    f.write(json.dumps(observation)+'\n')
                gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
                for line in gpu.stdout.splitlines():
                    g, used = line.split(',')
                    memory.append(dict(gpu=int(g), used_GiB=float(used)/1024, epoch_s=time.time()))
                if time.time()-started >= timeout:
                    hit = True; terminate(child); break
                time.sleep(.5)
        finally:
            if child.poll() is None:
                terminate(child)
    settled = wait_for_gpu_release(list(range(8)), owner_pid=os.getpid(), verified_owned=identities,
        baseline=baseline, out=out, max_idle_drift_wait_s=120)
    rec.dump(out/'resources_settled.json', settled)
    after = rec.snapshot(list(range(8)), os.getpid(), identities)
    rec.dump(out/'resources_after.json', after)
    rec.dump(out/'device_memory.json', memory)
    valid = not settled['foreign_process_observed'] and not any(o['foreign_processes'] or o['cpu_contention_observed']
        or o['gpu']['returncode'] or o['compute']['returncode'] for o in [before, *observations, after])
    info.update(status='timeout' if hit else 'OK' if child.returncode==0 else 'incomplete', returncode=child.returncode,
        timing_valid=valid, process_ended_epoch_s=time.time(), process_wall_s=time.time()-started,
        gpu_released_before_unlock=settled['idle'], worst_card_peak_gib=max((r['used_GiB'] for r in memory), default=None),
        peak_memory_definition='0.5s sampled whole-card memory including desktop baseline; maximum over 8 GPUs')
    if job['kind']=='replay':
        return final_record(out, info, job)
    if (out/'measurement.json').exists():
        info.update(json.loads((out/'measurement.json').read_text()))
    if (out/'failure.json').exists():
        info['failure']=json.loads((out/'failure.json').read_text())
        info['status']=info['failure']['status']
    info['assertions']['gpu_released_before_unlock']=settled['idle']
    info['Q_attempt']=0
    info['Qbench']=None
    info['retrain_launches']=len(rec.read_rows(out/'launch_ledger.jsonl'))
    info['checkpoint_paths']=[info.get('reference_result',{}).get('metrics',{}).get('checkpoint_path')]
    passed=info['status']=='OK' and all(info['assertions'].values())
    if not valid:
        info['status']='contaminated'; passed=False
    if info['status']=='OK' and not passed:
        info['status']='incomplete'
    for name in ('candidates','shortlist','retrain'):
        if not (out/(name+'.tsv')).exists():
            rec.write_table(out/(name+'.tsv'), [], ['run_id','fingerprint','status'])
    rec.dump(out/'run.json', info)
    (out/'summary.md').write_text(f"run_id: {out.name}\n\nrecord_type: reference\n\nstatus: {info['status']}\n\ntiming_valid: {valid}\n\nASSERTION_PASS: {str(passed).lower()}\n\n"+
        '\n'.join(f'- {k}: {v}' for k,v in info['assertions'].items())+'\n')
    return 0 if passed else 2


def prepare():
    QUEUE.mkdir(parents=True, exist_ok=False)
    (QUEUE/'jobs').mkdir()
    old=json.loads((ROOT/'results/icde27/x9_queue/state.json').read_text())
    jobs=[]
    for source in [j for j in old['jobs'] if j['stage']=='a_cost' and not j['canonical_nf']]:
        for repeat in range(1,4):
            for on in (False,True):
                job=copy.deepcopy(source)
                label=f"cost-{job['cost_category']}-r{repeat}-{'on' if on else 'off'}"
                job.update(experiment_id='X10',cell_id='X10_'+label,label=label,stage='cost',repeat=repeat,
                    quality_only=False,canonical_nf=on,run_id=f'expX10_SREGNAS-{label}_Products_E1_P1S1_Q0K1_seed42_reference_20261026')
                jobs.append(job)
    protected=list((ROOT/'results/icde27').glob('EXP*/*/run.json'))+[p for p in (ROOT/'results/icde27/tables').glob('*') if p.is_file()]
    protected += [ROOT/'docs/audit/exp1r_report.md',ROOT/'docs/PROTOCOL.md',ROOT/'docs/audit/x9_report.md']
    state=dict(anchor_epoch_s=ANCHOR,deadline_epoch_s=DEADLINE,jobs=jobs,attempts=[],stage='prepared',
        protected={str(p):rec.sha(p) for p in protected})
    save(state)


def run_job(job,state,deadline):
    if time.time()>=deadline-130:
        state.setdefault('unresolved',[]).append(dict(cell_id=job['cell_id'],reason='budget exhausted'))
        save(state); return
    out=ROOT/'results/icde27/EXPX10'/job['run_id']
    attempt=dict(cell_id=job['cell_id'],stage=job['stage'],run_id=job['run_id'],result_path=str(out),started_epoch_s=time.time(),status='running')
    state['attempts'].append(attempt); save(state)
    try:
        rc=record(job,out,min(job.get('timeout_s',1800),deadline-time.time()-125))
        info=json.loads((out/'run.json').read_text())
        attempt.update(status=info['status'],timing_valid=info['timing_valid'],returncode=rc)
    except Exception:
        attempt.update(status='incomplete',error=traceback.format_exc())
        traceback.print_exc()
    attempt['ended_epoch_s']=time.time();save(state)
    print(json.dumps(attempt),flush=True)


def run():
    assert os.environ.get('SREGNAS_X10_LOCKED')=='1'
    state=json.loads((QUEUE/'state.json').read_text());state['supervisor_pid']=os.getpid();save(state)
    try:
        state['stage']='cost';save(state)
        for job in state['jobs']:
            run_job(job,state,DEADLINE-180)
        state['stage']='waiting_degree_tools';save(state)
        while not (QUEUE/'degree_ready.json').exists() and time.time()<DEADLINE-300:
            time.sleep(2)
        if (QUEUE/'degree_ready.json').exists():
            from .x10_degree import run_degree
            run_degree(state)
    finally:
        from sregnas.execution.telemetry import gpu_idle_preflight
        state['release']=gpu_idle_preflight(list(range(8)),owner_pid=os.getpid(),verified_owned={})
        state['protected_unchanged']=all(rec.sha(p)==h for p,h in state['protected'].items())
        state.update(stage='stopped',ended_epoch_s=time.time());save(state)
        rec.dump(ROOT/'docs/audit/x10_release.json',state['release'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    prepare() if args.prepare else run()
