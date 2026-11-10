
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from .common import ROOT, OUT, PYTHON, LOCK, T0, GPU_END, read, write, now, environment

def gpu_processes():
    cards=subprocess.run(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],capture_output=True,text=True)
    if cards.returncode or len(cards.stdout.splitlines())!=8:raise RuntimeError('eight-card inventory unavailable')
    p = subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True)
    if p.returncode: raise RuntimeError(p.stderr or p.stdout)
    return list(csv.reader(p.stdout.splitlines()))

def proc_identity(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=int(pid),ppid=int(stat[1]),pgid=int(stat[2]),start_ticks=int(stat[19]))
    except (OSError,ValueError,IndexError): return None

def same_process(identity):
    current=proc_identity(identity['pid'])
    return current is not None and current['start_ticks']==identity['start_ticks']

def owned_tree(pid, known):

    entries=[r for p in Path('/proc').iterdir() if p.name.isdigit() and (r:=proc_identity(p.name))]
    owned=({pid} if pid not in known or same_process(known[pid]) else set())|{p for p,r in known.items() if same_process(r)}
    for _ in range(12):
        extra={r['pid'] for r in entries if r['ppid'] in owned}
        if extra <= owned: break
        owned |= extra
    for r in entries:
        if r['pid'] in owned: known[r['pid']]=r

def stop_owned(proc,known):
    owned_tree(proc.pid,known)
    def owned_groups():
        result=set()
        for pid,identity in known.items():
            current=proc_identity(pid)
            if current is not None and current['start_ticks']==identity['start_ticks']:result.add(current['pgid'])
        return result
    groups=owned_groups()
    assert os.getpgrp() not in groups
    for sig in (signal.SIGTERM,signal.SIGKILL):
        for group in groups:
            if group in owned_groups():
                try: os.killpg(group,sig)
                except ProcessLookupError: pass
        if sig==signal.SIGTERM: time.sleep(5)
    try: proc.wait(timeout=10)
    except subprocess.TimeoutExpired: pass

def append(row):
    with (OUT/'queue_status.jsonl').open('a') as f:
        f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())

def progress(state,finished=None):
    completed=len(state['cells']);planned=max(completed,state.get('planned',completed))
    fraction=completed/planned if planned else 0;elapsed=(time.time()-T0)/3600
    active=state.get('active') or {};next_text=active.get('cell_name','等待下一个已提交队列文件')
    text=f'# X23 进度\n\n{"█"*round(20*fraction)}{"░"*(20-round(20*fraction))}  {completed}/{planned}（{100*fraction:.0f}%）  已用 {elapsed:.2f} h / 9 h\n\n'
    text+=f'更新时间：{now()}\n\nGPU 截止：2026-11-05T10:48:03Z；最终截止：2026-11-05T11:18:03Z。\n\n'
    if finished: text+=f'刚完成：{finished["cell_name"]}；{finished["status"]}；用时 {finished.get("wall_s",0):.2f} s；{finished.get("reason", "详见 run.json")}。\n\n'
    text+=f'下一项/当前项：{next_text}；预计结束：{active.get("expected_end", "按队列与剩余必做预算计算")}。\n\n'
    if (OUT/'early_readout.md').exists():text+=(OUT/'early_readout.md').read_text()+'\n'
    if (OUT/'cpu_progress.md').exists():text+=(OUT/'cpu_progress.md').read_text()+'\n'
    (OUT/'progress.md.tmp').write_text(text);(OUT/'progress.md.tmp').replace(OUT/'progress.md')

def queued(state):
    formal=(OUT/'formal_submitted.json').exists()
    if formal and not state.get('fallback_cancelled'):
        state['fallback_cancelled']=now();write(OUT/'fallback_cancelled.json',dict(at=now(),reason='formal queue submitted'))
    if time.time()>=T0+4*3600 and not formal and not (OUT/'queue/900_fallback_main.json').exists():
        write(OUT/'queue/900_fallback_main.json',read(OUT/'fallback_queue.json'))
        state['fallback_activated']=now()
    result=[]
    for path in sorted((OUT/'queue').glob('*.json')):
        cells=read(path)
        for c in cells:
            if c['cell_name'] in state['cells']: continue
            if c.get('fallback') and formal:continue
            result.append((path,c))
    state['planned']=len(state['cells'])+len(result)
    return result

def reserve_seconds(pending,current):
    reserved=read(OUT/'reservations.json') if (OUT/'reservations.json').exists() else {}

    submitted={c.get('phase') for p in (OUT/'queue').glob('*.json') for c in read(p)}
    reserve=sum(v for k,v in reserved.items() if k not in submitted)
    reserve+=sum(c['expected_s'] for _,c in pending if c is not current and c.get('mandatory',True))
    return reserve

def admit(c,reserve):
    if time.time()>=GPU_END:return 'gpu_deadline'
    if c.get('launch_deadline') and time.time()>=c['launch_deadline']:return 'development_qualification_deadline'
    if time.time()+c['expected_s']+reserve+1800>GPU_END:
        if not c.get('mandatory',True):return 'optional_budget_skip'

        if time.time()+c['expected_s']+1800>GPU_END:return 'mandatory_cannot_fit'
    return None

def reduce_phase(pending,state):
    pass
    if not pending:return pending
    current=pending[0][1];phase=current['phase']
    groups=[]
    cells=[c for _,c in pending if c['phase']==phase]
    if phase=='unused':groups=[]
    elif phase=='1.4':groups=[[c for c in cells if c.get('reduction_rank')==1]]
    elif phase=='2.1':groups=[[c for c in cells if c.get('replicate')==2]]
    elif phase=='2.2':groups=[[c for c in cells if c.get('reduction_rank')==rank] for rank in (3,2,1)]
    elif phase=='3.main':groups=[[c for c in cells if c.get('seed')==seed] for seed in (44,43)]
    for group in groups:
        if time.time()+current['expected_s']+reserve_seconds(pending,current)+1800<=GPU_END:break
        for c in group:
            row=dict(cell_name=c['cell_name'],phase=phase,started_at=now(),ended_at=now(),status='skipped',
                reason='preregistered_group_budget_reduction',run_id=c.get('run_id'),wall_s=0)
            append(row);state['cells'][c['cell_name']]=row
        pending=[(p,c) for p,c in pending if c['cell_name'] not in state['cells']]
        if not pending:break
        current=pending[0][1]
    write(OUT/'driver_state.json',state)
    return pending

def run_cell(c,state,reserve):
    start=time.time();row=dict(cell_name=c['cell_name'],phase=c['phase'],started_at=now(),
        started_epoch_s=start,run_id=c.get('run_id'),result_path=c.get('out'),expected_s=c['expected_s'],
        remaining_mandatory_s=reserve,status='running')
    reason=admit(c,reserve)
    if reason:row.update(status='skipped',reason=reason)
    else:
        wait_started=time.time()
        while True:
            try: external=gpu_processes()
            except Exception as exc:row.update(status='skipped',reason='resource_check_failed: '+str(exc));break
            if not external:break
            if time.time()-wait_started>=1800 or time.time()>=GPU_END:
                row.update(status='skipped',reason='external_gpu_processes_after_30min',external=external);break
            write(OUT/'external_wait.json',dict(at=now(),cell=c['cell_name'],external=external))
            time.sleep(min(30,max(0,GPU_END-time.time())))
        if row['status']=='running':
            reason=admit(c,reserve)
            if reason:row.update(status='skipped',reason=reason)
        if row['status']=='running':
            code=Path(c['code']);env=environment(code,c.get('variant'),c.get('dataset'),c.get('flags'))
            argv=list(c['argv']);timeout=max(2*c['expected_s'],c['expected_s']+600)
            timeout=min(timeout,max(1,GPU_END-time.time()-15))
            if c.get('workspace'):env['CUBLAS_WORKSPACE_CONFIG']=c['workspace']
            if c.get('legacy_x18'):
                from sregnas.icde27.x18.common import environment as old_env

                env=read(c['legacy_env'])
            if not c.get('internal_lock',True):argv=['flock',str(LOCK),*argv]
            row.update(command=argv,code=str(code),timeout_s=timeout,flags=c.get('flags',{}),
                numeric_environment={k:env.get(k) for k in ('CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})
            (OUT/'logs').mkdir(exist_ok=True);log_path=OUT/'logs'/(c['cell_name']+'.log')
            row['log']=str(log_path)
            if log_path.exists():raise FileExistsError(log_path)
            with log_path.open('x') as log:
                proc=subprocess.Popen(argv,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                known={};row['pid']=proc.pid
                state['active']=dict(row,expected_end=__import__('datetime').datetime.fromtimestamp(time.time()+c['expected_s'],__import__('datetime').timezone.utc).isoformat())
                write(OUT/'driver_state.json',state);progress(state)
                deadline=time.time()+timeout;hit=False
                while proc.poll() is None:
                    owned_tree(proc.pid,known)
                    if time.time()>=deadline or time.time()>=GPU_END-10:
                        hit=True;stop_owned(proc,known);break
                    time.sleep(1)
                info=read(Path(c['out'])/'run.json') if c.get('out') and (Path(c['out'])/'run.json').exists() else {}
                row.update(status='timeout' if hit else info.get('status','OK' if proc.returncode==0 else 'incomplete'),
                    returncode=proc.returncode,timing_valid=info.get('timing_valid',False),Q_attempt=info.get('Q_attempt',0),
                    owned_processes=list(known.values()))

                alive={p:r for p,r in known.items() if same_process(r)}
                if alive:stop_owned(proc,alive);row['cleanup_owned_remaining']=list(alive)
            if c.get('out') and Path(c['out']).exists():write(Path(c['out'])/'x23_options.json',row)
    row.update(ended_at=now(),ended_epoch_s=time.time(),wall_s=time.time()-start)
    append(row);state['cells'][c['cell_name']]=row;state['active']=None
    write(OUT/'driver_state.json',state);progress(state,row)
    return row

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'queue').mkdir(exist_ok=True)
    with (OUT/'driver.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(OUT/'driver_state.json') if (OUT/'driver_state.json').exists() else dict(started_at=now(),cells={},active=None)
        write(OUT/'driver_identity.json',dict(pid=os.getpid(),identity=proc_identity(os.getpid()),at=now()))
        while time.time()<GPU_END:
            pending=reduce_phase(queued(state),state)
            if not pending:
                if (OUT/'FINISH').exists():break
                write(OUT/'driver_state.json',state);time.sleep(5);continue
            path,c=pending[0]
            try:row=run_cell(c,state,reserve_seconds(pending,c))
            except Exception as exc:
                traceback.print_exc();row=dict(cell_name=c['cell_name'],phase=c['phase'],started_at=now(),ended_at=now(),
                    status='driver_error',reason=repr(exc),run_id=c.get('run_id'))
                append(row);state['cells'][c['cell_name']]=row;write(OUT/'driver_state.json',state)

            if row.get('returncode',0)!=0 and row.get('Q_attempt',0)==0 and not c.get('retry_of'):
                log=Path(row.get('log','/nonexistent'))
                tail=log.read_text()[-12000:] if log.exists() else ''
                if any(t in tail for t in ('Address already in use','temporarily unavailable','Connection refused')):
                    retry=dict(c,cell_name=c['cell_name']+'_retry1',run_id=c.get('run_id','')+'_retry1',retry_of=c['cell_name'])
                    retry['out']=c['out']+'_retry1';retry['argv']=[retry['out'] if v==c['out'] else v for v in c['argv']]
                    run_cell(retry,state,reserve_seconds(queued(state),retry))
            if all(c['cell_name'] in state['cells'] for c in read(path)):
                from .summary import summarize
                summarize();write(OUT/'queue_done'/path.name,dict(at=now(),queue=str(path)))
        from .summary import summarize
        for path,c in queued(state):
            row=dict(cell_name=c['cell_name'],phase=c['phase'],started_at=now(),ended_at=now(),status='skipped',
                reason='campaign_GPU_deadline',run_id=c.get('run_id'),wall_s=0)
            append(row);state['cells'][c['cell_name']]=row
        summarize()
        try:
            external=gpu_processes()
            with LOCK.open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);free=True
                except BlockingIOError:free=False
            write(OUT/'gpu_release.json',dict(at=now(),processes=external,global_lock_available=free,driver_pid=os.getpid()))
        finally:state.update(ended_at=now(),active=None);write(OUT/'driver_state.json',state);progress(state)

if __name__=='__main__':main()
