
import os

def enabled():
    return any(os.environ.get(k)=='1' for k in ('SREGNAS_X22_INCREMENTAL_AUDIT','SREGNAS_X22_DEFER_RECORDS'))

def finish_worker():
    if os.environ.get('SREGNAS_X22_INCREMENTAL_AUDIT')=='1':
        from .origins import finalize
        from .receipt_cache import finalize as finish_receipts
        finalize();finish_receipts()
    if os.environ.get('SREGNAS_X22_DEFER_RECORDS')=='1':
        from .deferred import flush
        flush()
    return dict(pid=os.getpid(),status='complete')

def close_workers(workers):
    if not enabled() or not workers:return []
    import ray
    return ray.get([worker.x22_finalize_framework.remote() for worker in workers])
