
import os

def assertions(out):
    checks={}
    if os.environ.get('SREGNAS_X22_INCREMENTAL_AUDIT')=='1':
        from .origins import assertions as source_assertions
        checks.update(source_assertions(out))
        from .receipt_cache import assertions as partition_assertions
        checks.update(partition_assertions(out))
    if os.environ.get('SREGNAS_X22_DEFER_RECORDS')=='1':
        from .deferred import assertions as writer_assertions
        checks.update(writer_assertions(out))
    return checks
