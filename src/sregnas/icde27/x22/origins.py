
import atexit
import json
import os
from pathlib import Path
import sys
import threading

_cache = {}
_origins = {}
_lock = threading.RLock()
_registered = False
_calls = 0
_resolved = 0

def enabled():return os.environ.get('SREGNAS_X22_INCREMENTAL_AUDIT')=='1'

def receipt(status, **fields):
    root=os.environ.get('SREGNAS_ICDE27_RECORD_DIR')
    if not root:return
    path=Path(root)/'x22_framework'/f'origins_{os.getpid()}.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(pid=os.getpid(),status=status,calls=_calls,
        modules_resolved=_resolved,**fields),sort_keys=True)+'\n');tmp.replace(path)

def audit(*, full=False):
    global _registered,_calls,_resolved
    from sregnas.paths import SOURCE_ROOT,PACKAGE_ROOT
    from sregnas.execution.startup import _PROJECT_NAMES
    from sregnas.icde27.x20.origins import resolver
    root=SOURCE_ROOT or PACKAGE_ROOT
    with _lock:
        if not _registered:
            _registered=True;atexit.register(finalize);receipt('started');full=True
        _calls+=1
        resolve=resolver();rejected={};present=set()
        for name,module in tuple(sys.modules.items()):
            if module is None:continue
            attributes=vars(module)
            raw=(attributes.get('__file__'),getattr(attributes.get('__spec__'),'origin',None),
                 tuple(attributes.get('__path__',())))
            signature=(id(module),raw);present.add(name)
            if not full and _cache.get(name)==signature:continue
            paths={resolve(v) for v in raw[:2] if v and v not in {'built-in','frozen'}}
            paths.update(resolve(v) for v in raw[2]);_resolved+=1
            project=name.split('.')[0].lower() in _PROJECT_NAMES or any(
                p.is_relative_to(root) or any(part.lower() in {'distnas','sregnas','fastgnas'} or
                    part.lower().startswith(('distnas-','fastgnas_snapshot')) for part in p.parts)
                and 'site-packages' not in p.parts and '.venv' not in p.parts for p in paths)
            _cache[name]=signature
            if project:
                _origins[name]=sorted(map(str,paths))
                if not paths or any(not p.is_relative_to(root) for p in paths):rejected[name]=_origins[name]
            else:_origins.pop(name,None)
        for name in (set(_cache)|set(_origins))-present:_cache.pop(name,None);_origins.pop(name,None)
        if rejected:

            for name in rejected:_cache.pop(name,None)
            receipt('failed',rejected=rejected)
            raise AssertionError('project modules outside SREGNAS: '+json.dumps(rejected,sort_keys=True))
        return dict(root=str(root),loaded_modules=dict(_origins),assertion='PASS')

def finalize():
    if not _registered:return
    try:
        value=audit(full=True);receipt('complete',root=value['root'],loaded_modules=value['loaded_modules'],assertion='PASS')
    except BaseException as exc:
        receipt('failed',error=repr(exc));raise

def assertions(out):
    records=[json.loads(p.read_text()) for p in (Path(out)/'x22_framework').glob('origins_*.json')]
    return {'x22_complete_origin_audits':all(r.get('status')=='complete' and r.get('assertion')=='PASS' for r in records)}
