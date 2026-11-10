
import argparse
import json
import sys
import traceback
from pathlib import Path
from .recorder import dump


def normalization_metadata(enabled):
    from .x10 import ROOT
    from .recorder import sha
    from sregnas.models.normalization_fix import SCHEMA
    return dict(normalization_fix=bool(enabled),normalization_fix_schema=SCHEMA if enabled else 'legacy',
        normalization_fix_sha256=sha(ROOT/'src/sregnas/models/normalization_fix.py'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path);parser.add_argument('--out',type=Path)
    args=parser.parse_args();job=json.loads(args.job.read_text());out=args.out
    try:
        if job['kind']=='reference':
            from .quality import post
            from .x9_worker import dispatch
            post.dispatch=dispatch
            info=json.loads((out/'run.json').read_text())
            metadata=normalization_metadata(job['canonical_nf'])
            info.update(metadata)
            result=post.reference(job,out,info)
            result.update(metadata)
        else:
            from .x10 import configuration
            if job.get('degree'):
                from .x10_degree import install
                install(job,out)
            elif job.get('factorial') and job['dataset']=='Products':
                from .x10_factorial import install
                install(out)
            if job['method']=='SREGNAS':
                from .p2p6.canonical import execute
            elif job['method']=='FastGNAS':
                from .p2p6.native import execute
            else:
                from .p2p6.ports import execute
            result=execute(job,configuration(job),out)
        dump(out/'measurement.json',result)
        return 0 if all(result.get('assertions',{}).values()) and all(r['status']=='OK' for r in result.get('rows',[])) else 2
    except BaseException as exc:
        from .p2p6.resources import is_oom
        traceback.print_exc()
        dump(out/'failure.json',dict(status='OOM' if is_oom(str(exc)) else 'incomplete',error=str(exc),exception=type(exc).__name__))
        return 2


if __name__=='__main__':
    raise SystemExit(main())
