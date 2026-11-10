
import os
import hashlib
import json
from pathlib import Path


def active():
    return os.environ.get('SREGNAS_X_QUEUE_AMENDMENT')=='1'


def unchanged_preexisting_audit(root, porcelain):
    pass
    if not active():return False
    manifest=root/'docs/audit/x_preexisting_changes.json'
    if not manifest.exists():return False
    allowed=json.loads(manifest.read_text())['files']
    for line in porcelain.splitlines():
        path=line[3:]
        if line[:2]!=' M' or path!='docs/audit/exp1r_report.md' or path not in allowed:return False
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=allowed[path]:return False
    return True


def seed_allowed(seed):
    return seed==42 or (active() and seed in (43,44))


def gc_full_allowed(args):
    return (active() and args.method=='AP' and args.mode=='full_nas'
            and args.stage=='complete' and args.record_type=='full_nas'
            and not args.from_search_record and args.max_search_candidates is None
            and args.max_search_seconds is None and args.search_limit is None)


def validate(cfg, method, gcfix=False):
    from sregnas.paths import PROJECT_ROOT
    if not active():raise ValueError('X amendment flag required')
    if not (PROJECT_ROOT/'docs/audit/protocol_amendment_x.md').is_file():raise ValueError('missing amendment')
    ds=cfg.graph.dataset;seed=cfg.search.base_seed
    cells=(method=='SREGNAS' and seed in (43,44)) or (
        method=='FastGNAS' and seed in (43,44)) or (
        method in ('AP','GraphNAP') and seed in (43,44) and ds in ('flickr','ogbn-arxiv')) or (
        method=='AP' and seed==42 and ds in ('reddit','ogbn-products') and gcfix)
    if not cells or ds not in ('flickr','ogbn-arxiv','reddit','ogbn-products'):
        raise ValueError('outside authorized X timing matrix')
    if not (cfg.search.q==400 and cfg.search.m==16 and cfg.fixed_x==8 and cfg.graph.num_gpus==8
            and cfg.runtime.system_gpus==list(range(8)) and cfg.runtime.baseline_gpus==list(range(8))):
        raise ValueError('X timing requires Q400/K8/M16/S8/P8')
    if gcfix and not (method=='AP' and seed==42 and ds in ('reddit','ogbn-products')):
        raise ValueError('X gcfix is limited to AP Reddit/Products seed42')
