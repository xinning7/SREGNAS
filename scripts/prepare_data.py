
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
from artifact import ROOT, environment, read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data")
    p.add_argument("--dataset", choices=["Flickr", "arXiv", "Reddit", "Products", "all"], default="all")
    p.add_argument("--degree", action="store_true")
    p.add_argument("--cache-group", choices=["x25", "x27", "x28"], default="x25")
    args = p.parse_args()
    data = args.data.expanduser().resolve()
    env = environment({}, data)
    env.update(CUDA_VISIBLE_DEVICES="", SREGNAS_ALLOW_GPU="0")
    env["NUMBA_CACHE_DIR"] = str(ROOT / "results/icde27" / args.cache_group / "numba_cache")
    os.environ.update(env)
    from sregnas.paths import GPU_LOCK
    from sregnas.icde27.x22.canonical_input import KEYS, build_one, manifest_path, raw_dataset
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with GPU_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        runtime_digests = ROOT / "results/icde27/x18/v2/csr_validation.json"
        if not runtime_digests.exists():
            runtime_digests.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / "configs/data/csr_validation.json", runtime_digests)
        import numpy as np
        import scipy.sparse as sp
        import torch
        from sregnas.data import proxy_partition as pp
        torch.set_num_threads(1)
        records = []
        for label in KEYS if args.dataset == "all" else [args.dataset]:
            if label == "Reddit":
                from sregnas.data.reddit import prepare_reddit_graph
                prepare_reddit_graph()
            if label == "Products":
                from sregnas.data.products_comparison import prepare_products_graph
                prepare_products_graph()
            path = manifest_path(KEYS[label])
            if not path.exists():
                build_one(label)
            meta = read(path)
            assert meta["x22_bitwise_validated"] is True

            from sregnas.data.products import ProductsMmapGraph
            graph = ProductsMmapGraph.open(path)
            for name, expected in meta["array_sha256"].items():
                array = np.load(path.parent / meta["arrays"][name]["path"], mmap_mode="r")
                assert hashlib.sha256(memoryview(array).cast("B")).hexdigest() == expected, name
                del array
            raw = raw_dataset(KEYS[label])
            for name in ("x", "labels", "train_mask", "val_mask", "test_mask", "rowptr", "col"):
                assert hashlib.sha256(memoryview(raw[name]).cast("B")).hexdigest() == meta["array_sha256"][name], name
            y, split = pp.masked_labels(raw["labels"], raw["train_mask"], raw["val_mask"])
            A = sp.csr_matrix((np.ones(len(raw["col"]), dtype=np.int8), raw["col"], raw["rowptr"]), shape=(len(y), len(y)))
            memberships = {}

            for arm in ("da_random", "la_metis"):
                part = getattr(pp, arm)(A, y, split, 8, seed=42)
                pp.fi_metrics(A, y, split, part, 8, 5)
                memberships[arm] = hashlib.sha256(part.astype("int64").tobytes()).hexdigest()
            choices, selection = pp.select_partition(A, y, split, 8, seed=42)
            records.append(dict(dataset=label, canonical=str(path), selected=selection["selected"], membership_sha256=memberships))
            del raw, graph, y, split, A, part, choices
        if args.degree:
            from sregnas.icde27 import x10_degree

            if data != ROOT / "data":
                p.error("--degree currently requires --data equal to the artifact data directory")
            (ROOT / "results/icde27/x10_queue").mkdir(parents=True, exist_ok=True)
            x10_degree.prepare_graphs()
    if args.degree:

        from sregnas.icde27.x18.degree import inputs
        inputs()
    target = ROOT / "outputs" / f"data_preparation_{args.cache_group}_{args.dataset}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(records, indent=2) + "\n")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
