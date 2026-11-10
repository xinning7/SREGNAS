
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from sregnas.icde27 import ledger
from sregnas.icde27.recorder import dump
from sregnas.paths import PROJECT_ROOT as ROOT
from .plan import selected_rows, digest
from .profiling import capture, worker_tail


def tensor_hash(tensors):
    import hashlib
    h = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        h.update(str(array.shape).encode()); h.update(str(array.dtype).encode()); h.update(array.tobytes())
    return h.hexdigest()


def prepare(cell, cfg, out):
    import torch
    if cell["kind"] != "replay":
        from sregnas.paths import DATA_DIR
        if cfg.graph.dataset in ("reddit", "ogbn-products"):
            path = DATA_DIR / ("Reddit/sregnas_cache_v1/full/manifest.json" if cfg.graph.dataset == "reddit"
                               else "ogbn_products/sregnas_comparison_v1/full/manifest.json")
            if not path.exists(): raise FileNotFoundError(path)

            from sregnas.data.products import ProductsMmapGraph
            ProductsMmapGraph.open(path)
        else:
            from sregnas.data import load_dataset
            path = out/"full_graph.pt"
            torch.save(load_dataset(cfg.graph.dataset, data_dir=str(DATA_DIR), normalize=False, preprocess=False), path)
        return [str(path)]*8, None
    from sregnas.data.graph import build_graph_stage, build_uniform_sampled_full_graph
    if cell["domain"] == "local":
        manifest = build_graph_stage(cfg, out/"prep")
        from sregnas.execution.startup import partition_receipt
        cfg._partition_receipt = partition_receipt(out/"prep/graph/manifest.json")
        paths = [r["path"] for r in manifest["shards"]]
    else:
        assert (cfg.graph.fanout_a, cfg.graph.fanout_b) == (5, 5)
        manifest = build_uniform_sampled_full_graph(cfg.graph.dataset, out/"prep")
        paths = [manifest["payload_path"]]*8
    rebuild = None
    if cell["experiment_id"] == "4" or cell.get("factorial"):
        import numpy as np
        from sregnas.data.graph import _load_dataset
        from sregnas.data import sampling
        raw = _load_dataset(cfg.graph.dataset)
        rowptr, col, _ = sampling.build_csr(raw["edge_index"].cpu().numpy(), int(raw["num_nodes"]))
        membership = np.load(out/"prep/graph/membership.npy") if cell["domain"] == "local" else np.zeros(raw["num_nodes"], dtype=np.int64)
        directory = out/"rebuild_inputs"; directory.mkdir()
        for name, array in (("rowptr", rowptr), ("col", col), ("membership", membership)):
            np.save(directory/(name+".npy"), array)
        rebuild = str(directory)
    return paths, rebuild


def execute(cell, cfg, out):
    from sregnas.core.contracts import TaskKey
    from sregnas.execution.tasks import _cell_spec
    from .resources import mark_stage
    ledger.start_clock("P2P6 before reading local graph")
    mark_stage(out, "prep/graph_and_neighborhood")
    paths, rebuild = prepare(cell, cfg, out)
    selected = selected_rows(cell)
    groups = {p: [] for p in range(cell["P"])}
    neighborhoods = {}
    if cell["kind"] == "replay":
        import torch
        from sregnas.icde27.x18.cold_prep import enabled as memory_prep_enabled
        from sregnas.training.fullbatch import load_graph_payload
        for path in set(paths):
            if memory_prep_enabled():
                neighborhoods[path]=None
            else:
                payload = load_graph_payload(path)
                neighborhoods[path] = tensor_hash([payload["edge_index_layer1"], payload["edge_index_layer2"]])
    for index, row in enumerate(selected):
        frozen_index = cell.get("pool_indices", list(range(len(selected))))[index]
        shard = frozen_index % 8 if cell["kind"] == "replay" else 0
        gpu = frozen_index % cell["P"]
        mode = "proxy" if cell["kind"] == "replay" else "retrain"
        native = cell["method"] == "FastGNAS"
        architecture = json.loads(row["native_architecture_json"] if native else row["architecture_json"])
        spec = _cell_spec(key=TaskKey("P2P6:system:"+mode, index//8, shard, row["fingerprint"]),
            mode=mode, shard=shard, architecture=architecture, graph_path=paths[shard], cfg=cfg, physical_gpu=gpu)
        spec["checkpoint_path"] = str(out/"checkpoints"/(row["fingerprint"]+".pt"))
        if native: spec["dropout"] = .6
        if cell["domain"] == "global": spec.update(partition_strategy="none", require_fi_gpu=False)
        spec.update(p2p6_index=index, p2p6_rebuild=rebuild if cell["cache_mode"] == "rebuild" else None,
                    p2p6_neighborhood_hash=neighborhoods.get(paths[shard]))
        groups[gpu].append(spec)
    dump(out/"specs.json", groups)
    def worker(gpu, specs):
        job = out/f"worker_{gpu}.job.json"
        output = out/f"worker_{gpu}.json"
        dump(job, dict(cell=cell, gpu=gpu, specs=specs, output=str(output)))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        with (out/f"worker_{gpu}.log").open("w") as log:
            process = subprocess.run([sys.executable, "-m", "sregnas.icde27.p2p6.canonical", "--job", str(job)],
                env=env, stdout=log, stderr=subprocess.STDOUT)
        if not output.exists(): raise RuntimeError(f"worker {gpu} failed before result: exit {process.returncode}")
        result = json.loads(output.read_text())
        if process.returncode: result["process_failed"] = True
        return result
    from sregnas.icde27.x23.pool import active as x23_pool_active
    if x23_pool_active():
        from sregnas.icde27.x23.pool import execute as ready_pool
        output, started, ended = ready_pool(out, cell, groups)
    else:
        mark_stage(out, "pool/proxy" if cell["kind"] == "replay" else "pool/retrain_and_validation")
        ledger.event("pool_start")
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=cell["P"]) as executor:
            futures = [executor.submit(worker, gpu, specs) for gpu, specs in groups.items() if specs]
            output = [future.result() for future in futures]
        ended = time.perf_counter()
        ledger.event("pool_end")
        ledger.stop_clock()
    rows = sorted([r for w in output for r in w["rows"]], key=lambda r: r["index"])
    spans = [s for w in output for s in w["spans"]]
    dump(out/"worker_tails.json", worker_tail(spans, started, ended))
    if cell["kind"] == "replay":
        dump(out/"logical_worker_tails.json",worker_tail([dict(s,worker=s["logical_worker"]) for s in spans],started,ended))
    communication=[]
    for gpu, specs in groups.items():
        if specs:
            communication.extend([dict(category="candidate_dispatch", worker=gpu, bytes=(out/f"worker_{gpu}.job.json").stat().st_size),
                dict(category="metric_return", worker=gpu, bytes=(out/f"worker_{gpu}.json").stat().st_size)])
    dump(out/"communication.json", dict(definition="actual serialized local IPC file bytes; no physical network inference", records=communication))
    neighborhoods = [r.get("neighborhood_hash") for r in rows if r.get("neighborhood_hash")]
    return dict(rows=rows, controller_updates=0,
        cell_s=ended-started if cell["kind"] == "cell" else None,
        makespan_s=ended-started if cell["kind"] == "retrain_pool" else None,
        neighborhood_id=digest(neighborhoods), neighborhood_policy="offline_seed42_per_layer_5_5",
        assertions=dict(rebuild_structure_identical=all(w["structure_assertions"] for w in output),
            worker_processes_complete=not any(w.get("process_failed") for w in output)))


def run_worker(job):
    if job.get('cell',{}).get('x28_fixed_pool'):
        from sregnas.icde27.x28.ablation import run_worker as x28_worker
        return x28_worker(job)
    if os.environ.get("SREGNAS_X27_MEMORY_AUDIT") == "1":
        from sregnas.icde27.x27.b2 import install_worker_observer
        install_worker_observer()
    import torch
    from sregnas.training.fullbatch import PersistentCellRuntime, run_cell
    from sregnas.training.minibatch import ProductsRuntimeCache, run_products_cell
    from sregnas.comparison.retrain import native_cell
    from sregnas.execution.startup import assert_cell_start
    cell, gpu = job["cell"], job["gpu"]
    torch.set_num_threads(int(os.environ.get('SREGNAS_CPU_THREADS', '4')))
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    full = PersistentCellRuntime()
    sampled = ProductsRuntimeCache()
    payloads = {}
    result = dict(rows=[], spans=[], structure_assertions=True)
    from sregnas.icde27.x23.pool import worker_ready, worker_done, active as x23_pool_active
    worker_ready(job)
    for spec in job["specs"]:
        started = time.perf_counter()
        assert_cell_start(spec)
        neighborhood_hash = None
        try:
            with capture(f"worker{gpu}_{spec['fingerprint']}_{spec['mode']}"):
                if spec["mode"] == "proxy":
                    data, _, _ = full.load_payload(spec["graph_path"], "proxy")
                    neighborhood_hash = spec["p2p6_neighborhood_hash"]
                    if spec.get("p2p6_rebuild"):
                        import numpy as np
                        from sregnas.data.sampling import build_layer_perms
                        from sregnas.data.graph import _per_layer_capped_edges
                        directory = Path(spec["p2p6_rebuild"])
                        rowptr, col, membership = (np.load(directory/(n+".npy"), mmap_mode="r") for n in ("rowptr", "col", "membership"))
                        perms = build_layer_perms(rowptr, col, 42, num_layers=2)
                        part = spec["shard"] if cell["domain"] == "local" else 0
                        edges = [_per_layer_capped_edges(rowptr, perm, membership, part, 5)[0] for perm in perms]
                        assert tensor_hash(edges) == neighborhood_hash, "rebuild changed frozen neighborhood"
                        data["edge_index_layer1"], data["edge_index_layer2"] = [e.to(device) for e in edges]
                    if os.environ.get('SREGNAS_X18_COST_PROFILE') in ('sync', 'control', 'trace'):
                        from sregnas.icde27.x18.cost_hooks import shard_call
                        def observed_cell(spec, runtime):
                            return shard_call(spec, run_cell, runtime)
                        row = asdict(ledger.call(spec, observed_cell, runtime=full))
                    else:
                        row = asdict(ledger.call(spec, run_cell, runtime=full))
                elif cell["method"] == "FastGNAS":
                    row = ledger.call(spec, native_cell, device, backend="fastgnas-native", runtime=sampled, payloads=payloads)
                elif spec["dataset"] in ("reddit", "ogbn-products"):
                    row = asdict(ledger.call(spec, run_products_cell, device=device, runtime=sampled))
                else:
                    row = asdict(ledger.call(spec, run_cell, runtime=full))
                torch.cuda.synchronize()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            from .resources import is_oom
            row = dict(status="OOM" if is_oom(repr(exc)) else "incomplete", error=repr(exc), failure_stage="pool/"+spec["mode"])
            if isinstance(exc, AssertionError): result["structure_assertions"] = False
        ended = time.perf_counter()
        row.update(fingerprint=spec["fingerprint"], index=spec["p2p6_index"], worker=gpu, gpu=gpu,
            wall_s=ended-started, checkpoint_path=spec["checkpoint_path"] if spec["mode"] == "retrain" else None,
            neighborhood_hash=neighborhood_hash, epochs=spec["epochs"],
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3)
        result["rows"].append(row)
        result["spans"].append(dict(worker=gpu, logical_worker=spec["shard"], start_perf_s=started, end_perf_s=ended))
        if not x23_pool_active(): dump(Path(job["output"]), result)
        if row["status"] not in ("OK", "OOM", "DIVERGE"): return worker_done(job, result, 2)
    return worker_done(job, result, 0)


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--job", type=Path, required=True)
    raise SystemExit(run_worker(json.loads(p.parse_args().job.read_text())))
