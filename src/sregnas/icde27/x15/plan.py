
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex

PYTHON = __import__("sys").executable
AUTHORIZATION = "X15 阶段 2 开始：服务器已独占"
DATASETS = ("Flickr", "arXiv", "Reddit", "Products")
ORIGINAL_SECONDS = dict(zip(DATASETS, (128, 169, 215, 767)))
SUMMARY = "results/icde27/EXP1/EXP1_summary.tsv"
SEEDS = "results/icde27/tables/exp1_seeds.csv"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def source_rows(root):
    with (root / SUMMARY).open() as stream:
        return [r for r in csv.DictReader(stream, delimiter="\t")
                if r["method"] in {"AP", "GraphNAP", "FastGNAS", "SREGNAS"}
                and r["status"] == "OK" and r["record_type"] == "full_nas"
                and r["timing_valid"].lower() == "true" and "_seed42_" in r["run_id"]
                and not r["run_id"].endswith(("_nf", "_nfc"))]


def source_for(root, dataset, method):
    rows = [r for r in source_rows(root) if r["dataset"] == dataset and r["method"] == method
            and (method != "SREGNAS" or "SREGNAS-s0_" in r["run_id"])]
    if len(rows) != 1:
        raise ValueError(f"expected one frozen source: {dataset} {method}; found {len(rows)}")
    return root / "results/icde27/EXP1" / rows[0]["run_id"]


def build_plan(root, work):
    pass
    cells = []

    def add(step, kind, dataset, seed=42, seconds=300, *, replica=0, optional=False):
        method = "SREGNAS" if kind == "control" else "PAS-PL"
        cell_id = f"{kind}_{dataset}_s{seed}" + (f"_r{replica}" if replica else "")
        cell = dict(cell_id=cell_id, step=step, kind=kind, method=method, dataset=dataset,
                    seed=seed, replica=replica, expected_seconds=seconds, optional=optional,
                    timed=kind in ("nas", "control"), Q=16 if kind in ("smoke", "precheck") else
                    0 if kind == "replica" else 400, K=2 if kind == "smoke" else
                    0 if kind == "precheck" else 1 if kind == "replica" else 8,
                    timeout_seconds={"smoke": 600, "precheck": 600, "replica": 600}.get(
                        kind, 6000 if dataset == "Reddit" else 6500 if dataset == "Products" else 900))
        if kind == "nas" and dataset in ("Reddit", "Products"):
            cell["requires"] = f"precheck_{dataset}_s42"
        if kind == "replica":
            cell["requires"] = f"nas_{dataset}_s42"
            cell["gpu"] = DATASETS.index(dataset) * 2 + replica - 1
        source = source_for(root, dataset, "AP" if method == "PAS-PL" else method)
        info = read(source / "run.json")
        cfg = deepcopy(info["config"])
        cfg["search"]["base_seed"] = seed
        cfg["search"]["q"] = cell["Q"] or 400


        cfg["fixed_x"] = 1 if kind == "replica" else 8
        if kind == "replica":
            cfg["graph"]["num_gpus"] = 1
            cfg["runtime"].update(system_gpus=[cell["gpu"]], baseline_gpus=[cell["gpu"]],
                                   require_fi_gpu=False)
            cfg["train"]["retrain_replica"] = replica
        cell.update(config=str(work / "configs" / (cell_id + ".json")), config_payload=cfg,
                    source_run_id=info["run_id"],
                    source_run_sha256=hashlib.sha256((source / "run.json").read_bytes()).hexdigest())
        cells.append(cell)

    add("2.0", "smoke", "Flickr", seconds=300)
    for ds in DATASETS[:2]:
        add("2.1", "nas", ds, seconds=450)
    for seed in (43, 44):
        for ds in DATASETS[:2]:
            add("2.2", "nas", ds, seed, seconds=450)
    add("2.3", "precheck", "Reddit", seconds=480)
    add("2.4", "nas", "Reddit", seconds=5040)
    add("2.5", "precheck", "Products", seconds=480)
    add("2.6", "nas", "Products", seconds=5400)
    for ds in DATASETS:
        for replica in (1, 2):
            add("2.7", "replica", ds, seconds=600, replica=replica)
    for ds in DATASETS[:2]:
        add("2.8", "control", ds, seconds=300, optional=True)
    return dict(schema="x15_v1", prepared_at=now(), root=str(root), work=str(work),
                phase2_budget_seconds=18000, cells=cells,
                replica_schedule="eight independent single-GPU jobs concurrently under the global lock",
                final_protocol=dict(NF=1, sampled_grad_clip=1.0, executor="G3"))


def run_path(cell, work, attempt=1, stamp=None):
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%d")
    kind = "reference" if cell["kind"] == "replica" else "full_nas" if cell["timed"] else "profile"
    p = 1 if cell["kind"] == "replica" else 8
    label = cell["method"] + "-" + cell["kind"] + (str(cell["replica"]) if cell["replica"] else "")
    return work / "runs" / (f"expX15_{label}_{cell['dataset']}_E1_P{p}S{p}_Q{cell['Q']}K{cell['K']}"
                             f"_seed{cell['seed']}_{kind}_{stamp}_a{attempt}")


def command(cell, root, work, out, *, timeout=None):
    timeout = max(1, int(timeout if timeout is not None else cell["timeout_seconds"]))
    if cell["kind"] == "replica":
        return [PYTHON, "-m", "sregnas.icde27.x15.replica", "--work", str(work),
                "--cell", cell["cell_id"], "--out", str(out), "--max-wall-seconds", str(timeout),
                "--gpu-lock", str(work / "locks" / f"gpu{cell['gpu']}.lock")]
    argv = [PYTHON, "-m", "sregnas.comparison", "--method", cell["method"],
            "--config", cell["config"], "--hardware", str(root / "configs/hardware/a6000.yaml"),
            "--data-dir", str(root / "data"), "--out", str(out), "--allow-gpu", "--icde27-record",
            "--canonical-winner-tie", "--experiment-id", "X15", "--record-type",
            "full_nas" if cell["timed"] else "profile", "--max-wall-seconds", str(timeout)]
    if cell["method"] == "PAS-PL":
        argv += ["--paspl-full-nas", "--stage", "search" if cell["kind"] == "precheck" else "complete"]
        if cell["dataset"] in ("Reddit", "Products"):
            argv += ["--paspl-collect-search-garbage"]
    else:
        argv += ["--s0-parallel"]
    if cell["kind"] == "smoke":
        argv += ["--shortlist-size", "2"]
    if not cell["timed"]:
        argv += ["--quality-only"]
    return argv


def dry_run(plan):
    root, work = Path(plan["root"]), Path(plan["work"])
    lines = ["X15 STAGE 1 / DRY RUN: no GPU or dataset loading; no commands executed.",
             "Stage 2 requires: " + AUTHORIZATION,
             "NF=1; sampled final gradient clip=1.0; G3; total deadline=18000 s.",
             'Each comparison recorder acquires the shared SREGNAS_GPU_LOCK before data/GPU access.',
             "2.7: all eight replicas run concurrently; the driver holds the global lock for the group.",
             "Execution uses a clean local checkout of the committed code; PYTHONPATH points to its src."]
    for cell in plan["cells"]:
        out = run_path(cell, work)
        lines += [f"{cell['step']} {cell['cell_id']} expected={cell['expected_seconds']}s "
                  f"timed={str(cell['timed']).lower()} optional={str(cell['optional']).lower()} "
                  f"requires={cell.get('requires', 'none')}",
                  "  " + shlex.join(command(cell, root, work, out))]
    return "\n".join(lines) + "\n"
