
import csv
import hashlib
import json
from pathlib import Path

from sregnas.paths import PROJECT_ROOT as ROOT

DATASETS = {"Flickr": "flickr", "arXiv": "arxiv", "Reddit": "reddit", "Products": "products"}
METHODS = ("SREGNAS", "FastGNAS", "GraphNAP", "AP")
EXECUTORS = {
    "H0": ("python", "host", "host"),
    "G1": ("gpu", "host", "host"),
    "G2": ("gpu", "gpu", "host"),
    "G3": ("gpu", "gpu", "gpu"),
}
LIMITS = {"5": 7*3600*1.5, "4": 3600*1.5, "6": 5*3600*1.5,
          "7": 2*3600*1.5, "3": 1.5*3600*1.5}
PLAN_REVISION = "20261024_capacity_and_time_v2"


def apply_amendments(cell):
    pass
    cell = dict(cell)
    if (str(cell.get("experiment_id")) in ("6", "7", "3")
            and cell.get("method") == "AP" and cell.get("kind") == "replay"):
        if not cell.get("gc_consistency_repair"):
            cell["collect_search_garbage"] = True
        elif not isinstance(cell.get("collect_search_garbage"), bool):
            raise ValueError("authorized GC consistency repair requires a boolean target")
        cell["amendments_file"] = "docs/audit/p2p6_amendments.md"
    return cell


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pool(dataset, root=ROOT):
    path = root / "pools" / (DATASETS[dataset] + "_pool32.tsv")
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 32 and len({r["fingerprint"] for r in rows}) == 32
    return rows


def extremes(dataset, root=ROOT):
    rows = sorted(pool(dataset, root), key=lambda r: (int(r["canonical_parameters"]), r["fingerprint"]))
    return dict(zip(("light", "medium", "heavy"), (rows[0], rows[15], rows[-1])))


def freeze(root=ROOT):
    rows = [dict(architecture_size=size, pool_index=pool(ds, root).index(row), **row)
            for ds in ("Reddit", "Products") for size, row in extremes(ds, root).items()]
    target = root / "pools/exp5_archs.tsv"
    import io
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, list(rows[0]), delimiter="\t", lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    if target.exists() and target.read_text() != buffer.getvalue():
        raise AssertionError("frozen Exp5 selection drift")
    target.write_text(buffer.getvalue())
    return rows


def make_plan():
    cells = []
    def add(exp, dataset, method="SREGNAS", kind="replay", P=8, count=32, **kw):
        cell = dict(experiment_id=str(exp), dataset=dataset, method=method, kind=kind,
                    P=P, count=count, executor="G3", domain="local", cache_mode="cached",
                    record_type="profile" if exp == 3 else "reference" if kind == "cell" else "fixed_pool", **kw)
        cell["cell_id"] = "_".join(map(str, [exp, dataset, method, kind, P, count,
            cell.get("architecture_size", "pool"), cell["executor"], cell["domain"], cell["cache_mode"]]))
        cells.append(cell)
        return cell
    for ds in ("Reddit", "Products"):
        for size in ("light", "medium", "heavy"):
            for mode in EXECUTORS:
                c=add(5, ds, kind="cell", P=1, count=1, architecture_size=size)
                c["executor"]=mode; c["cell_id"] += "_" + mode
    for ds in ("Reddit", "Products"):
        for mode in ("H0", "G3"):
            c=add(5, ds, kind="retrain_pool", count=8)
            c["executor"]=mode; c["cell_id"] += "_" + mode
    for ds in ("Reddit", "Products"):
        for size in ("light", "medium", "heavy"):
            add(5, ds, "FastGNAS", "cell", 1, 1, architecture_size=size)
        for size in ("light", "medium", "heavy"):
            c=add(5, ds, "FastGNAS", "cell", 1, 1, architecture_size=size, skip_reason="时间")
            c["executor"]="H0"; c["cell_id"]+="_H0"
    for ds in ("Flickr", "arXiv"):
        for domain in ("local", "global"):
            for cache in ("cached", "rebuild"):
                c=add(4, ds); c.update(domain=domain, cache_mode=cache)
                c["cell_id"] += "_" + domain + "_" + cache
    sources = {}
    for ds in ("arXiv", "Products"):
        for method in METHODS:
            for p in (2, 4, 8):
                c=add(6, ds, method, P=p)
                if method == "FastGNAS" and p == 2:
                    c.update(skip_status="OOM", skip_reason="用户确认固定 S=8、P=2 容量不足；沿用 Flickr 冒烟容量证据，不重触发",
                             capacity_reference=True, failure_stage="fixed_S8_capacity")
                if method == "GraphNAP":
                    c["scaling_mode"] = "natural_scaling"
                    if ds == "Products": c.update(skip_status="NT", skip_reason="时间")
                if p == 8: sources[ds, method] = c["cell_id"]
            if method == "FastGNAS":
                for p in (2, 4, 8):
                    c=add(6, ds, method, P=p, scaling_mode="natural_scaling")
                    c["cell_id"] += "_natural_scaling"

                    if p == 8: c["reuse_cell_id"] = sources[ds, method]
    for ds in DATASETS:
        for method in (*METHODS, "PAS-PL"):
            c=add(7, ds, method)
            if (ds, method) in sources: c["reuse_cell_id"] = sources[ds, method]
            if method == "GraphNAP":
                c["scaling_mode"] = "natural_scaling"
                if ds in ("Reddit", "Products"):
                    c.pop("reuse_cell_id", None)
                    c.update(count=8, estimate_target_count=32, record_type="estimate")
                    c["cell_id"] += "_prefix8_estimate32"
            if method == "PAS-PL": c["capacity_probe"] = True
    for ds in ("arXiv", "Products"):
        for method in METHODS:
            add(3, ds, method, count=4 if method == "GraphNAP" and ds == "Products" else 16, profile=True)
            add(3, ds, method, "cell", 1, 1, architecture_size="medium", profile=True)
    assert len(cells) == len({c["cell_id"] for c in cells})
    return [apply_amendments(c) for c in cells]


def selected_rows(cell):
    if cell.get("x27_replay_manifest"):
        from sregnas.icde27.x27.replay_list import selected
        return selected(cell)
    if cell["kind"] == "cell":
        return [extremes(cell["dataset"])[cell.get("architecture_size", "medium")]]
    if "pool_indices" in cell:
        indices = cell["pool_indices"]
        if len(indices) != cell["count"] or len(set(indices)) != len(indices) or any(i < 0 or i >= 32 for i in indices):
            raise ValueError("invalid fixed-pool subset")
        rows = pool(cell["dataset"])
        return [rows[i] for i in indices]
    return pool(cell["dataset"])[:cell["count"]]


def logical_shards(cell):
    if cell["kind"] != "replay": return cell["P"]
    if cell.get("scaling_mode") == "natural_scaling" and cell["method"] in ("FastGNAS", "GraphNAP"):
        return cell["P"]
    return 8 if cell["method"] in ("SREGNAS", "FastGNAS", "GraphNAP") else None


def candidate_mapping(cell):
    n = logical_shards(cell)
    if cell["method"] == "SREGNAS": return "index_mod_8_v1"
    if cell["method"] == "FastGNAS": return f"native_rotation_{n}_v1"
    if cell["method"] == "GraphNAP": return f"all_{n}_replicas"
    return "architecture_parallel"


def execution_key(cell, cfg, rows):
    pass
    from sregnas.execution.provenance import implementation_sha256
    data_audit=ROOT/"docs/audit"/("dataset_"+cell["dataset"]+".json")
    stable = {k: cell.get(k) for k in ("dataset", "method", "kind", "P", "count", "executor",
        "domain", "cache_mode", "capacity_probe", "profile")}
    if cell["method"] in ("AP", "GraphNAP") and cell["kind"] == "replay":
        stable["collect_search_garbage"] = apply_amendments(cell).get("collect_search_garbage", False)
    stable.update(config=cfg, candidates=[r["fingerprint"] for r in rows],
                  architecture_semantics=[json.loads(r["native_architecture_json"] if cell["method"]=="FastGNAS" else r["architecture_json"]) for r in rows],
                  implementation_sha256=implementation_sha256(),
                  frozen_data_audit_sha256=hashlib.sha256(data_audit.read_bytes()).hexdigest(),
                  seed_rule="P0C_D8_explicit_per_architecture", checkpoint_rule="final_epoch",
                  S=logical_shards(cell), seed=42, candidate_mapping=candidate_mapping(cell),
                  cache_state="cold", boundary="local_graph_read_to_pool_end_v1")
    return digest(stable)
