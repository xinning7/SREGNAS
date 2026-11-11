
import argparse
import ast
import json
from pathlib import Path
from artifact import ROOT, jobs, external_missing


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    files = list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")) + list((ROOT / "tests").glob("*.py"))
    for path in files:
        ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))
    from sregnas.config import load_config
    from sregnas.core import Architecture
    from sregnas.icde27.space import layer_space
    from sregnas.icde27.p2p6.plan import pool
    configs = []
    assert len({j["id"] for j in jobs()}) == len(jobs()), "duplicate workload identifier"
    for job in jobs():
        path = ROOT / job["config"]
        assert path.is_file()
        if job["kind"] == "search":
            cfg = load_config(path)
            cfg.validate()
            assert cfg.search.base_seed == job["seed"]
            configs.append(job["id"])
    candidates = 0
    for dataset in ("Flickr", "arXiv", "Reddit", "Products"):
        for row in pool(dataset):
            assert Architecture(json.loads(row["architecture_json"])).fingerprint() == row["fingerprint"]
            candidates += 1
    assert len(layer_space()) == 2496
    result = dict(status="PASS", python_files=len(files), search_configs=len(configs),
                  candidate_fingerprints=candidates, legal_layers=2496, architecture_pairs=2496**2,
                  external_files_missing=len(external_missing()),
                  scope="CPU syntax/configuration/identity only; no real data or GPU")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
