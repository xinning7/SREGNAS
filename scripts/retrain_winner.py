
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from artifact import ROOT, read, environment, external_missing


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="new search output directory")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    info = read(source / "run.json")
    if info["status"] != "OK":
        p.error("source search must have completed successfully")
    if output.exists():
        p.error("choose a new output directory")

    from sregnas.icde27.x17.inventory import results, specs
    from sregnas.icde27.x17.timing_analysis import accuracy
    from sregnas.icde27.quality.replicas import derived_seed
    winner = accuracy(source, info)["winner_fingerprint"]
    matched = [s for s in specs(source, info, results(source)) if s["fingerprint"] == winner]
    assert len(matched) == 1 and matched[0]["replica"] == 0 and "retrain_seed" not in matched[0]
    template = matched[0]
    backend = "FastGNAS" if info["backend"] == "fastgnas-native" else "SREGNAS"
    plan = []
    for replica in (1, 2, 3, 4):
        spec = deepcopy(template)
        spec.update(replica=replica, gpu=args.gpu, mode="retrain", phase="replicas:retrain",
                    checkpoint_path=str(output / f"r{replica}/checkpoint.pt"))
        for field in ("proxy_steps", "proxy_batch_targets", "proxy_ap_targets"):
            spec.pop(field, None)
        plan.append(dict(replica=replica, derived_seed=derived_seed(backend, spec), spec=spec))
    if not args.execute:
        print(json.dumps(dict(source=str(source), winner=winner, tasks=plan, timing_valid=False), indent=2))
        return 0
    if backend == "FastGNAS" and external_missing():
        p.error("the authorized native baseline dependency is missing")
    env = environment({}, ROOT / "data")
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), SREGNAS_APPROVED_GPUS=str(args.gpu))
    receipt = template.get("partition_receipt")
    if receipt:
        choice = read(receipt["path"]).get("partition_selection", {})
        if choice.get("partition_mode") == "proxy_aware":
            env.update(SREGNAS_PARTITION_MODE="proxy_aware", SREGNAS_X25_PARTITION_RECORDS="1")
        if choice.get("forced_quality_arm"):
            env["SREGNAS_X25_PARTITION_ARM"] = choice["forced_quality_arm"]

    lockpath = Path(env["SREGNAS_GPU_LOCK"])
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    before = hashlib.sha256((source / "run.json").read_bytes()).hexdigest()
    with lockpath.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        from sregnas.icde27.x16.resources import inventory
        card = next(c for c in inventory() if c["gpu"] == args.gpu)
        assert not card["processes"], "GPU must be empty"
        output.mkdir(parents=True, exist_ok=False)
        observations = []
        for item in plan:
            folder = output / f'r{item["replica"]}'
            folder.mkdir()
            spec = item["spec"]
            test = dict(spec, mode="post_selection_test", phase="replicas:test")
            job = dict(method=backend, specs=[spec, test], output=str(folder / "production_results.json"))
            path = folder / "job.json"
            path.write_text(json.dumps(job, indent=2) + "\n")
            env["SREGNAS_ICDE27_RECORD_DIR"] = str(folder)
            with (folder / "raw.log").open("w") as log:
                subprocess.run([sys.executable, "-m", "sregnas.comparison.retrain", "--job", str(path)],
                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            values = read(job["output"])
            assert len(values) == 2 and all(v["status"] == "OK" for v in values)
            assert all(v["metrics"]["seed"] == item["derived_seed"] for v in values)
            observations.append(dict(replica=item["replica"], seed=item["derived_seed"],
                                     val_acc=values[0]["value"], test_acc=values[1]["metrics"]["test_acc"]))
        assert hashlib.sha256((source / "run.json").read_bytes()).hexdigest() == before
        (output / "summary.json").write_text(json.dumps(dict(source=str(source), source_sha256=before,
            winner=winner, timing_valid=False, replica_zero_preserved=True, observations=observations), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
