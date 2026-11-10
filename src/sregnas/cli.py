

import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sregnas")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "prepare", "run", "estimate"):
        p = sub.add_parser(command)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--data-dir", type=Path)
        p.add_argument(
            "--partition-strategy", choices=["metis", "hierarchical", "random", "fi"]
        )
        if command in {"run", "inspect"}:
            from sregnas.training.executor import add_executor_arguments

            add_executor_arguments(p)
            p.add_argument("--require-fi-gpu", action="store_true")
        if command in {"prepare", "run"}:
            p.add_argument("--out", type=Path, required=True)
        if command == "run":
            p.add_argument(
                "--allow-gpu",
                action="store_true",
                help="use only after the GPU allocation was approved",
            )
            p.add_argument("--resume", action="store_true")
            p.add_argument("--hardware", type=Path, required=True)
        if command == "estimate":
            p.add_argument("--hardware", choices=["a6000", "p100"], default="a6000")
    p = sub.add_parser("report")
    p.add_argument("--run-dir", type=Path, required=True)
    p = sub.add_parser("compare")
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="run directory or profile directory",
    )
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if getattr(args, "data_dir", None):
        os.environ["SREGNAS_DATA_DIR"] = str(args.data_dir.expanduser().resolve())
    if args.command == "run" and not args.allow_gpu:
        parser.error(
            "run requires --allow-gpu after explicit GPU approval; inspect/prepare/estimate are CPU-only"
        )
    if args.command in {"inspect", "prepare", "run", "estimate"}:
        from .config import load_config

        cfg = load_config(args.config)
        if args.partition_strategy is not None:
            cfg.graph.partition_strategy = args.partition_strategy
        if args.command in {"run", "inspect"}:
            from sregnas.execution.options import resolve_executor

            resolve_executor(cfg, args)
            if args.require_fi_gpu:
                cfg.runtime.require_fi_gpu = True
        cfg.validate()
    if args.command == "inspect":
        print(json.dumps(cfg.to_dict(), indent=2))
    elif args.command == "estimate":
        from .reporting.resources import estimate

        print(json.dumps(estimate(cfg, args.hardware), indent=2))
    elif args.command == "prepare":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from .data.graph import build_graph_stage

        print(
            json.dumps(
                build_graph_stage(cfg, args.out.expanduser().resolve()), indent=2
            )
        )
    elif args.command == "run":
        from .execution.runner import run

        run(cfg, args.out.expanduser().resolve(), args.hardware, resume=args.resume)
    elif args.command == "report":
        from .reporting.report import generate_report

        generate_report(args.run_dir)
    elif args.command == "compare":
        from .reporting.compare import compare

        print(json.dumps(compare(args.reference, args.candidate, args.out), indent=2))
    return 0
