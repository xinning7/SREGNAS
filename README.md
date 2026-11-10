# SREGNAS

Code and runnable experiment configurations for the ICDE 2027 regular research paper. The authors confirm that the submission manuscript uses the 6 November 2026 result freeze. This public package omits datasets, historical run records, checkpoints, experimental reports and the adapted external baseline bundle.

## Requirements

The recorded platform is Linux x86_64, Python 3.11.13, eight NVIDIA RTX A6000 GPUs (48 GiB each), two Intel Xeon Gold 6326 CPUs and approximately 755 GiB host RAM. Full timing workloads require exclusive access to the eight GPUs. Minimum requirements on smaller hardware and numerical or timing tolerances on another platform have not been established. Large graph preparation requires substantial RAM, shared memory and scratch disk.

The pinned CUDA 12.1 dependencies are in `requirements.txt`. The CUDA-enabled `torch-sparse` wheel must include its METIS operator.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python scripts/check.py
CUDA_VISIBLE_DEVICES="" SREGNAS_CPU_THREADS=1 python -m pytest -q
```

## Public datasets and preparation

Use the official Flickr and Reddit datasets through PyTorch Geometric, and `ogbn-arxiv` and `ogbn-products` through OGB. The loaders are in `src/sregnas/data/`; official splits and the recorded preprocessing are used. `configs/data/csr_validation.json` and `docs/audit/dataset_*.json` contain input identity metadata, not dataset arrays. Dataset terms remain with their providers.

```bash
python scripts/prepare_data.py --data "$PWD/data" --dataset all
```

The command downloads missing public data, constructs canonical inputs, checks the recorded CSR digests and compiles both partition arms. It writes its own partition selections and hashes under `outputs/`; historical run records are not required. Preparation has not been verified on a clean host as part of this public packaging.

## Run experiments

`experiments/manifest.json` holds the effective arguments and environments. `scripts/run.py` prints a resolved command by default; `--execute` launches it. Use a new output directory for every run. Main SREGNAS searches use seeds 42, 43 and 44, Q=400 attempts, K=8 finalists, eight logical shards and eight workers. The recorded command overrides the stored proxy epoch setting with 50 optimizer steps.

```bash
python scripts/run.py --list --group main
python scripts/run.py --job main_sregnas_flickr_s42
python scripts/run.py --job main_sregnas_flickr_s42 \
  --data "$PWD/data" --output outputs/flickr_s42 --execute
python scripts/retrain_winner.py --source outputs/flickr_s42 \
  --output outputs/flickr_s42_replicas --gpu 0 --execute
```

The manifest also includes `table_v_fixed_pool`, `controlled_c1`, `controlled_c2`, `locality`, `table_vii_sensitivity`, partition ablations, cost, degree and scaling groups. Their configurations and candidate/replay lists are in `configs/` and `pools/`. Controlled comparisons and locality need the matching compilation cache; use `scripts/prepare_data.py --cache-group x27` or `--cache-group x28` before those workloads. Degree experiments also need `--degree`. A new run may select a different winner or produce a different timing on another machine.

## External baselines

The project-owned adapters are included. Exact historical baseline reruns require 39 adapted external files listed by path and SHA-256 in `provenance/external_files.json`: 36 under `baselines/fastgnas/` and three reference executors for AP, GraphNAP and PAS-PL. The upstream FastGNAS project is [lghosth/FastGNAS](https://github.com/lghosth/FastGNAS), recorded at revision `dc5cc1b41b64244941055a3dcd5d8dd2df1c6012`. An upstream checkout has not been verified as a byte-identical replacement for the adapted bundle. `scripts/attach_external.py` can verify and install an independently obtained exact bundle. Without it, the SREGNAS main path and CPU checks are available, while those baseline reruns are unavailable.

## Scope

This repository provides code and recipes to generate new measurements from public datasets. It does not provide historical measurements or a script to reconstruct the frozen paper tables without rerunning the experiments. The recorded GPU workloads and a fresh dependency installation have not yet been validated in a clean environment.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
