# SREGNAS

Code and experiment configurations for the ICDE 2027 regular research paper, aligned with the 6 November 2026 result freeze.

## Repository contents

- `src/sregnas/`: SREGNAS implementation, data processing, training, search and evaluation.
- `experiments/manifest.json` and `configs/`: resolved commands and settings for the main experiments, fixed pools, controlled comparisons, locality, sensitivity and scaling studies.
- `pools/`: candidate architectures and replay lists.
- `scripts/`: data preparation, experiment launch, winner retraining and checks.
- `tests/`: synthetic CPU regression tests.

Datasets and previous run outputs are not included. The commands below obtain public datasets and generate new results under `outputs/`.

## Environment

The recorded platform is Linux x86_64 with Python 3.11.13, eight NVIDIA RTX A6000 GPUs (48 GiB each), two Intel Xeon Gold 6326 CPUs and approximately 755 GiB host RAM. Timed runs require exclusive access to the GPU set. The pinned CUDA 12.1 dependencies are in `requirements.txt`; the CUDA-enabled `torch-sparse` wheel must provide METIS.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python scripts/check.py
CUDA_VISIBLE_DEVICES="" SREGNAS_CPU_THREADS=1 python -m pytest -q
```

## Data preparation

Flickr and Reddit are loaded through PyTorch Geometric; `ogbn-arxiv` and `ogbn-products` are loaded through OGB. The loaders use the official splits and recorded preprocessing. `configs/data/csr_validation.json` and `docs/audit/dataset_*.json` identify the expected inputs.

```bash
python scripts/prepare_data.py --data "$PWD/data" --dataset all
```

## Run experiments

`scripts/run.py` resolves a job from `experiments/manifest.json` and prints its command by default. Add `--execute` to run it; choose a fresh output directory for each run. The main SREGNAS configuration uses Q=400 launch attempts, K=8 finalists, eight logical shards and eight workers. The resolved command sets 50 proxy optimizer steps.

```bash
python scripts/run.py --list --group main
python scripts/run.py --job main_sregnas_flickr_s42
python scripts/run.py --job main_sregnas_flickr_s42 --data "$PWD/data" --output outputs/flickr_s42 --execute
python scripts/retrain_winner.py --source outputs/flickr_s42 --output outputs/flickr_s42_replicas --gpu 0 --execute
```

Other groups include `table_v_fixed_pool`, `controlled_c1`, `controlled_c2`, `locality`, `table_vii_sensitivity`, partition studies, cost, degree and scaling. List a group with `python scripts/run.py --list --group GROUP`. Controlled comparisons and locality require an untimed compilation cache prepared with `scripts/prepare_data.py --cache-group x27` or `--cache-group x28`; degree studies also require `--degree`.

For the reported comparison, GraphNAP-c was measured at Q=20 and its Q=400 search time was extrapolated. In the 32-candidate fixed-pool results, SREGNAS, AP-c and PAS-PL-c use the X25 worker-ready timing boundary; FastGNAS and GraphNAP-c inherit X18 measurements with a different boundary. The corresponding job definitions are in `experiments/manifest.json` and `configs/pools/`.

## External baselines

The repository includes comparison configurations and project adapters. The FastGNAS upstream source is [lghosth/FastGNAS](https://github.com/lghosth/FastGNAS) at revision `dc5cc1b41b64244941055a3dcd5d8dd2df1c6012`:

```bash
git clone https://github.com/lghosth/FastGNAS.git FastGNAS-upstream
git -C FastGNAS-upstream checkout --detach dc5cc1b41b64244941055a3dcd5d8dd2df1c6012
```

The recorded AP, GraphNAP, PAS-PL and FastGNAS experiments also used 39 locally adapted files, including a load-prediction checkpoint. Their source paths, destination paths and hashes are listed in `provenance/external_files.json`. These files are not included in this repository; the upstream checkout alone does not reproduce the exact adapted environment. With authorized access to the matching adapted checkout, verify and install it locally:

```bash
python scripts/attach_external.py --bundle /path/to/authorized/ShardGNAS
python scripts/attach_external.py --bundle /path/to/authorized/ShardGNAS --install
python -m pip install -e '.[baseline]'
```

The installer checks the listed hashes before copying files and applies the recorded project namespace replacement. Baseline jobs require this adapted bundle or a separately validated equivalent implementation.

## Verification

The package passed CPU configuration and identity checks, and representative commands from all experiment groups resolved. A fresh installation and full GPU rerun have not yet been validated.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
