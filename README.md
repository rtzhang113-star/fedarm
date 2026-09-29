# FedARM

This repository contains the source code and numerical experiment runners for
**FedARM: Adaptive Residual Modeling for Federated Continual QoS Prediction in
Cloud--Edge Environments**.

The release contains core implementation code, baseline implementations, and
the commands needed to run the experiments reported in the paper. It does not
contain plotting code, paper figures, generated results, checkpoints, raw
data, caches, or internal experiment-management records. Every runner writes
auditable CSV/JSON summaries below `results/`.

## Repository layout

```text
code/                         Core implementation and baselines
  core/                       FedARM and shared numerical modules
  baseline/                   FedAvg, FL-LFT, MultiFed, FedRecGEL, FedEM
  config/                     Shared configuration
  prepare_data.py             Raw-data to local-cache conversion
experiment/                   Numerical experiment entry points
  rq2/                        FedARM and baseline comparison
  rq3/                        Component ablation
  rq4/                        Robustness evaluation
  rq5/                        Continual retention evaluation
  rq6/                        Parameter sensitivity
dataset/WSDream-QoSDataset3/ Dataset instructions only
results/                      Empty output directory
```

## Environment

Use Python 3.11 and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

FedARM and the RQ3--RQ6 runners use Numba CUDA and require an NVIDIA GPU with
a compatible CUDA driver. The three NumPy baselines use CPU execution.
PyTorch is not required by the public execution paths.

## Dataset

Obtain WS-DREAM Dataset 3 from the
[dataset provider](https://www.inpluslab.com/wsdream/) and place the raw
`rtRate` and `tpRate` files in `dataset/WSDream-QoSDataset3/`. The required
shape is `(64, 142, 4500)`. See
`dataset/WSDream-QoSDataset3/readme.txt` for the input format and validity
rules. Raw data and generated caches are not distributed.

Prepare the local caches from the repository root:

```bash
python code/prepare_data.py --qos rt tp
```

The generated caches are stored under `dataset/cache/`.

## Reproducing the paper experiments

All commands below are run from the repository root. Use `--max-time-slices 1`
or `--smoke` for a bounded preflight; a preflight is not a paper result.

### RQ2: FedARM and baseline comparison

Run FedARM on the five densities and five seeds:

```bash
python experiment/rq2/run_fedarm.py \
  --qos rt tp \
  --densities 0.01 0.05 0.10 0.15 0.20 \
  --seeds 42 3407 1024 1 0 \
  --output results/rq2_fedarm
```

Run the two baseline groups:

```bash
python code/baseline/run_gpu_baselines.py \
  --output results/rq2_baselines_gpu

python code/baseline/run_reproduced_baselines.py \
  --output results/rq2_baselines_numpy
```

### RQ3: component ablation

```bash
python experiment/rq3/run_rq3_ablation.py --dry-run
python experiment/rq3/run_accuracy_ablation.py --stop-on-error
python experiment/rq3/run_local_rumf.py --stop-on-error
```

The six-variant protocol and variant manifest are in
`experiment/rq3/rq3_ablation_protocol.json` and
`experiment/rq3/rq3_variants.json`.

### RQ4: robustness

```bash
python experiment/rq4/run_robustness.py \
  --protocol experiment/rq4/rq4_robustness_formal_protocol.json \
  --run-formal --resume --stop-on-error
```

Use `--smoke --stop-on-error` for a bounded execution check.

### RQ5: continual retention

```bash
python experiment/rq5/run_retention.py \
  --config experiment/rq5/rq5_retention_formal_protocol.json \
  --run-formal --resume

python experiment/rq5/run_retention.py \
  --config experiment/rq5/rq5_retention_supplement.json \
  --run-formal --resume
```

Use `--smoke` for the bounded configuration in
`experiment/rq5/smoke_config.json`.

### RQ6: parameter sensitivity

```bash
python experiment/rq6/run_sensitivity.py \
  --protocol experiment/rq6/rq6_sensitivity_formal_protocol.json \
  --run-formal --resume --stop-on-error
```

Use `--smoke --stop-on-error` for a bounded execution check. The sensitivity
protocol independently varies latent rank, private-residual step size, and
support-count exponent.

Formal runners refuse to overwrite a non-empty result directory. Use a fresh
output path below `results/`, or use `--resume` only when the existing
manifest and source hashes match.

## Core implementation and baselines

`code/core/` contains the causal data protocol, GPU models, residual updates,
service aggregation, quantization, and communication accounting modules.
`code/baseline/` contains FedAvg, FL-LFT, MultiFed, FedRecGEL, and FedEM.
`experiment/rq2/`--`experiment/rq6/` contain only numerical runners and their
paper-aligned protocol files. No plotting code is required for reproducing the
raw numerical outputs.

## License and data

No project license has been assigned. Dataset terms remain those of the
WS-DREAM provider, and third-party method provenance is documented in the
baseline source files.
