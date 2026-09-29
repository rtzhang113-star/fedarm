# FedARM

This repository contains the source code and execution commands for
**FedARM: Adaptive Residual Modeling for Federated Continual QoS Prediction in
Cloud--Edge Environments**.

The public release contains the numerical runners and their configurations. It
does not contain plotting scripts, paper figures, generated results,
checkpoints, raw data, or internal experiment-management records. Each command
writes auditable CSV/JSON summaries below `results/`.

## Environment

Use Python 3.11 and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

FedARM and the ablation, robustness, retention, and sensitivity runners use
Numba CUDA and require an NVIDIA GPU with a compatible CUDA driver. The three
NumPy baselines use CPU execution. PyTorch is not required by the public
execution paths.

## Dataset

Obtain WS-DREAM Dataset 3 from the
[dataset provider](https://www.inpluslab.com/wsdream/) and place the raw
`rtRate` and `tpRate` files in `data/WSDream-QoSDataset3/`. The required shape
is `(64, 142, 4500)`. See `data/WSDream-QoSDataset3/readme.txt` for the input
format and validity rules. Raw data and generated caches are not distributed.

Prepare the local caches from this directory:

```bash
python scripts/prepare_data.py --qos rt tp
```

## Reproducing the paper experiments

The manuscript uses the following RQ numbering. Some internal source
directories retain their historical names; the mapping is stated explicitly.

### RQ2: prediction and resource overhead

Run FedARM on the five densities and five seeds:

```bash
python scripts/run_fedarm.py \
  --qos rt tp \
  --densities 0.01 0.05 0.10 0.15 0.20 \
  --seeds 42 3407 1024 1 0 \
  --output results/rq2_fedarm
```

Run the GPU and NumPy baseline groups separately:

```bash
python code/baseline/rq1/run_gpu_baselines.py \
  --output results/rq2_baselines_gpu

python code/baseline/rq1/run_reproduced_baselines.py \
  --output results/rq2_baselines_numpy
```

The result directories contain per-slice metrics, resource telemetry, and
method manifests. Use `--max-time-slices 1` for a bounded preflight; a
preflight is not a paper result.

### RQ3: core-module ablation

The paper RQ3 runner is retained under the historical path
`code/rq2/ablation/`. Inspect the six-variant matrix, then run the accuracy
matrix and the Local RUMF control:

```bash
python code/rq2/ablation/run_rq2_ablation.py --dry-run
python code/rq2/ablation/run_accuracy_ablation.py --stop-on-error
python code/rq2/ablation/run_local_rumf.py --stop-on-error
```

Use `--smoke --stop-on-error` for a bounded execution check. The outputs are
written below `results/rq3_`.

### RQ4: service-support imbalance and partial participation

The paper RQ4 runner is retained under `code/rq3/robustness/`:

```bash
python code/rq3/robustness/run_robustness.py \
  --protocol code/rq3/robustness/rq3_robustness_formal_r2_protocol.json \
  --run-formal --resume --stop-on-error
```

For a bounded run, use `--smoke --stop-on-error`. The formal matrix contains
nine intervention scenarios, five densities, five seeds, both QoS attributes,
and 64 time slices. Outputs are written below `results/rq4_robustness_`.

### RQ5: continual state accumulation and historical retention

The paper RQ5 runner is retained under `code/rq4/retention/`. The two formal
configurations cover the five reported densities and should be run separately:

```bash
python code/rq4/retention/run_retention.py \
  --config code/rq4/retention/rq4_retention_formal_protocol_20260909.json \
  --run-formal --resume

python code/rq4/retention/run_retention.py \
  --config code/rq4/retention/rq4_retention_supplement_20260910.json \
  --run-formal --resume
```

Use `--smoke` with `smoke_config.json` for a bounded execution check. Outputs
are written below `results/rq5_retention_`.

### RQ6: parameter sensitivity

Run the robustness reference matrix first, because the sensitivity protocol
audits and reuses the default reference cells from RQ4. Then run:

```bash
python code/rq6/sensitivity/run_sensitivity.py \
  --protocol code/rq6/sensitivity/rq6_sensitivity_formal_protocol.json \
  --run-formal --resume --stop-on-error
```

Use `--smoke --stop-on-error` for a bounded execution check. The sensitivity
runner varies latent rank, private-residual step size, and support-count
exponent independently and writes raw summaries below
`results/rq6_sensitivity_`.

All formal runners refuse to overwrite a non-empty result directory. Use a
fresh `--output` path below `results/` for an independent run, or use
`--resume` only when the existing manifest and source hashes match.

## Core implementation

`continual_qos/` contains the causal data protocol, GPU model, residual
updates, service aggregation, quantization, and delivery accounting modules.
`code/baseline/rq1/` contains FedAvg, FL-LFT, MultiFed, FedRecGEL, and FedEM.
The RQ-specific directories contain only the numerical runners, protocols, and
JSON configurations needed to produce the raw experiment outputs.

The public source uses the paper's `FedSAA` spelling. Internal identifiers in
manifests are kept stable where they are part of the execution contract.

## License and data

No project license has been assigned. Dataset terms remain those of the
WS-DREAM provider, and third-party method provenance is documented in the
baseline source files.
