# FedARM baseline implementations

This directory contains the five baselines used in the FedARM comparison:
FedAvg, FL-LFT, MultiFed, FedRecGEL, and FedEM.  The implementations are
causal adaptations for continuous QoS regression on WS-DREAM Dataset 3; they
are not claims that the original recommendation or offline QoS experiments
were reproduced unchanged.

The code is split by execution backend.  `run_gpu_baselines.py` runs
FedAvg and FL-LFT with Numba CUDA.  `run_reproduced_baselines.py` runs
MultiFed, FedRecGEL, and FedEM with their NumPy reference adaptations.  Both
use the same protocol supplied by `code.core`:

```text
history state + support[t] -> update -> read-only query[t]
```

Query labels and future slices are never passed to an update.  RT and TP are
run separately with the fixed Dataset 3 shape `(64, 142, 4500)`, observation
density `1%, 5%, 10%, 15%, 20%`, and seeds `42, 3407, 1024, 1, 0`.

## Layout

- `methods/`: FedAvg, FL-LFT, FedRecGEL, and FedEM numerical modules;
- `gpu_models.py`: CUDA model wrappers for FedAvg and FL-LFT;
- `reference_models.py`: adapters for the three NumPy baselines;
- `run_*_baselines.py`: causal experiment runners;
- `*_config.json`: frozen method settings.

MultiFed is retained in `code/core/multifed.py` with its shared residual
model dependencies.  Dataset 3 has no geographic labels, so the public
adaptation uses the documented label-independent partition
`edge_id = user_id % 6`.

## Data and outputs

Place the provider's raw Dataset 3 files under `dataset/WSDream-QoSDataset3/` and
generate the local value caches at:

```text
dataset/cache/rt_values.npy
dataset/cache/tp_values.npy
```

Raw data and generated caches are excluded from the release.  The runners
write manifests, per-slice metrics, and summaries below `results/`; pass a
different `--output` path under `results/` when needed.  No paper tables,
experiment logs, checkpoints, or scheduling files are required to run these
baselines.

## Quick preflight

Run one causal time slice at 10% density before a formal matrix:

```powershell
python code/baseline/run_gpu_baselines.py `
  --methods fedavg fl_lft --densities 10 --seeds 42 --qos rt tp `
  --max-time-slices 1 --output results/preflight_gpu --stop-on-error

python code/baseline/run_reproduced_baselines.py `
  --methods multifed fedrecgel fedem --densities 10 --seeds 42 --qos rt tp `
  --max-time-slices 1 --output results/preflight_numpy --stop-on-error
```

Run the five-density, five-seed formal matrices explicitly when the caches and
runtime are available:

```powershell
python code/baseline/run_gpu_baselines.py --output results/baselines_gpu_formal
python code/baseline/run_reproduced_baselines.py --output results/baselines_numpy_formal
```

The CLI exposes only the five baselines listed above.  The CUDA module keeps a
small shared kernel container for numerical compatibility, but unreachable
compatibility branches are not selectable by the public factory or runners.

## Task adaptations

- **FedAvg** uses complete-model, support-cardinality-weighted averaging of a
  sigmoid AMF QoS regressor.  The optional first-slice support-intercept
  initialization uses support values only.
- **FL-LFT** factorizes the user--service--time tensor; user factors remain
  private while service and time factors are shared.
- **MultiFed** performs client-to-edge and edge-to-cloud aggregation using the
  fixed synthetic edge partition described above.
- **FedRecGEL** preserves separate private and shared representations and
  sharpness-aware local updates, while replacing binary recommendation loss
  with fixed-log1p QoS regression.
- **FedEM** preserves personalized service-state merging and similarity-based
  aggregation, adapted to continuous QoS prediction.

These substitutions are recorded in the runner manifests and should be kept
when reporting results.  Communication bytes follow each method's actual
payload convention; framework overhead and raw data transfer are excluded.
