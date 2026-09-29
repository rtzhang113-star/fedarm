# FedARM implementation package

The modules in this directory are the runtime dependency closure of
`scripts/run_fedarm.py`.  The public runner uses `gpu_full_training.py` for
FedARM's full GPU arithmetic and imports the data, causal split, prediction,
and communication-accounting modules through the package's relative imports.

Several filenames describe an earlier implementation layer, such as
`residual_model.py`, `drift_residual_recenter.py`, and
`personalized_service_adapter.py`.  They are retained because the validated
FedARM implementation inherits their state-transition and proposal classes;
they are not additional baselines or runnable historical experiments.  The
repository intentionally excludes the other research and experiment scripts.

`data.py` produces the read-only `time x user x service` NumPy caches consumed
by the runner.  Use `scripts/prepare_data.py` to create them from the raw
WS-DREAM Dataset 3 files.
