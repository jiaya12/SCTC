# (Temporary)SCTC — Severity-Conditioned Threshold Calibration

Code for the paper *"Severity-Stratified Calibration and Thresholding for
Reliable ICU Alerts."*

This repository implements the full pipeline that turns ICU risk trajectories
into severity-safe alerts: cohort construction, risk prediction, and the SCTC
decision layer (severity-stratified calibration, per-stratum threshold selection
under sensitivity floors, and a finite-sample floor certificate), together with
the evaluation used in the paper. The primary upstream predictor is the released
MIMIC-Sepsis LSTM, used frozen. The repository also includes the
logistic-regression and gradient-boosted backbones that the paper trains for its
cross-backbone robustness analysis (Task 1) and for the septic-shock task
(Task 2), so the pipeline runs end to end from the benchmark output.

## What this repository provides

- `sctc_core.py` — severity-stratified isotonic calibration, per-stratum
  threshold selection, the finite-sample (Clopper–Pearson) certificate, and the
  metrics (ECE, Brier, AUROC), all under stay-level cross-fitting.
- `backbones.py` — the logistic-regression and gradient-boosted predictors
  (scikit-learn) used for the cross-backbone robustness check (Task 1) and the
  septic-shock task (Task 2), returning out-of-fold predictions.
- `evaluation.py` — leave-one-ICU-out transfer of the calibration and
  thresholds, and decision-curve net benefit used as a no-harm check.
- `config.py` — data locations, severity-strata definition, and default
  settings in one place.
- `run_vaso.py` — Task 1 (vasopressor escalation): main results, backbone
  comparison, the thresholding-vs-calibration ablation, finite-sample coverage,
  leave-one-ICU transfer, and the net-benefit check.
- `run_septic.py` — Task 2 (septic-shock onset within 24 h): cohort
  construction, main results, and the label-shuffled control.
- `reproduce_figures.py` — regenerates the reliability and per-stratum
  sensitivity figures into `./figures`.
- `make_demo_data.py` — synthetic-data generator for running the pipeline
  without PhysioNet credentials.
- `tests/` — unit tests for the core routines.

## Requirements

Python 3.9+ 

```bash
pip install -r requirements.txt
```

## Data

The pipeline starts from **MIMIC-IV** (PhysioNet, credentialed) through its
**MIMIC-Sepsis** benchmark. The raw data are not redistributed here.

1. Obtain MIMIC-IV access on
   [PhysioNet](https://physionet.org/content/mimiciv/) and complete the Data Use
   Agreement.
2. Run the [MIMIC-Sepsis benchmark](https://github.com/yongh7/MIMIC-sepsis) to
   build the cohort and the upstream risk predictions.
3. Put the two resulting tables in a data directory (default `./data`):
   - `timeseries.csv` — per-window features, SOFA components, and labels.
   - `predictions.csv` — the upstream risk predictions.
4. Point `SCTC_DATA_DIR` at that directory (and optionally `SCTC_PRED_CSV`).

### Demo data (no credentials required)

To run the pipeline immediately, generate a synthetic dataset with the same
schema as the benchmark output:

```bash
python make_demo_data.py                       # writes ./demo_data/{timeseries,predictions}.csv
SCTC_DATA_DIR=./demo_data python run_vaso.py
SCTC_DATA_DIR=./demo_data python run_septic.py
```

Or run the whole pipeline with a single command:

```bash
python run_all.py --demo      # or: make demo
```

The demo data are synthetic. They reproduce the qualitative behaviour of the
pipeline, not the numbers reported in the paper.

## Running

```bash
SCTC_DATA_DIR=./data python run_vaso.py      # Task 1
SCTC_DATA_DIR=./data python run_septic.py    # Task 2
```

`run_vaso.py` prints the main result, the alternative-backbone rows, the
thresholding-vs-calibration ablation, and the finite-sample coverage.
`run_septic.py` prints the main result and the label-shuffled control. Values
are averaged over five stay-level splits, with no patient overlap across folds.

## Figures

```bash
SCTC_DATA_DIR=./demo_data python reproduce_figures.py   # writes ./figures/*.png
```

## Tests

```bash
python tests/test_sctc.py     # or: pytest
```
