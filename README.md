# SCTC

Code for *Severity-Stratified Calibration and Thresholding for Reliable ICU Alerts*.
SCTC applies probability calibration and severity-specific alert thresholds to a fixed ICU predictor.

## What is included

- `run_experiments.py` — run the experiments from local inputs.
- `sctc/` — cohort preparation, calibration, thresholds and metrics.
- `reproduce.py` — generate tables and figures from experiment results.
- `make_demo_data.py` — generate synthetic inputs for a trial run.
- `reference/` — saved aggregate results used for the paper tables and figures.
- `protocol.json` — experiment settings and seeds.

## Requirements

Python 3.12 with NumPy, pandas, SciPy, scikit-learn and Matplotlib.

```bash
pip install -r requirements.txt
```

## Data

Patient data are not included. Use your local [MIMIC-Sepsis](https://github.com/yongh7/MIMIC-sepsis) files and LSTM prediction export:

- `patient_timeseries_v4.csv` — trajectories from `src/format_traj.py`.
- `icustays.csv` — ICU-stay mapping extracted from MIMIC-IV by `src/sql/icustays.py`.
- `lstm_predictions.csv` — frozen LSTM predictions with `stay_id, time, predicted_prob, true_event`.

See [input formats](docs/inputs.md) for the required columns. Calibration is fitted by this repository; no additional calibration of the prediction file is required.

For a trial run without MIMIC access, generate synthetic inputs:

```bash
python make_demo_data.py
```

These files are saved in `outputs/demo_data/`. They are generated entirely from random numbers to try the workflow and do not reproduce the paper results.

## Running

Run the experiments with your local files:

```bash
python run_experiments.py \
  --timeseries data/patient_timeseries_v4.csv \
  --predictions data/lstm_predictions.csv \
  --icustays data/icustays.csv \
  --output outputs/recomputed

python reproduce.py --results outputs/recomputed --output outputs/recomputed/paper
```

For synthetic inputs, replace `data/` with `outputs/demo_data/` and use `--output outputs/demo` in the first command. Then run `python reproduce.py --results outputs/demo --output outputs/demo/paper`.

To generate the paper tables and figures directly from saved aggregate results:

```bash
python reproduce.py
```

This last command does not rerun patient-level experiments.

## Figures and tables

- Local experiments: results in `outputs/recomputed/`; figures and tables in `outputs/recomputed/paper/`.
- Saved paper results: figures and tables in `outputs/paper/`.
- Synthetic trial run: results in `outputs/demo/`; figures and tables in `outputs/demo/paper/`.

Figures are saved as PDF and PNG; tables are saved as CSV. Further details are in [docs/protocol.md](docs/protocol.md) and [docs/figures.md](docs/figures.md).
