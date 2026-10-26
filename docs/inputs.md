# Input files

The runner starts from frozen prediction exports and benchmark trajectories. It does not train the upstream LSTM. All three inputs remain local. Never commit patient-level predictions, labels, identifiers, or trajectories to the public repository, even after renaming the files.

## Predictions

| Column | Meaning |
| --- | --- |
| `stay_id` | ICU stay identifier |
| `time` | Numeric benchmark `charttime` for the first target window, in seconds |
| `predicted_prob` | Original model probability in [0, 1] |
| `true_event` | Binary recorded-use label for the next six target windows |

Use `lstm_predictions.csv` for the frozen prediction export. The runner reads `predicted_prob` as its input score. Each `(stay_id, time)` pair must be unique.

For each patient split, the code fits global and severity-specific isotonic calibration on the calibration partition and evaluates the resulting probabilities on the test partition. Alert thresholds are selected separately using the original input scores. No additional calibration of the CSV is required before running the code.

## Trajectories

The local MIMIC-Sepsis preprocessing script `src/format_traj.py` writes `patient_timeseries_v4.csv`; the vasopressor benchmark reads it as model input. It contains processed trajectories rather than model predictions.

Required columns in `patient_timeseries_v4.csv`:

`stay_id, timestamp, charttime, timestep, vaso_max, vaso_median, sofa_score, sofa_cv, age, gender`

`timestamp` is the regular 4-hour trajectory time in seconds; `charttime` is the numeric timestamp used to link exported predictions. The runner matches prediction time to `charttime` within one second, identifies the first target window, then takes severity from the preceding completed window. Within each stay, consecutive `timestamp` values must differ by 14,400 seconds.

A target window records use when `vaso_max > 0` or `vaso_median > 0`. The runner checks that a positive indicator anywhere in six complete target windows reproduces `true_event`. The no-current-use subset requires both indicators to be nonpositive in the preceding input window; it is not a lifetime treatment-naive cohort.

The original row order before sorting determines the first occurrence of each stay. The upstream partition is reconstructed by shuffling that stay list with NumPy `RandomState(42)` and taking the first 80% as training stays. Reordering the input file can therefore change this reconstruction. Patients with any stay in that partition are excluded from the post-hoc cohort. All prediction stays must belong to the reconstructed held-out partition.

## Stay-to-patient mapping

The local MIMIC-Sepsis script `src/sql/icustays.py` extracts this mapping from `mimiciv_icu.icustays`. It is a database export, not a model prediction. That extraction script writes pipe-delimited output; the local file used in the paper is comma-delimited, as required by this runner. Convert the delimiter if using the extraction output directly. The retained scripts establish the pipeline roles; no execution log was available to establish the complete generation history of these local copies.

`icustays.csv` requires `stay_id, subject_id, first_careunit`, with one row per stay. The original paper cohort contains 14,568 windows from 1,849 stays and 1,833 patients after exclusions. The no-current-use subset has 11,194 windows from 1,627 patients.

## Reproducibility scope

Use the upstream MIMIC-Sepsis documentation to generate inputs under the relevant data access agreement. The post-hoc code does not redistribute the prediction file. `reference/input_fingerprints.json` identifies the exact files used for the paper by SHA-256. SHA-256 equality verifies byte identity; it does not establish the provenance of the upstream training run.

The available prediction export matches the reconstructed partition and labels, but a matching trained checkpoint and training run record were not available when packaging this release. Consequently, exact reproduction from newly trained LSTM predictions is not promised. The included aggregate outputs allow every reader to reconstruct the reported tables and figures without those restricted inputs.
