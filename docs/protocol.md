# Experiment settings

- The endpoint is recorded vasopressor use within the next six 4-hour windows, including continued treatment.
- Severity is non-cardiovascular SOFA from the preceding completed input window. Default cut points are 3 and 6.
- Patients appearing in the reconstructed upstream training partition are excluded.
- Each of 20 patient-level splits assigns 40% to calibration, 30% to threshold selection and 30% to testing. The predictor stays fixed.
- Thresholds act on the original scores. Isotonic calibration is evaluated separately on the probabilities.
- The common sensitivity target is 0.8; the primary margin parameter is 0.1. All settings and seeds are in `protocol.json`.
- Five random patient-group assignments are evaluated on the same 20 splits. Detection is evaluated in the real severity strata.
- Alert rates average feasible splits. Target-success counts include all planned splits. Repeated splits share patients and are not independent cohorts.

The stay-count margin is an empirical rule, not a finite-sample confidence guarantee. The benchmark preprocessing used for these results permits information across time and patients; this release reproduces the paper's post-hoc evaluation rather than validating an end-to-end prospective prediction system.


## Checking results

Run `python verify_results.py --results outputs/recomputed` to compare all per-split outputs against the recorded results. The `--check` option of `reproduce.py` checks the rounded primary results only. Omit it when using different inputs.

Run tests with `python -m unittest discover -s tests -v`.

## Earlier implementation

The initial submission code is preserved in [commit 83d387f](https://github.com/jiaya12/SCTC/tree/83d387fc2c620b49890ffb352cc94335f99bc1dd). It corresponds to the initial submission rather than the final paper.
