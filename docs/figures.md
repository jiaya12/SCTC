# Paper and supplementary figures

All figures are generated from the same final experiment results by `reproduce.py`. Each percentage is a mean over the corresponding repeated patient splits. The lines connect evaluated settings; they do not establish performance at unevaluated settings.

- **Figure 1** matches the paper's margin sweep: alert-positive windows and evaluations meeting all three sensitivity targets. Random-group results average five assignments within each split.
- **Group detection** shows sensitivity, false-positive rate and positive predictive value in the three severity strata at delta = 0.1. It displays both the reduction in false alerts and the reduction in detected positive windows.
- **Detection trade-off** shows the mean minimum stratum sensitivity versus alert-positive windows across the fixed margin grid. It is not a matched-sensitivity superiority test.
- **Calibration** compares original, global-isotonic and stratum-isotonic ECE within each severity group. It uses ten equal-width probability bins, as in the paper. This bar chart is not a reliability diagram.

Earlier figures for finite-sample guarantees, a second shock task or alternative backbones are retained only in the initial-submission Git history. They are not part of the final-paper reproduction workflow. Restoring a historical figure requires rerunning its analysis under the final cohort and protocol rather than copying the old image.
