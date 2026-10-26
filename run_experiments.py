"""Recompute the paper experiments from locally held benchmark inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sctc.cohort import prepare, dump
from sctc.core import split_masks, fit_rules, operating, calibrate, metrics

ROOT = Path(__file__).resolve().parent


def evaluate_rules(data, fit_groups, evaluation_groups, selection, test, target, delta):
    scores = data.predicted_prob.to_numpy()
    labels = data.true_event.to_numpy(int)
    thresholds, details = fit_rules(
        scores, labels, fit_groups, data.stay_id.to_numpy(), selection, target, delta
    )
    rows = []
    for method in ["single", "stratified"]:
        row = {
            "method": method,
            "feasible": thresholds is not None,
            "joint_success": False,
        }
        if thresholds is not None:
            cutoffs = (
                np.repeat(thresholds.min(), 3) if method == "single" else thresholds
            )
            alerts = scores >= cutoffs[fit_groups]
            for group, detail in enumerate(details):
                positive = selection & (fit_groups == group) & (labels == 1)
                assert alerts[positive].mean() + 1e-12 >= detail["adjusted_target"]
            row.update(
                operating(labels[test], evaluation_groups[test], alerts[test], target)
            )
            row["fit_group_joint_success"] = operating(
                labels[test], fit_groups[test], alerts[test], target
            )["joint_success"]
        rows.append(row)
    return rows


def run(data, config, output):
    scores = data.predicted_prob.to_numpy()
    labels = data.true_event.to_numpy(int)
    patients = np.sort(data.subject_id.unique())
    random_groups = {}
    for seed in config["random_group_seeds"]:
        order = np.random.RandomState(seed).permutation(patients)
        mapping = dict(zip(order, np.arange(len(order)) % 3))
        random_groups[seed] = data.subject_id.map(mapping).to_numpy(int)
    records, calibration, subset_records = [], [], []
    off_treatment = ~data.on_vaso.to_numpy(bool)
    subset = data.loc[off_treatment].copy()
    subset_groups = np.digitize(subset.severity, [3.5, 6.5])
    for seed in config["task1_seeds"]:
        calibration_set, selection, test = split_masks(data, seed)
        for name, low, high, target in config["conditions"]:
            severity = np.digitize(data.severity.to_numpy(), [low + 0.5, high + 0.5])
            strategies = [("severity", 0, severity)]
            if name == "default":
                strategies += [
                    ("random", rs, groups) for rs, groups in random_groups.items()
                ]
            for strategy, random_seed, groups in strategies:
                for delta in config["margin_grid"]:
                    for row in evaluate_rules(
                        data, groups, severity, selection, test, target, delta
                    ):
                        records.append(
                            dict(
                                seed=seed,
                                condition=name,
                                strategy=strategy,
                                random_seed=random_seed,
                                delta="none" if delta is None else str(delta),
                                target=target,
                                **row,
                            )
                        )
            if name == "default":
                global_scores, group_scores, _, _ = calibrate(
                    scores, labels, severity, calibration_set
                )
                for kind, probabilities in [
                    ("raw", scores),
                    ("global", global_scores),
                    ("group", group_scores),
                ]:
                    overall = metrics(
                        probabilities[test], labels[test], probabilities[test] >= 0.5
                    )
                    row = dict(
                        seed=seed,
                        calibration=kind,
                        **{
                            k: overall[k]
                            for k in ["brier", "auroc", "ece", "ece_quantile"]
                        },
                    )
                    for group in range(3):
                        mask = test & (severity == group)
                        result = metrics(
                            probabilities[mask],
                            labels[mask],
                            probabilities[mask] >= 0.5,
                        )
                        row[f"ece_{group}"] = result["ece"]
                        row[f"offset_{group}"] = float(
                            (labels[mask] - probabilities[mask]).mean()
                        )
                    row["worst_ece"] = max(row[f"ece_{g}"] for g in range(3))
                    calibration.append(row)
        # Keep the main-cohort patient assignments when restricting to off-treatment windows.
        for delta in config["margin_grid"]:
            for row in evaluate_rules(
                subset,
                subset_groups,
                subset_groups,
                selection[off_treatment],
                test[off_treatment],
                0.8,
                delta,
            ):
                subset_records.append(
                    dict(
                        seed=seed,
                        delta="none" if delta is None else str(delta),
                        method=row["method"],
                        feasible=row["feasible"],
                        budget=row.get("budget", np.nan),
                        **{
                            f"sens_{g}": row.get(f"sensitivity_{g}", np.nan)
                            for g in range(3)
                        },
                        joint=row["joint_success"],
                    )
                )
        print(f"Completed patient split {seed}", flush=True)
    pd.DataFrame(records).to_csv(output / "all_operating_points.csv", index=False)
    pd.DataFrame(calibration).to_csv(output / "calibration.csv", index=False)
    pd.DataFrame(subset_records).to_csv(output / "no_current_use.csv", index=False)
    groups = np.digitize(data.severity, [3.5, 6.5])
    info = dict(
        windows=len(data),
        on_vaso_windows=int(data.on_vaso.sum()),
        event_rate_on_vaso=float(data.loc[data.on_vaso, "true_event"].mean()),
        event_rate_naive=float(subset.true_event.mean()),
        on_vaso_share_by_stratum=[
            float(data.on_vaso[groups == g].mean()) for g in range(3)
        ],
        naive_windows=len(subset),
        naive_patients=int(subset.subject_id.nunique()),
        naive_event_rate_by_stratum=[
            float(subset.true_event[subset_groups == g].mean()) for g in range(3)
        ],
    )
    dump(output / "treatment_status.json", info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeseries", type=Path, required=True)
    parser.add_argument(
        "--predictions", type=Path, required=True,
        help="Local LSTM prediction export, e.g. data/lstm_predictions.csv",
    )
    parser.add_argument("--icustays", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/recomputed")
    args = parser.parse_args()
    if args.output.resolve() == (ROOT / "reference").resolve():
        parser.error(
            "Use a separate output directory to preserve the reference results."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "protocol.json").read_text())
    data, _ = prepare(args.timeseries, args.predictions, args.icustays, args.output)
    run(data, config, args.output)
    dump(args.output / "protocol.json", config)
    import sklearn, scipy, matplotlib, platform

    dump(
        args.output / "environment.json",
        dict(
            python=platform.python_version(),
            numpy=np.__version__,
            pandas=pd.__version__,
            scipy=scipy.__version__,
            sklearn=sklearn.__version__,
            matplotlib=matplotlib.__version__,
        ),
    )


if __name__ == "__main__":
    main()
