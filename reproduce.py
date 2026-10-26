"""Build paper tables and figures from per-split experiment results."""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
ORDER = ["none", "0.5", "0.3", "0.2", "0.1", "0.05", "0.02"]


def summarize(results):
    keys = ["condition", "strategy", "random_seed", "delta", "method"]
    rows = []
    for key, group in results.groupby(keys):
        row = dict(zip(keys, key))
        row.update(
            n_planned=len(group),
            n_feasible=int(group.feasible.sum()),
            n_joint_success=int(group.joint_success.sum()),
            joint_over_all=group.joint_success.mean(),
        )
        for metric in [
            "budget",
            "worst_sensitivity",
            "sensitivity_0",
            "sensitivity_1",
            "sensitivity_2",
            "fpr_0",
            "fpr_1",
            "fpr_2",
            "ppv_0",
            "ppv_1",
            "ppv_2",
        ]:
            row[metric + "_mean"] = group.loc[group.feasible, metric].mean()
            row[metric + "_sd"] = group.loc[group.feasible, metric].std()
        rows.append(row)
    return pd.DataFrame(rows)


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.pdf")
    fig.savefig(output / f"{name}.png", dpi=220)
    plt.close(fig)


def build(source, output):
    output.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(source / "all_operating_points.csv", dtype={"delta": str})
    calibration = pd.read_csv(source / "calibration.csv")
    subset = pd.read_csv(source / "no_current_use.csv", dtype={"delta": str})
    summary = summarize(results)
    summary.to_csv(output / "aggregate_all.csv", index=False)
    primary = summary[
        (summary.condition == "default") & (summary.strategy == "severity")
    ]
    paired = results[results.strategy == "severity"].pivot(
        index=["seed", "condition", "delta"], columns="method", values="budget"
    )
    paired["saved_pp"] = 100 * (paired.single - paired.stratified)
    paired.to_csv(output / "paired_savings.csv")
    rows = []
    for delta in ORDER:
        group = primary[primary.delta == delta].set_index("method")
        differences = paired.xs(
            ("default", delta), level=("condition", "delta")
        ).saved_pp
        rows.append(
            dict(
                margin=delta,
                single_alert_pct=100 * group.loc["single", "budget_mean"],
                sctc_alert_pct=100 * group.loc["stratified", "budget_mean"],
                single_success=f"{int(group.loc['single','n_joint_success'])}/20",
                sctc_success=f"{int(group.loc['stratified','n_joint_success'])}/20",
                saved_pp=differences.mean(),
                saved_sd_pp=differences.std(),
            )
        )
    table1 = pd.DataFrame(rows)
    table1.to_csv(output / "table_1.csv", index=False)
    rows = []
    for name, cuts, target in [
        ("sofa_lower", "2/5", 0.8),
        ("default", "3/6", 0.8),
        ("sofa_upper", "4/7", 0.8),
        ("target70", "3/6", 0.7),
        ("target90", "3/6", 0.9),
    ]:
        group = summary[
            (summary.condition == name)
            & (summary.strategy == "severity")
            & (summary.delta == "0.02")
        ].set_index("method")
        rows.append(
            dict(
                cuts=cuts,
                target=target,
                feasible=f"{int(group.loc['single','n_feasible'])}/20",
                single_alert_pct=100 * group.loc["single", "budget_mean"],
                sctc_alert_pct=100 * group.loc["stratified", "budget_mean"],
                single_success=f"{int(group.loc['single','n_joint_success'])}/20",
                sctc_success=f"{int(group.loc['stratified','n_joint_success'])}/20",
            )
        )
    pd.DataFrame(rows).to_csv(output / "table_2.csv", index=False)
    table3 = (
        calibration.groupby("calibration")[["worst_ece", "brier", "auroc", "offset_2"]]
        .mean()
        .loc[["raw", "global", "group"]]
    )
    table3 = table3.rename(
        columns={"worst_ece": "worst_ece_pct", "offset_2": "high_severity_offset_pp"}
    )
    table3[["worst_ece_pct", "high_severity_offset_pp"]] *= 100
    table3.to_csv(output / "table_3.csv")
    subset.groupby(["delta", "method"]).agg(
        feasible=("feasible", "sum"),
        budget=("budget", "mean"),
        low_sensitivity=("sens_0", "mean"),
        moderate_sensitivity=("sens_1", "mean"),
        high_sensitivity=("sens_2", "mean"),
        all_targets_met=("joint", "sum"),
    ).to_csv(output / "no_current_use_summary.csv")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.3,
        }
    )
    fig, axes = plt.subplots(
        2, 1, figsize=(3.45, 3.4), sharex=True, layout="constrained"
    )
    for strategy, method, label, color, marker in [
        ("severity", "single", "Single threshold", "#C46A00", "s"),
        ("severity", "stratified", "SCTC (SOFA strata)", "#006D9C", "o"),
        ("random", "stratified", "Random groups", "#7F7F7F", "^"),
    ]:
        values = summary[
            (summary.condition == "default")
            & (summary.strategy == strategy)
            & (summary.method == method)
        ]
        values = (
            values.groupby("delta")[["budget_mean", "joint_over_all"]].mean().loc[ORDER]
        )
        for ax, column in zip(axes, ["budget_mean", "joint_over_all"]):
            ax.plot(
                range(7),
                100 * values[column],
                marker=marker,
                ls="--" if strategy == "random" else "-",
                color=color,
                label=label,
                markersize=3.5,
            )
    axes[0].set_ylabel("Windows flagged (%)")
    axes[1].set_ylabel("Evaluations meeting\nall targets (%)")
    axes[0].set_ylim(35, 90)
    axes[1].set_ylim(0, 105)
    axes[0].legend(
        fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, frameon=False
    )
    axes[1].set_xticks(range(7), ["None", ".5", ".3", ".2", ".1", ".05", ".02"])
    axes[1].set_xlabel(r"Margin parameter $\delta$ (larger margin to the right)")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    save_figure(fig, output, "figure_1")
    # Supplementary figures use the same fixed protocol as the paper.
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), layout="constrained")
    for ax, metric, title in zip(
        axes,
        ["sensitivity", "fpr", "ppv"],
        ["Sensitivity (%)", "False-positive rate (%)", "Positive predictive value (%)"],
    ):
        for method, label, color, offset in [
            ("single", "Single threshold", "#C46A00", -0.18),
            ("stratified", "SCTC", "#006D9C", 0.18),
        ]:
            row = primary[(primary.delta == "0.1") & (primary.method == method)].iloc[0]
            ax.bar(
                np.arange(3) + offset,
                [100 * row[f"{metric}_{g}_mean"] for g in range(3)],
                width=0.36,
                label=label,
                color=color,
            )
        ax.set_xticks(range(3), ["Low", "Moderate", "High"])
        ax.set_ylabel(title)
        ax.set_ylim(0, 105)
    axes[0].axhline(80, color="0.4", ls=":", lw=1)
    axes[0].legend(
        fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, frameon=False
    )
    save_figure(fig, output, "supplement_group_detection")
    fig, ax = plt.subplots(figsize=(4.5, 3), layout="constrained")
    for method, label, color in [
        ("single", "Single threshold", "#C46A00"),
        ("stratified", "SCTC", "#006D9C"),
    ]:
        values = primary[primary.method == method].set_index("delta").loc[ORDER]
        ax.plot(
            100 * values.budget_mean,
            100 * values.worst_sensitivity_mean,
            "o-",
            color=color,
            label=label,
        )
    ax.set_xlabel("Windows flagged (%)")
    ax.set_ylabel("Mean minimum group sensitivity (%)")
    ax.legend(fontsize=7)
    save_figure(fig, output, "supplement_detection_tradeoff")
    fig, ax = plt.subplots(figsize=(4, 2.8), layout="constrained")
    means = calibration.groupby("calibration")[[f"ece_{g}" for g in range(3)]].mean()
    for j, (kind, label, color) in enumerate(
        [
            ("raw", "Original", "#888888"),
            ("global", "Global isotonic", "#C46A00"),
            ("group", "Stratum isotonic", "#006D9C"),
        ]
    ):
        ax.bar(
            np.arange(3) + (j - 1) * 0.25,
            100 * means.loc[kind],
            width=0.25,
            label=label,
            color=color,
        )
    ax.set_xticks(range(3), ["Low", "Moderate", "High"])
    ax.set_ylabel("Mean ECE (%)")
    ax.legend(fontsize=7)
    save_figure(fig, output, "supplement_calibration")
    # The table assertion checks provenance errors without changing any computed value.
    row = table1.set_index("margin").loc["0.1"]
    checks = {"single_alert_pct": 71.40, "sctc_alert_pct": 58.92, "saved_pp": 12.48}
    status = {
        key: abs(float(row[key]) - value) < 0.005 for key, value in checks.items()
    }
    status["single_success"] = row.single_success == "19/20"
    status["sctc_success"] = row.sctc_success == "18/20"
    (output / "paper_check.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "reference")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/paper")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if primary results differ from paper rounding.",
    )
    args = parser.parse_args()
    status = build(args.results, args.output)
    print(f"Tables and figures written to {args.output}")
    if args.check and not all(status.values()):
        raise SystemExit(
            "Primary results differ from the paper. See paper_check.json and input fingerprints."
        )


if __name__ == "__main__":
    main()
