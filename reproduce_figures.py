"""
Regenerate the main result figures from a run.

Produces reliability-by-stratum and per-stratum sensitivity panels into
./figures. Uses the primary (LSTM) predictions in the data directory.

Usage:
    SCTC_DATA_DIR=./demo_data python reproduce_figures.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sctc_core import oof_calibrate, per_stratum_threshold, recall

OURS, BASE = "#0072B2", "#D55E00"
NAMES = ["low", "moderate", "high"]


def _reliability(ax, score, y, label, color):
    edges = np.linspace(0, 1, 11)
    xs, ys = [], []
    for i in range(10):
        m = (score >= edges[i]) & (score < edges[i + 1])
        if m.sum() > 5:
            xs.append(score[m].mean())
            ys.append(y[m].mean())
    ax.plot(xs, ys, "o-", color=color, label=label)


def figure_reliability(raw, cal, y, sev, groups, path):
    fig, axes = plt.subplots(1, len(groups), figsize=(9, 3), sharey=True)
    for s, ax in zip(groups, np.atleast_1d(axes)):
        m = sev == s
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        _reliability(ax, raw[m], y[m], "raw", BASE)
        _reliability(ax, cal[m], y[m], "SCTC", OURS)
        ax.set_title(NAMES[s])
        ax.set_xlabel("predicted risk")
    axes[0].set_ylabel("observed rate")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_perstratum(score_single, score_sctc, y, sev, groups, budget, floor, path):
    thr = np.quantile(score_single, 1 - budget)
    single = [recall(score_single, thr, y, sev, s) for s in groups]
    sctc = [recall(score_sctc, per_stratum_threshold(score_sctc, y, sev, s, floor), y, sev, s)
            for s in groups]
    x = np.arange(len(groups))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(x - w / 2, np.array(single) * 100, w, color=BASE, label="single threshold")
    ax.bar(x + w / 2, np.array(sctc) * 100, w, color=OURS, label="SCTC")
    ax.set_xticks(x)
    ax.set_xticklabels(NAMES)
    ax.set_ylabel("sensitivity (%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    from run_vaso import load
    m, sev = load()
    y = m["true_event"].values.astype(float)
    raw = m["predicted_prob"].values
    stays = m["stay_id"].values
    groups = [0, 1, 2]
    g, gr = oof_calibrate(raw, y, sev, stays, 0)
    os.makedirs("figures", exist_ok=True)
    figure_reliability(raw, gr, y, sev, groups, "figures/reliability.png")
    figure_perstratum(g, gr, y, sev, groups, 0.30, 0.80, "figures/perstratum_sensitivity.png")
    print("wrote figures/reliability.png and figures/perstratum_sensitivity.png")


if __name__ == "__main__":
    main()
