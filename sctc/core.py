"""Patient splits, calibration and alert metrics."""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import binom
from functools import lru_cache


def split_masks(d, seed):
    u = np.sort(d.subject_id.unique())
    u = np.random.RandomState(seed).permutation(u)
    n = len(u)
    a = int(0.4 * n)
    b = int(0.7 * n)
    sets = [set(u[:a]), set(u[a:b]), set(u[b:])]
    assert not any(sets[i] & sets[j] for i in range(3) for j in range(i))
    return [d.subject_id.isin(s).to_numpy() for s in sets]


def calibrate(raw, y, sev, cal):
    """Fit global and severity-specific isotonic maps on calibration patients.

    These maps are evaluated for probability calibration; alert thresholds
    are selected separately on the original LSTM input scores.
    """
    glob = IsotonicRegression(out_of_bounds="clip").fit(raw[cal], y[cal])
    gs = glob.predict(raw)
    gr = gs.copy()
    models = []
    for s in range(3):
        mk = cal & (sev == s)
        if mk.sum() < 20 or len(np.unique(y[mk])) < 2:
            raise ValueError(
                "Calibration group needs at least 20 windows and both outcome classes"
            )
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw[mk], y[mk])
        gr[sev == s] = iso.predict(raw[sev == s])
        models.append(iso)
    return gs, gr, glob, models


def threshold(score, y, floor):
    positive = np.sort(score[y == 1])
    if not len(positive):
        raise ValueError("No positives for threshold selection")
    # >= convention: keep at least ceil(alpha*n) positives, including all ties.
    need = int(np.ceil(floor * len(positive) - 1e-12))
    return float(positive[len(positive) - need])


def ece(score, y, equal_frequency=False):
    if not len(y):
        return np.nan
    edges = (
        np.unique(np.quantile(score, np.linspace(0, 1, 11)))
        if equal_frequency
        else np.linspace(0, 1, 11)
    )
    if len(edges) < 2:
        return abs(float(score.mean() - y.mean()))
    bins = np.digitize(score, edges[1:-1], right=True)
    return sum(
        (bins == i).mean() * abs(score[bins == i].mean() - y[bins == i].mean())
        for i in np.unique(bins)
    )


def metrics(score, y, alert):
    tp = int((alert & (y == 1)).sum())
    fp = int((alert & (y == 0)).sum())
    pos = int(y.sum())
    neg = len(y) - pos
    return dict(
        n=len(y),
        positives=pos,
        alerts=int(alert.sum()),
        tp=tp,
        fp=fp,
        budget=float(alert.mean()),
        sensitivity=tp / pos if pos else np.nan,
        fpr=fp / neg if neg else np.nan,
        ppv=tp / (tp + fp) if tp + fp else np.nan,
        specificity=1 - fp / neg if neg else np.nan,
        auroc=float(roc_auc_score(y, score)) if pos and neg else np.nan,
        brier=float(brier_score_loss(y, score)),
        ece=ece(score, y),
        ece_quantile=ece(score, y, True),
    )


@lru_cache(None)
def adjusted_target(n, target, delta):
    if delta is None:
        return target
    if n < 1:
        return None
    k = np.arange(int(np.ceil(n * target)), n + 1)
    feasible = k[binom.sf(k - 1, n, target) <= delta / 3]
    return float(feasible[0] / n) if len(feasible) else None


def fit_rules(raw, y, groups, stays, selection, target, delta):
    th = []
    audit = []
    for g in range(3):
        mk = selection & (groups == g) & (y == 1)
        n = int(np.unique(stays[mk]).size)
        a = adjusted_target(n, target, delta)
        t = None if a is None or not mk.any() else threshold(raw[mk], y[mk], a)
        th.append(t)
        audit.append(
            dict(
                group=g,
                positive_windows=int(mk.sum()),
                positive_stays=n,
                adjusted_target=a,
                threshold=t,
            )
        )
    if any(t is None for t in th):
        return None, audit
    return np.array(th), audit


def operating(y, g, alert, target):
    r = {"n": len(y), "positives": int(y.sum()), "budget": float(alert.mean())}
    for s in range(3):
        mk = g == s
        p = int(y[mk].sum())
        neg = int(mk.sum() - p)
        tp = int((alert & mk & (y == 1)).sum())
        fp = int((alert & mk & (y == 0)).sum())
        r.update(
            {
                f"sensitivity_{s}": tp / p if p else np.nan,
                f"ppv_{s}": tp / (tp + fp) if tp + fp else np.nan,
                f"fpr_{s}": fp / neg if neg else np.nan,
            }
        )
    v = np.array([r[f"sensitivity_{s}"] for s in range(3)])
    r["worst_sensitivity"] = float(np.min(v))
    r["joint_success"] = bool(np.isfinite(v).all() and (v >= target).all())
    return r
