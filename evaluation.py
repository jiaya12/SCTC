"""
Evaluation utilities beyond the core metrics:

  - leave-one-ICU-out transfer of the SCTC calibration and thresholds
  - decision-curve net benefit, used as a no-harm check

These support the robustness and no-harm analyses reported in the paper.
"""
import numpy as np

from sctc_core import pava, iso_predict, per_stratum_threshold


# ---------- fit / apply group-isotonic calibration ----------
def fit_group_iso(raw, y, sev, groups, min_group=50):
    """Fit a global isotonic map and one per-stratum map on the given data."""
    glob = pava(raw, y)
    grp = {}
    for s in groups:
        m = sev == s
        grp[s] = pava(raw[m], y[m]) if m.sum() > min_group else glob
    return glob, grp


def apply_group_iso(glob, grp, raw, sev, groups):
    """Apply calibrators fitted by :func:`fit_group_iso` to new scores."""
    g = iso_predict(glob[0], glob[1], raw)
    gr = g.copy()
    for s in groups:
        m = sev == s
        xs, ft = grp[s]
        gr[m] = iso_predict(xs, ft, raw[m])
    return g, gr


# ---------- leave-one-ICU-out transfer ----------
def leave_one_icu(raw, y, sev, stays, careunit, floor, groups, min_stays=100):
    """Budget saved by SCTC on each held-out ICU.

    For each unit, the calibration maps and thresholds are fitted on all other
    units and evaluated on the held-out unit, so no information from the test
    unit is used during fitting.
    """
    units, saved = [], []
    for u in np.unique(careunit):
        te = careunit == u
        tr = ~te
        if np.unique(stays[te]).size < min_stays or tr.sum() == 0:
            continue
        glob, grp = fit_group_iso(raw[tr], y[tr], sev[tr], groups)
        g_tr, gr_tr = apply_group_iso(glob, grp, raw[tr], sev[tr], groups)
        g_te, gr_te = apply_group_iso(glob, grp, raw[te], sev[te], groups)
        tau_single = min(per_stratum_threshold(g_tr, y[tr], sev[tr], s, floor) for s in groups)
        b_single = (g_te >= tau_single).mean()
        alert = np.zeros(int(te.sum()), bool)
        for s in groups:
            tau = per_stratum_threshold(gr_tr, y[tr], sev[tr], s, floor)
            alert |= (sev[te] == s) & (gr_te >= tau)
        units.append(u)
        saved.append(b_single - alert.mean())
    return units, np.array(saved)


# ---------- decision-curve net benefit ----------
def net_benefit(y, alert, pt):
    """Net benefit of an alert rule at threshold probability ``pt``.

    NB = TP/N - FP/N * pt/(1-pt). Used to check that SCTC does not reduce net
    benefit relative to the single-threshold rule (a no-harm check).
    """
    y = np.asarray(y)
    alert = np.asarray(alert).astype(bool)
    n = len(y)
    tp = int((alert & (y == 1)).sum())
    fp = int((alert & (y == 0)).sum())
    return tp / n - fp / n * (pt / (1.0 - pt))
