"""
Core utilities for Severity-Conditioned Threshold Calibration (SCTC).

Contents:
  - isotonic calibration via pool-adjacent-violators (PAVA)
  - metrics: expected calibration error, Brier score, AUROC
  - stay-level cross-fitting and out-of-fold calibration
  - per-stratum threshold selection and alert-budget metrics
  - finite-sample (Clopper-Pearson) threshold correction

Implemented in NumPy.
"""
import numpy as np
import math


# ---------- calibration ----------
def pava(x, y):
    """Isotonic regression via pool-adjacent-violators.
    Returns (sorted_x, fitted_y) usable with np.interp for prediction."""
    o = np.argsort(x, kind="mergesort")
    x = np.asarray(x)[o]
    y = np.asarray(y, float)[o]
    out = []                       # each block: [mean, weight]
    for v in y:
        out.append([v, 1.0])
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            v2, w2 = out.pop()
            v1, w1 = out.pop()
            nw = w1 + w2
            out.append([(v1 * w1 + v2 * w2) / nw, nw])
    fit = np.empty(len(x))
    i = 0
    for v, w in out:
        n = int(round(w))
        fit[i:i + n] = v
        i += n
    return x, fit


def iso_predict(xs, ft, xn):
    return np.interp(np.clip(xn, xs[0], xs[-1]), xs, ft)


def ece(p, y, bins=10):
    """Expected calibration error with equal-width bins."""
    p = np.asarray(p); y = np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1]) if i < bins - 1 else (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum() > 0:
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y, float)) ** 2))


# ---------- stay-level cross-fitting ----------
def make_folds(stays, seed, k=5):
    u = np.unique(stays)
    rng = np.random.RandomState(seed); rng.shuffle(u)
    fmap = {s: i % k for i, s in enumerate(u)}
    return np.array([fmap[s] for s in stays])


def oof_calibrate(raw, y, sev, stays, seed, k=5, min_group=50):
    """Out-of-fold global- and group-isotonic calibrated scores."""
    fa = make_folds(stays, seed, k)
    g = np.zeros(len(y)); gr = np.zeros(len(y))
    for f in range(k):
        tr = np.where(fa != f)[0]; te = np.where(fa == f)[0]
        xs, ft = pava(raw[tr], y[tr]); g[te] = iso_predict(xs, ft, raw[te])
        for s in np.unique(sev):
            a = tr[sev[tr] == s]; b = te[sev[te] == s]
            if len(a) > min_group:
                xs2, ft2 = pava(raw[a], y[a]); gr[b] = iso_predict(xs2, ft2, raw[b])
            else:
                gr[b] = g[b]
    return g, gr


# ---------- decision layer ----------
def per_stratum_threshold(score, y, sev, s, floor):
    """Largest threshold meeting `floor` sensitivity in stratum s (expectation)."""
    m = (sev == s) & (y == 1)
    sc = np.sort(score[m])
    i = min(max(int(np.floor((1 - floor) * len(sc))), 0), len(sc) - 1)
    return sc[i]


def recall(score, thr, y, sev, s):
    m = sev == s; a = score >= thr
    return ((a) & (y == 1) & m).sum() / max(((y == 1) & m).sum(), 1)


def budget_single(score, y, sev, floor, groups):
    """Alert budget of the single global threshold that meets `floor` in every stratum."""
    tau = min(per_stratum_threshold(score, y, sev, s, floor) for s in groups)
    return (score >= tau).mean()


def budget_sctc(score, y, sev, floor, groups):
    a = np.zeros(len(y), bool)
    for s in groups:
        a |= ((sev == s) & (score >= per_stratum_threshold(score, y, sev, s, floor)))
    return a.mean()


def budget_saved(g_score, gr_score, y, sev, floor, groups):
    """Budget saved (pts) by SCTC vs single global threshold, at a per-stratum floor."""
    return (budget_single(g_score, y, sev, floor, groups) -
            budget_sctc(gr_score, y, sev, floor, groups))


def worst_recall_gain(g_score, gr_score, y, sev, budget, groups, iters=30):
    """Worst-stratum sensitivity gain (SCTC vs single) at a fixed total budget."""
    thr = np.quantile(g_score, 1 - budget)
    gm = min(recall(g_score, thr, y, sev, s) for s in groups)
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        t = (lo + hi) / 2; a = np.zeros(len(y), bool)
        for s in groups:
            a |= ((sev == s) & (gr_score >= per_stratum_threshold(gr_score, y, sev, s, t)))
        if a.mean() > budget: hi = t
        else: lo = t
    a = np.zeros(len(y), bool)
    for s in groups:
        a |= ((sev == s) & (gr_score >= per_stratum_threshold(gr_score, y, sev, s, lo)))
    sm = min(((a) & (y == 1) & (sev == s)).sum() / max(((y == 1) & (sev == s)).sum(), 1) for s in groups)
    return sm - gm


# ---------- finite-sample (Clopper-Pearson) threshold ----------
def binom_cdf(n, p):
    """P(Bin(n,p) <= t) for t = 0..n."""
    lg = math.lgamma
    logpmf = np.array([lg(n + 1) - lg(j + 1) - lg(n - j + 1) + j * math.log(p) + (n - j) * math.log(1 - p)
                       for j in range(n + 1)])
    return np.cumsum(np.exp(logpmf))


def conformal_threshold(pos_scores, m_eff, alpha, delta):
    """Distribution-free lower-confidence threshold guaranteeing
    P(sensitivity >= alpha) >= 1 - delta, using effective sample size m_eff
    (e.g. number of positive STAYS to be conservative under within-stay correlation)."""
    sc = np.sort(pos_scores); n = len(sc)
    cdf = binom_cdf(m_eff, alpha)
    tmin = int(np.argmax(cdf >= (1 - delta)))     # first t with coverage >= 1-delta
    a_emp = min((tmin + 1) / m_eff, 0.999)         # inflated empirical sensitivity target
    idx = min(max(int(np.floor((1 - a_emp) * n)), 0), n - 1)
    return sc[idx]


def auroc(p, y):
    p = np.asarray(p); y = np.asarray(y)
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    pos = y == 1; n1 = pos.sum(); n0 = (~pos).sum()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
