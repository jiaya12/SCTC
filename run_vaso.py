"""
Task 1: vasopressor escalation. Runs the main results, the backbone comparison,
the thresholding-vs-calibration ablation, and the finite-sample coverage
experiment, starting from the MIMIC-Sepsis benchmark output.

Set SCTC_DATA_DIR (and optionally SCTC_PRED_CSV) to point at your data directory.
"""
import os
import numpy as np, pandas as pd
from sctc_core import (oof_calibrate, ece, budget_saved, worst_recall_gain,
                       budget_single, budget_sctc, per_stratum_threshold, recall,
                       conformal_threshold, make_folds, pava, iso_predict, auroc)
from backbones import gbdt_oof, logreg_oof
from evaluation import leave_one_icu, net_benefit

# Data directory produced by the MIMIC-Sepsis benchmark (see README), or set
# SCTC_DATA_DIR. For a run on synthetic data, use SCTC_DATA_DIR=./demo_data.
DATA_DIR = os.environ.get("SCTC_DATA_DIR", "./data")
PRED_CSV = os.environ.get("SCTC_PRED_CSV", os.path.join(DATA_DIR, "predictions.csv"))
FEATS = ['map','heart_rate','sbp_arterial','dbp_arterial','respiratory_rate','spo2','lactate',
         'creatinine','wbc','urea_nitrogen','platelets','gcs','age','shock_index',
         'sofa_resp','sofa_coag','sofa_liver','sofa_cns','sofa_renal']


def load():
    ev = pd.read_csv(PRED_CSV)[['stay_id','time','predicted_prob','true_event']]
    ev['time'] = ev['time'].astype('int64'); ev['stay_id'] = ev['stay_id'].astype('int64')
    ts = pd.read_csv(f"{DATA_DIR}/timeseries.csv",
                     usecols=['stay_id','careunit','timestamp','sofa_score','sofa_cv']+FEATS).astype({'stay_id':'int64','timestamp':'int64'})
    ev = ev.sort_values('time'); ts = ts.sort_values('timestamp')
    m = pd.merge_asof(ev, ts, left_on='time', right_on='timestamp', by='stay_id',
                      direction='nearest', tolerance=10800)
    noncv = (m['sofa_score'] - m['sofa_cv']).values
    sev = np.where(noncv <= 3, 0, np.where(noncv <= 6, 1, 2))
    return m, sev


def main():
    m, sev = load()
    y = m['true_event'].values.astype(float)
    lstm = m['predicted_prob'].values
    stays = m['stay_id'].values
    groups = [0, 1, 2]

    # ---- main results (LSTM), 5 seeds ----
    saved, gain, we_g, we_gr = [], [], [], []
    for seed in range(5):
        g, gr = oof_calibrate(lstm, y, sev, stays, seed)
        saved.append(budget_saved(g, gr, y, sev, 0.80, groups))
        gain.append(worst_recall_gain(g, gr, y, sev, 0.30, groups))
        we_g.append(max(ece(g[sev == s], y[sev == s]) for s in groups))
        we_gr.append(max(ece(gr[sev == s], y[sev == s]) for s in groups))
    print(f"[LSTM] AUROC={auroc(lstm,y):.3f}  worstECE {np.mean(we_g)*100:.1f}%->{np.mean(we_gr)*100:.1f}%"
          f"  budget saved@80%={np.mean(saved)*100:.1f}+/-{np.std(saved)*100:.1f}"
          f"  worst-sens gain@30%={np.mean(gain)*100:.1f}+/-{np.std(gain)*100:.1f}")

    # ---- alternative backbones (Table I robustness rows) ----
    try:
        X = m[FEATS].values.astype(float)
        for name, fn in [("GBDT", gbdt_oof), ("LogReg", logreg_oof)]:
            sv, gn = [], []
            for seed in range(5):
                raw = fn(X, y, stays, seed)
                g2, gr2 = oof_calibrate(raw, y, sev, stays, seed)
                sv.append(budget_saved(g2, gr2, y, sev, 0.80, groups))
                gn.append(worst_recall_gain(g2, gr2, y, sev, 0.30, groups))
            print(f"[{name}] budget saved@80%={np.mean(sv)*100:.1f}+/-{np.std(sv)*100:.1f}"
                  f"  worst-sens gain@30%={np.mean(gn)*100:.1f}+/-{np.std(gn)*100:.1f}")
    except ImportError:
        print("[backbones] scikit-learn not installed; skipping backbone comparison")

    # ---- ablation: thresholding vs calibration (seed 0) ----
    g, gr = oof_calibrate(lstm, y, sev, stays, 0)
    print("\n[ablation] budget saved / worst-ECE")
    print(f"  global+global   {0.0:+.1f}   {max(ece(g[sev==s],y[sev==s]) for s in groups)*100:.1f}%")
    b_glob = budget_single(g, y, sev, 0.80, groups)
    print(f"  global+PS       {(b_glob-budget_sctc(g,y,sev,0.80,groups))*100:+.1f}   (same ECE)")
    print(f"  raw+PS          {(b_glob-budget_sctc(lstm,y,sev,0.80,groups))*100:+.1f}   {max(ece(lstm[sev==s],y[sev==s]) for s in groups)*100:.1f}%")
    print(f"  groupISO+PS     {(b_glob-budget_sctc(gr,y,sev,0.80,groups))*100:+.1f}   {max(ece(gr[sev==s],y[sev==s]) for s in groups)*100:.1f}%")

    # ---- finite-sample coverage (naive vs conformal) ----
    alpha, delta = 0.80, 0.10
    inpool = make_folds(stays, 1) < 2  # fixed calibration pool for the calibrator
    grpool = g.copy()  # (reuse OOF group-cal scores as the calibrated score)
    cov_n = {s: 0 for s in groups}; cov_c = {s: 0 for s in groups}; splits = 100
    R = np.where(~inpool)[0]; Rstays = np.unique(stays[R]); rng = np.random.RandomState(7)
    for _ in range(splits):
        perm = rng.permutation(Rstays); thr = set(perm[:len(perm)//2])
        isthr = np.array([s in thr for s in stays]) & (~inpool); istest = (~inpool) & (~isthr)
        for s in groups:
            pos = grpool[(sev == s) & (y == 1) & isthr]
            if len(pos) < 20: continue
            mst = np.unique(stays[(sev == s) & (y == 1) & isthr]).size
            tn = np.sort(pos)[min(max(int(np.floor((1-alpha)*len(pos))),0),len(pos)-1)]
            tc = conformal_threshold(pos, mst, alpha, delta)
            post = (sev == s) & istest & (y == 1)
            if ((grpool >= tn) & post).sum()/max(post.sum(),1) >= alpha: cov_n[s] += 1
            if ((grpool >= tc) & post).sum()/max(post.sum(),1) >= alpha: cov_c[s] += 1
    print("\n[finite-sample] coverage of 80% floor (naive / conformal), target 90%")
    for s, nm in zip(groups, ['low','mod','high']):
        print(f"  {nm:4s}  {cov_n[s]/splits*100:.0f}%  /  {cov_c[s]/splits*100:.0f}%")

    # ---- leave-one-ICU-out transfer ----
    careunit = m['careunit'].values
    units, licu = leave_one_icu(lstm, y, sev, stays, careunit, 0.80, groups)
    if len(licu):
        print("\n[leave-one-ICU] budget saved on held-out units (pts)")
        for u, v in zip(units, licu):
            print(f"  {u:9s} {v*100:+.1f}")
        print(f"  mean {np.mean(licu)*100:+.1f}+/-{np.std(licu)*100:.1f}")

    # ---- net-benefit no-harm check ----
    pt = 0.2
    tau_s = min(per_stratum_threshold(g, y, sev, s, 0.80) for s in groups)
    alert_single = g >= tau_s
    alert_sctc = np.zeros(len(y), bool)
    for s in groups:
        alert_sctc |= (sev == s) & (gr >= per_stratum_threshold(gr, y, sev, s, 0.80))
    print(f"\n[net benefit @pt={pt}] single={net_benefit(y, alert_single, pt):.4f}  "
          f"SCTC={net_benefit(y, alert_sctc, pt):.4f}")


if __name__ == "__main__":
    main()
