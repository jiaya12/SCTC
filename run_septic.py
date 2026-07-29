"""
Task 2: septic-shock onset within 24 h. Builds a temporally-safe early-warning
label over at-risk windows, trains a logistic backbone, and reproduces the SCTC
budget/sensitivity numbers and the negative control.

Edit DATA_DIR to your local copy.
"""
import os
import numpy as np, pandas as pd
from sctc_core import (oof_calibrate, ece, budget_saved, worst_recall_gain, auroc, make_folds)
from backbones import logreg_oof

# Point at your local MIMIC-Sepsis processed files, or set SCTC_DATA_DIR.
# For an out-of-the-box run on synthetic data, use SCTC_DATA_DIR=./demo_data (see README).
DATA_DIR = os.environ.get("SCTC_DATA_DIR", "./data")
H = 6  # 24 h horizon at 4 h/step
FEATS = ['map','heart_rate','sbp_arterial','dbp_arterial','respiratory_rate','spo2','lactate',
         'creatinine','wbc','urea_nitrogen','platelets','gcs','age','shock_index',
         'sofa_resp','sofa_coag','sofa_liver','sofa_cns','sofa_renal']


def build_cohort(n_control_stays=5500, seed=7):
    ts = pd.read_csv(f"{DATA_DIR}/timeseries.csv",
                     usecols=['stay_id','timestep','septic_shock','sofa_score','sofa_cv']+FEATS)
    ts = ts.sort_values(['stay_id','timestep']).reset_index(drop=True)
    onset = ts[ts['septic_shock'] >= 1].groupby('stay_id')['timestep'].min()
    onset_stays = set(onset.index)
    ts['onset'] = ts['stay_id'].map(onset)
    risk = ts[ts['septic_shock'] == 0].copy()             # at-risk = not yet in shock
    risk['label'] = ((risk['onset'] > risk['timestep']) & (risk['onset'] <= risk['timestep'] + H)).astype(int)
    allst = risk['stay_id'].unique()
    ctrl = np.array([s for s in allst if s not in onset_stays])
    rng = np.random.RandomState(seed)
    keep = set(s for s in allst if s in onset_stays) | set(rng.choice(ctrl, min(n_control_stays, len(ctrl)), replace=False))
    risk = risk[risk['stay_id'].isin(keep)].reset_index(drop=True)
    noncv = risk['sofa_score'] - risk['sofa_cv']
    sev = np.where(noncv <= 3, 0, np.where(noncv <= 6, 1, 2))
    return risk[FEATS].values.astype(float), risk['label'].values.astype(float), sev, risk['stay_id'].values


def main():
    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("scikit-learn is required for Task 2 (it trains the upstream predictor).")
        print("Install it with:  pip install scikit-learn")
        return
    X, y, sev, stays = build_cohort()
    groups = [0, 1, 2]
    print(f"cohort: {len(y)} windows, {np.unique(stays).size} stays, event rate {y.mean():.3f}")

    saved, gain, shuf = [], [], []
    stay_modal = pd.Series(sev, index=stays).groupby(level=0).agg(lambda x: x.value_counts().index[0])
    for seed in range(5):
        raw = logreg_oof(X, y, stays, seed)
        g, gr = oof_calibrate(raw, y, sev, stays, seed)
        saved.append(budget_saved(g, gr, y, sev, 0.70, groups))
        gain.append(worst_recall_gain(g, gr, y, sev, 0.30, groups))
        # negative control: shuffle strata at stay level
        rng = np.random.RandomState(300 + seed); u = np.unique(stays); perm = rng.permutation(u)
        mp = dict(zip(u, stay_modal.loc[perm].values)); sv_sh = np.array([mp[s] for s in stays])
        g2, gr2 = oof_calibrate(raw, y, sv_sh, stays, seed)
        shuf.append(budget_saved(g2, gr2, y, sv_sh, 0.70, groups))
    raw0 = logreg_oof(X, y, stays, 0)
    print(f"[septic/LogReg] AUROC={auroc(raw0,y):.3f}")
    print(f"  budget saved@70%={np.mean(saved)*100:.1f}+/-{np.std(saved)*100:.1f}"
          f"  worst-sens gain@30%={np.mean(gain)*100:.1f}+/-{np.std(gain)*100:.1f}"
          f"  control={np.mean(shuf)*100:.1f}+/-{np.std(shuf)*100:.1f}")


if __name__ == "__main__":
    main()
