"""
Generate a small synthetic dataset with the same schema as the MIMIC-Sepsis
processed files, so the pipeline can be run without PhysioNet credentials.

The synthetic data mirror the structure and the qualitative properties used by
SCTC (severity strata with increasing event rates and stratum-dependent
miscalibration). They are not real patient data and do not reproduce the exact
numbers reported in the paper, which require the credentialed MIMIC-Sepsis cohort.

Usage:
    python make_demo_data.py            # writes ./demo_data/*.csv
    SCTC_DATA_DIR=./demo_data python run_vaso.py
    SCTC_DATA_DIR=./demo_data python run_septic.py
"""
import os
import numpy as np
import pandas as pd

FEATS = ['map', 'heart_rate', 'sbp_arterial', 'dbp_arterial', 'respiratory_rate', 'spo2',
         'lactate', 'creatinine', 'wbc', 'urea_nitrogen', 'platelets', 'gcs', 'age',
         'shock_index', 'sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cns', 'sofa_renal']

OUT_DIR = "./demo_data"
N_STAYS = 800
SEED = 0


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.RandomState(SEED)
    rows = []
    for stay in range(1, N_STAYS + 1):
        g = rng.choice([0, 1, 2], p=[0.45, 0.35, 0.20])          # severity stratum
        unit = rng.choice(["MICU", "SICU", "CCU", "TSICU", "CVICU", "MICU/SICU"])
        noncv_base = {0: rng.randint(0, 4), 1: rng.randint(4, 7), 2: rng.randint(7, 12)}[g]
        T = rng.randint(8, 18)                                    # windows in this stay
        # septic-shock onset in ~25% of stays
        onset = rng.randint(4, T) if rng.rand() < [0.15, 0.25, 0.40][g] else None
        for t in range(T):
            noncv = max(0, noncv_base + rng.randint(-1, 2))
            sofa_cv = rng.randint(0, 3)
            # split noncv across five organ components
            comp = rng.multinomial(noncv, [0.2] * 5)
            p_true = [0.16, 0.20, 0.37][g]
            true_event = int(rng.rand() < p_true)
            # stratum-dependent miscalibration: high stratum is under-confident
            factor = [1.0, 0.95, 0.72][g]
            latent = -1.6 + 1.7 * true_event + 0.6 * (g - 1) + rng.normal(0, 0.6)
            pred = float(np.clip(_sigmoid(latent) * factor, 1e-3, 0.999))
            shock = int(onset is not None and t >= onset)
            # physiology drifts upward in the windows preceding septic-shock onset,
            # so the septic label is weakly predictable from features
            pre = 1.0 if (onset is not None and onset - 6 <= t < onset) else 0.0
            rows.append(dict(
                stay_id=stay, careunit=unit, timestep=t, timestamp=t, septic_shock=shock,
                sofa_score=noncv + sofa_cv, sofa_cv=sofa_cv,
                map=rng.normal(75, 12) - 3 * pre, heart_rate=rng.normal(88, 16) + 6 * pre,
                sbp_arterial=rng.normal(118, 18), dbp_arterial=rng.normal(64, 11),
                respiratory_rate=rng.normal(19, 4) + 2 * pre, spo2=rng.normal(96, 3) - 1 * pre,
                lactate=abs(rng.normal(2.0 + g, 1.2)) + 0.7 * pre,
                creatinine=abs(rng.normal(1.1 + 0.3 * g, 0.6)),
                wbc=abs(rng.normal(11, 5)) + 1.2 * pre, urea_nitrogen=abs(rng.normal(22, 10)),
                platelets=abs(rng.normal(210, 70)), gcs=rng.randint(6, 16),
                age=rng.randint(40, 90), shock_index=rng.normal(0.8, 0.2) + 0.12 * pre,
                sofa_resp=int(comp[0]) + int(pre), sofa_coag=int(comp[1]), sofa_liver=int(comp[2]),
                sofa_cns=int(comp[3]), sofa_renal=int(comp[4]),
                predicted_prob=pred, true_event=true_event, time=t))
    df = pd.DataFrame(rows)

    ts_cols = ['stay_id', 'careunit', 'timestep', 'timestamp', 'septic_shock', 'sofa_score', 'sofa_cv'] + FEATS
    df[ts_cols].to_csv(f"{OUT_DIR}/timeseries.csv", index=False)
    df[['stay_id', 'time', 'predicted_prob', 'true_event']].to_csv(
        f"{OUT_DIR}/predictions.csv", index=False)
    print(f"wrote {OUT_DIR}/timeseries.csv and {OUT_DIR}/predictions.csv "
          f"({len(df)} windows, {N_STAYS} stays)")


if __name__ == "__main__":
    main()
