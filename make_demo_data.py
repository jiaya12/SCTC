"""Generate synthetic inputs for trying the workflow, not paper results."""
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    output = Path('outputs/demo_data')
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(17)
    stays = np.arange(1, 901)
    order = stays.copy()
    np.random.RandomState(42).shuffle(order)
    held_out = set(order[int(0.8 * len(stays)):])
    trajectories, predictions = [], []
    for stay in stays:
        severity = int(rng.choice([2, 5, 8]))
        treatment = rng.binomial(1, 0.08 + severity * 0.01, size=18)
        for step in range(18):
            time = float(stay * 1000000 + step * 14400)
            trajectories.append(dict(stay_id=int(stay), timestamp=time,
                charttime=time, timestep=step, vaso_max=int(treatment[step]),
                vaso_median=int(treatment[step]), sofa_score=severity + 1,
                sofa_cv=1, age=60, gender=0))
            if stay in held_out and 1 <= step <= 12:
                label = int(treatment[step:step + 6].max())
                score = float(rng.beta(4, 2) if label else rng.beta(2, 4))
                predictions.append(dict(stay_id=int(stay), time=time,
                    predicted_prob=score, true_event=label))
    pd.DataFrame(trajectories).to_csv(output / 'patient_timeseries_v4.csv', index=False)
    pd.DataFrame(predictions).to_csv(output / 'lstm_predictions.csv', index=False)
    pd.DataFrame(dict(stay_id=stays, subject_id=stays,
        first_careunit='Synthetic ICU')).to_csv(output / 'icustays.csv', index=False)
    print(f'Synthetic inputs written to {output}; these do not reproduce paper results.')


if __name__ == '__main__':
    main()
