"""Link frozen predictions to completed ICU input windows."""

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x: x.item()))


def prepare(timeseries, predictions, icustays, out):
    paths = [Path(predictions), Path(timeseries), Path(icustays)]
    audit = {
        "input_sha256": dict(
            zip(["predictions", "timeseries", "icustays"], map(sha, paths))
        )
    }
    cols = [
        "stay_id",
        "timestamp",
        "charttime",
        "timestep",
        "vaso_max",
        "vaso_median",
        "sofa_score",
        "sofa_cv",
        "age",
        "gender",
    ]
    ts = pd.read_csv(paths[1], usecols=cols)
    original_ids = ts.stay_id.unique().copy()
    np.random.RandomState(42).shuffle(original_ids)
    train_ids = original_ids[: int(0.8 * len(original_ids))]
    val_ids = original_ids[int(0.8 * len(original_ids)) :]
    ids = pd.read_csv(paths[2], usecols=["stay_id", "subject_id", "first_careunit"])
    assert not ids.stay_id.duplicated().any()
    train_patients = set(ids.loc[ids.stay_id.isin(train_ids), "subject_id"])
    ev = pd.read_csv(paths[0])
    ev["row_id"] = np.arange(len(ev))
    assert not ev.duplicated(["stay_id", "time"]).any()
    assert set(ev.true_event.unique()) <= {0, 1}
    assert ev.predicted_prob.between(0, 1).all()
    audit.update(
        input_windows=len(ev),
        input_stays=int(ev.stay_id.nunique()),
        upstream_train_stay_overlap=int(ev.stay_id.isin(train_ids).sum()),
        source_validation_fraction=float(ev.stay_id.isin(val_ids).mean()),
    )
    assert ev.stay_id.isin(
        val_ids
    ).all(), "Upstream split reconstruction does not match input"
    ts = ts.sort_values(["stay_id", "timestep"]).reset_index(drop=True)
    assert ts.groupby("stay_id").timestamp.diff().dropna().eq(14400).all()
    # Match exported time back to the first target row, then use the previous completed row.
    ts["future6"] = (
        ts.assign(v=(ts.vaso_max > 0) | (ts.vaso_median > 0))
        .groupby("stay_id")
        .v.transform(lambda x: x.iloc[::-1].rolling(6, min_periods=6).max().iloc[::-1])
    )
    ts["noncv"] = ts.sofa_score - ts.sofa_cv
    for c in ["timestamp", "noncv", "age", "gender"]:
        ts["previous_" + c] = ts.groupby("stay_id")[c].shift(1)
    m = pd.merge_asof(
        ev.sort_values("time"),
        ts.sort_values("charttime"),
        left_on="time",
        right_on="charttime",
        by="stay_id",
        direction="nearest",
        tolerance=1.0,
    )
    audit["unmatched_prediction_rows"] = int(m.charttime.isna().sum())
    audit["label_mismatches"] = int(
        (m.future6.notna() & m.future6.ne(m.true_event)).sum()
    )
    assert audit["unmatched_prediction_rows"] == 0 and audit["label_mismatches"] == 0
    assert m.future6.notna().all()
    m = m.merge(ids, on="stay_id", validate="many_to_one")
    assert m.subject_id.notna().all()
    overlap = m.subject_id.isin(train_patients)
    audit["excluded_upstream_training_patient_windows"] = int(overlap.sum())
    audit["excluded_upstream_training_patients"] = int(
        m.loc[overlap, "subject_id"].nunique()
    )
    m = m.loc[~overlap].copy()
    good = (
        m.previous_noncv.notna()
        & m.previous_noncv.between(0, 20)
        & (m.previous_timestamp + 14400 <= m.timestamp)
    )
    audit["excluded_invalid_prewindow_severity"] = int((~good).sum())
    m = m.loc[good].sort_values("row_id").reset_index(drop=True)
    m["severity"] = m.previous_noncv
    m["decision_time"] = m.timestamp
    m["age"] = m.previous_age
    m["gender"] = m.previous_gender
    keep = [
        "row_id",
        "subject_id",
        "stay_id",
        "decision_time",
        "predicted_prob",
        "true_event",
        "severity",
        "age",
        "gender",
        "first_careunit",
    ]
    m = m[keep]
    prior = ts[["stay_id", "timestamp", "vaso_max", "vaso_median"]].copy()
    prior["decision_time"] = prior.timestamp + 14400
    prior["on_vaso"] = (prior.vaso_max > 0) | (prior.vaso_median > 0)
    m = m.merge(
        prior[["stay_id", "decision_time", "on_vaso"]],
        on=["stay_id", "decision_time"],
        how="left",
        validate="many_to_one",
    )
    assert m.on_vaso.notna().all(), "Missing preceding treatment window"

    audit.update(
        retained_windows=len(m),
        retained_stays=int(m.stay_id.nunique()),
        retained_patients=int(m.subject_id.nunique()),
        event_rate=float(m.true_event.mean()),
        endpoint="vasopressor requirement in next six 4-hour windows (24h), not dose escalation or incident treatment",
        temporal_limit="severity uses completed input row; upstream feature engineering and chart availability not fully audited",
    )
    dump(out / "data_audit.json", audit)
    return m, audit
