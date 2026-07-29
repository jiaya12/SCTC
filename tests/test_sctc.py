"""
Unit tests for the SCTC core. Run with pytest, or directly:

    python tests/test_sctc.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sctc_core import (pava, ece, per_stratum_threshold, recall,
                       conformal_threshold, auroc)


def test_pava_is_monotone():
    rng = np.random.RandomState(0)
    x = rng.rand(300)
    y = (rng.rand(300) < x).astype(float)
    _, fit = pava(x, y)
    assert np.all(np.diff(fit) >= -1e-9)


def test_ece_small_when_calibrated():
    p = np.linspace(0.02, 0.98, 3000)
    rng = np.random.RandomState(1)
    y = (rng.rand(3000) < p).astype(float)
    assert ece(p, y, bins=10) < 0.05


def test_threshold_meets_floor():
    rng = np.random.RandomState(2)
    score = rng.rand(3000)
    y = (rng.rand(3000) < 0.3).astype(float)
    sev = np.zeros(3000, int)
    tau = per_stratum_threshold(score, y, sev, 0, 0.8)
    assert recall(score, tau, y, sev, 0) >= 0.8 - 1e-6


def test_conformal_is_conservative():
    rng = np.random.RandomState(3)
    pos = rng.rand(300)
    naive = np.sort(pos)[int(np.floor(0.2 * len(pos)))]
    tc = conformal_threshold(pos, m_eff=60, alpha=0.8, delta=0.1)
    assert tc <= naive + 1e-9


def test_auroc_on_separable_data():
    y = np.r_[np.zeros(100), np.ones(100)]
    p = np.r_[np.full(100, 0.1), np.full(100, 0.9)]
    assert auroc(p, y) > 0.99


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok   {fn.__name__}")
    print(f"\n{len(tests)} tests passed")
