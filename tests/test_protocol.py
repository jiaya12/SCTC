import unittest
import numpy as np
from sctc.core import adjusted_target, fit_rules, operating
from sctc.core import split_masks, threshold
import pandas as pd


class ProtocolTests(unittest.TestCase):
    def test_no_margin_matches_empirical_target(self):
        self.assertEqual(adjusted_target(80, 0.8, None), 0.8)

    def test_infeasible_returned(self):
        self.assertIsNone(adjusted_target(1, 0.8, 0.1))

    def test_more_conservative_margin(self):
        a = [adjusted_target(100, 0.8, d) for d in [0.5, 0.3, 0.2, 0.1, 0.05, 0.02]]
        self.assertTrue(np.all(np.diff(a) >= 0))

    def test_raw_ties_inclusive(self):
        s = np.array([0.1, 0.5, 0.5, 0.5, 0.9])
        th = threshold(s, np.ones(5), 0.8)
        self.assertEqual(th, 0.5)
        self.assertEqual((s >= th).sum(), 4)

    def test_test_labels_cannot_change_rules(self):
        n = 600
        g = np.arange(n) % 3
        s = np.random.RandomState(7).rand(n)
        y = (np.arange(n) % 4 == 0).astype(int)
        sel = np.arange(n) < 300
        th, _ = fit_rules(s, y, g, np.arange(n), sel, 0.8, 0.1)
        y2 = y.copy()
        y2[~sel] = 1 - y2[~sel]
        th2, _ = fit_rules(s, y2, g, np.arange(n), sel, 0.8, 0.1)
        np.testing.assert_array_equal(th, th2)

    def test_rules_nested_on_unseen_scores(self):
        th = np.array([0.2, 0.4, 0.6])
        r = np.random.RandomState(4).rand(1000)
        g = np.arange(1000) % 3
        self.assertFalse(((r >= th[g]) & ~(r >= th.min())).any())

    def test_real_strata_evaluation(self):
        y = np.ones(6, int)
        g = np.array([0, 0, 1, 1, 2, 2])
        a = np.array([1, 1, 1, 1, 0, 0], bool)
        v = operating(y, g, a, 0.8)
        self.assertFalse(v["joint_success"])
        self.assertEqual(v["sensitivity_2"], 0)

    def test_patient_partition(self):
        d = pd.DataFrame({"subject_id": np.repeat(np.arange(30), 4)})
        ms = split_masks(d, 20261015)
        self.assertTrue((sum(ms) == 1).all())
        for a in range(3):
            for b in range(a):
                self.assertFalse(set(d.subject_id[ms[a]]) & set(d.subject_id[ms[b]]))


if __name__ == "__main__":
    unittest.main()
