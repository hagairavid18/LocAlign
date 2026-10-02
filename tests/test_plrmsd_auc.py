"""AUC scoring in scripts/plrmsd_auc_score.py counts ties at the pLRMSD cap as 0.5."""
import importlib.util
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn is not installed")
class TieSafeAucTest(unittest.TestCase):
    def test_float_noise_at_the_cap_is_a_tie(self):
        from plrmsd_auc_score import auc_score
        labels = np.array([1, 1, 0, 0])
        neg_plrmsd = -np.array([10.0, 10.0, 9.999997, 9.999998])
        self.assertAlmostEqual(auc_score(labels, neg_plrmsd), 0.5)

    def test_clear_separation(self):
        from plrmsd_auc_score import auc_score
        labels = np.array([1, 1, 0, 0])
        self.assertAlmostEqual(auc_score(labels, -np.array([2.0, 3.0, 8.0, 9.0])), 1.0)


if __name__ == "__main__":
    unittest.main()
