"""The stored per-pair tables reproduce the paper's Table 1 numbers for the main model.

Needs the untracked per-pair tables under ablation_dfs/ (DATA_ROOT, see .env.example). The test is
skipped when they are absent, for example in a fresh clone without the data.
"""
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "aligner_dl"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import offline_metrics
from utils.constants import ABLATION_DFS_DIR

TABLE1 = {
    "homology": {"different_fold": (1728, 37.0, 40.6), "same_fold": (585, 87.0, 87.0), "overall": (2313, 49.6, 52.3)},
    "ligand": {"different_fold": (2770, 13.1, 13.5), "same_fold": (416, 87.0, 87.0), "overall": (3186, 22.7, 23.1)},
}


class Table1ReproducibilityTest(unittest.TestCase):
    def test_main_model_success_rates(self):
        for split, expected in TABLE1.items():
            path = os.path.join(ABLATION_DFS_DIR, f"{split}_split", "baseline.csv")
            if not os.path.exists(path):
                self.skipTest(f"{path} not found; set LOCALIGN_DATA_ROOT to a checkout with the data")
            df = offline_metrics.process_experiment(path, "baseline")
            columns = {
                "different_fold": df[df["cath_degree"] < 4],
                "same_fold": df[df["cath_degree"] == 4],
                "overall": df,
            }
            for column, (n, composite, lrmsd_only) in expected.items():
                with self.subTest(split=split, column=column):
                    sub = columns[column]
                    self.assertEqual(len(sub), n)
                    self.assertAlmostEqual(100 * sub["success_ligand_rmsd<4"].mean(), composite, delta=0.05)
                    self.assertAlmostEqual(100 * sub["success_lrmsd_only<4"].mean(), lrmsd_only, delta=0.05)


if __name__ == "__main__":
    unittest.main()
