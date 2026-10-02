"""Success criteria in scripts/offline_metrics.py: composite, LRMSD-only and the symmetry upper bound."""
import os
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import offline_metrics


def toy_pairs() -> pd.DataFrame:
    """Four pairs: a clean success, a symmetric near miss, a correspondence failure and a NaN."""
    return pd.DataFrame({
        "ligand_id": ["ATP", "ACT", "ATP", "ATP"],
        "cath_degree": [4, 0, 1, 2],
        "ligand_rmsd": [1.0, 5.0, 3.0, float("nan")],
        "corr_rmsd": [1.0, 1.0, 3.0, 1.0],
        "atom_type_fraction": [0.9, 0.9, 0.9, 0.9],
    })


class OfflineMetricsTest(unittest.TestCase):
    def process(
        self,
        experiment: str,
        symmetric_ligands: set[str] | None = None,
    ) -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, f"{experiment}.csv")
            toy_pairs().to_csv(path, index=False)
            return offline_metrics.process_experiment(path, experiment, symmetric_ligands=symmetric_ligands)

    def test_localign_uses_the_composite_criterion(self):
        df = self.process("baseline")
        self.assertEqual(df["success_ligand_rmsd<4"].tolist(), [True, False, False, False])

    def test_lrmsd_only_ignores_the_other_conditions(self):
        df = self.process("baseline")
        self.assertEqual(df["success_lrmsd_only<4"].tolist(), [True, False, True, False])

    def test_baseline_aligners_use_lrmsd_only(self):
        df = self.process("TMalign")
        self.assertEqual(df["success_ligand_rmsd<4"].tolist(), df["success_lrmsd_only<4"].tolist())

    def test_symmetry_upper_bound_relaxes_only_symmetric_ligands(self):
        df = self.process("baseline", symmetric_ligands={"ACT"})
        self.assertEqual(df["success_sym_upper_bound<4"].tolist(), [True, True, False, False])
        self.assertEqual(df["success_lrmsd_only_sym_upper_bound<4"].tolist(), [True, True, True, False])


if __name__ == "__main__":
    unittest.main()
