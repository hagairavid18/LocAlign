"""Apo/holo evaluation: which CSVs are read, the ligand_rmsd unit check, and the apo structure cache."""
import gzip
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "aligner_dl"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "miners"))  # miners/aligners imports its siblings as top-level packages

import offline_metrics_apo
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure


def write(
    path: Path,
    ligand_rmsd: np.ndarray,
) -> None:
    pd.DataFrame({"ligand_rmsd": ligand_rmsd, "cath_degree": 1}).to_csv(path, index=False)


class ExperimentFilesTest(unittest.TestCase):
    def test_only_the_three_inputs_are_read(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for name in ("holo_holo_subset", "apo_apo", "apo_holo", "apo_holo.ligand_rmsd_angstrom",
                         "success_rates", "all_experiments_with_success", "old_vs_new_ckpt"):
                write(directory / f"{name}.csv", np.arange(60.0))
            self.assertEqual([p.name for p in offline_metrics_apo.experiment_files(str(directory))],
                             ["holo_holo_subset.csv", "apo_apo.csv", "apo_holo.csv"])

    def test_missing_inputs_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            write(Path(directory) / "apo_apo.csv", np.arange(60.0))
            self.assertEqual([p.name for p in offline_metrics_apo.experiment_files(directory)], ["apo_apo.csv"])


class UnitCheckTest(unittest.TestCase):
    def check(
        self,
        values: np.ndarray,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "apo_apo.csv"
            write(path, values)
            offline_metrics_apo.check_ligand_rmsd_unit(path)

    def test_loss_scale_values_are_rejected(self):
        angstrom = np.random.default_rng(0).uniform(0, 60, 200)
        with self.assertRaises(ValueError):
            self.check(angstrom / (1 + angstrom / 10))

    def test_angstrom_values_are_accepted(self):
        self.check(np.random.default_rng(0).uniform(0, 60, 200))

    def test_small_tables_are_not_judged(self):
        self.check(np.array([1.0, 2.0, 3.0]))


class RawStructureCacheTest(unittest.TestCase):
    def test_cache_hit_returns_the_first_model_like_a_download(self):
        try:
            from miners.apo_discovery.candidate_structure import download_raw_structure
        except ImportError as error:
            self.skipTest(f"miners dependencies are not installed: {error}")
        structure = Structure("1abc")
        model = Model(0)
        model.add(Chain("A"))
        structure.add(model)
        with tempfile.TemporaryDirectory() as directory:
            with gzip.open(os.path.join(directory, "1abc.pkl.gz"), "wb") as f:
                pickle.dump(structure, f)
            result = download_raw_structure("1abc", directory)
            self.assertIsInstance(result, Model)
            self.assertEqual(result["A"].id, "A")


if __name__ == "__main__":
    unittest.main()
