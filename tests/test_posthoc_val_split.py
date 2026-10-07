"""The a posteriori train/val assignment covers train.csv exactly and keeps val separate."""
import os
import sys
import unittest

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "aligner_dl"))

from datasets.utils.posthoc_val_split import cluster_split, ligand_split  # noqa: E402
from utils.constants import POSTHOC_VAL_ASSIGNMENT_CSV, POSTHOC_VAL_SPLITS  # noqa: E402

CSV_DIR = os.path.join(REPO, "datasets", "csv_files")


class PosthocValSplitTest(unittest.TestCase):
    def load(
        self,
        split: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train = pd.read_csv(os.path.join(CSV_DIR, split, "train.csv"))
        assignment = pd.read_csv(os.path.join(CSV_DIR, split, POSTHOC_VAL_ASSIGNMENT_CSV))
        return train, assignment

    def test_one_label_per_train_row_in_order(self):
        for split in POSTHOC_VAL_SPLITS:
            train, assignment = self.load(split)
            self.assertEqual(list(assignment["row"]), list(range(len(train))))
            self.assertTrue(set(assignment["split"]) <= {"train", "val", "straddling"})

    def test_committed_assignment_is_reproduced(self):
        for split, mode in POSTHOC_VAL_SPLITS.items():
            train, assignment = self.load(split)
            labels = cluster_split(train) if mode == "cluster" else ligand_split(train)
            self.assertEqual(labels, list(assignment["split"]))

    def test_no_cluster_shared_between_train_and_val(self):
        train, assignment = self.load("homology_25_10")
        part = {s: train[assignment["split"] == s] for s in ("train", "val")}
        clusters = {s: set(p["tar_cluster"]) | set(p["src_cluster"]) for s, p in part.items()}
        self.assertFalse(clusters["train"] & clusters["val"])

    def test_no_ligand_shared_between_train_and_val(self):
        train, assignment = self.load("ligand_25_10")
        ligands = {s: set(train.loc[assignment["split"] == s, "ligand_id"]) for s in ("train", "val")}
        self.assertFalse(ligands["train"] & ligands["val"])


if __name__ == "__main__":
    unittest.main()
