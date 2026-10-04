"""Motifs given to scripts/inference.py reach the dataset as lists in every mode.

The dataset only uses a motif that is a list of residue numbers; any other value is silently ignored.
`models` is imported before `datasets` to follow the package's import order.
"""
import os
import sys
import tempfile
import unittest

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "aligner_dl"))
sys.path.insert(0, REPO)

import models  # noqa: F401
from datasets.scannet_dataset import ScanNetDataset

try:
    from scripts.inference import InferenceRunner
except ImportError as error:
    InferenceRunner = None
    IMPORT_ERROR = str(error)


@unittest.skipIf(InferenceRunner is None, "inference dependencies are not installed")
class DatabaseSearchMotifTest(unittest.TestCase):
    def runner(
        self,
        directory: str,
        tar_motif: str | None,
    ) -> InferenceRunner:
        database = os.path.join(directory, "database.csv")
        pd.DataFrame([
            {"src_protein": "1abc", "src_chain": "A", "src_motif": "[5, 6]"},
            {"src_protein": "2abc", "src_chain": "B", "src_motif": ""},
        ]).to_csv(database, index=False)
        return InferenceRunner(
            checkpoint_path=os.path.join(directory, "model.ckpt"),
            base_save_dir=directory,
            protein_database_search=("9xyz", "A", database),
            tar_motif=tar_motif,
        )

    def test_target_motif_is_a_list_for_every_database_row(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(directory, "20,21,22")
            runner._prepare_dataframe()
            self.assertEqual([pair.tar_motif for pair in runner._pairs], [[20, 21, 22], [20, 21, 22]])
            self.assertEqual([pair.src_motif for pair in runner._pairs], [[5, 6], None])

    def test_target_motif_survives_the_csv_round_trip_into_the_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(directory, "20,21,22")
            runner._prepare_dataframe()
            path = os.path.join(directory, "filtered_pairs.csv")
            pd.DataFrame([pair.to_dict() for pair in runner._pairs]).to_csv(path, index=False)
            dataset = ScanNetDataset(df_path=path, inference=True, max_length=None, ligand_column="ligand")
            self.assertEqual(dataset._df.iloc[0]["tar_motif"], [20, 21, 22])

    def test_no_target_motif_stays_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(directory, None)
            runner._prepare_dataframe()
            self.assertTrue(all(pair.tar_motif is None for pair in runner._pairs))


@unittest.skipIf(InferenceRunner is None, "inference dependencies are not installed")
class ParseMotifTest(unittest.TestCase):
    def test_accepted_formats(self):
        parse = InferenceRunner._parse_motif
        self.assertEqual(parse("10,11,12"), [10, 11, 12])
        self.assertEqual(parse("[10, 11]"), [10, 11])
        self.assertEqual(parse(7), [7])
        self.assertEqual(parse("[-3,5]"), [-3, 5])
        self.assertIsNone(parse(None))
        self.assertIsNone(parse(float("nan")))
        self.assertIsNone(parse(""))

    def test_insertion_code_maps_to_residue_number(self):
        parse = InferenceRunner._parse_motif
        self.assertEqual(parse("[32,95C]"), [32, 95])
        self.assertEqual(parse("[35H,393,96H]"), [35, 393, 96])

    def test_unparsable_motif_warns_instead_of_vanishing_silently(self):
        import contextlib
        import io
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = InferenceRunner._parse_motif("[A, B]")
        self.assertIsNone(result)
        self.assertIn("[A, B]", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
