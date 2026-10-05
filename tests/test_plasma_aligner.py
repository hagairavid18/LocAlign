"""PlasmaAligner keeps the output of every call private, so pairs of one ligand cannot see each other's results."""
import importlib.util
import os
import sys
import tempfile
import textwrap
import types
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "aligner_dl"))

spec = importlib.util.spec_from_file_location("plasma_aligner", os.path.join(REPO, "miners", "aligners", "plasma_aligner.py"))
plasma_aligner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plasma_aligner)

STUB = textwrap.dedent("""
    import json, os, sys
    import numpy as np

    args = sys.argv[1:]
    output_dir = args[args.index("--output-dir") + 1]
    if os.path.basename(args[0]).startswith("silent"):
        sys.exit(0)
    scale = float(len(os.path.basename(args[0])))
    np.save(os.path.join(output_dir, "rotation_matrix.npy"), np.eye(3) * scale)
    np.save(os.path.join(output_dir, "translation_vector.npy"), np.full(3, scale))
    json.dump({"weighted_rmsd": scale}, open(os.path.join(output_dir, "metadata.json"), "w"))
""")


def protein(
    name: str,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(_pdb_name=name, _chain_id="A")


class PlasmaAlignerTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        script = os.path.join(self.directory.name, "stub_plasma.py")
        with open(script, "w") as f:
            f.write(STUB)
        self.aligner = plasma_aligner.PlasmaAligner()
        self.aligner.PLASMA_SCRIPT = script
        self.aligner._extract_and_cache_esm_embeddings = lambda *args: os.path.join(self.directory.name, "embedding.pt")
        self.ligand_dir = os.path.join(self.directory.name, "ligand")
        os.makedirs(self.ligand_dir)

    def tearDown(self):
        self.directory.cleanup()

    def test_each_call_returns_its_own_transformation(self):
        short = self.aligner.impose_structure(protein("t1"), protein("ab"), self.ligand_dir)
        long = self.aligner.impose_structure(protein("t1"), protein("abcdef"), self.ligand_dir)
        self.assertTrue(np.allclose(short[0][0], np.eye(3) * len("abA_non_ligand_.ent")))
        self.assertTrue(np.allclose(long[0][0], np.eye(3) * len("abcdefA_non_ligand_.ent")))

    def test_a_call_that_writes_nothing_does_not_return_the_previous_result(self):
        self.aligner.impose_structure(protein("t1"), protein("ab"), self.ligand_dir)
        R, t, corr_rmsd, _ = self.aligner.impose_structure(protein("t1"), protein("silent"), self.ligand_dir)
        self.assertEqual(R, [])
        self.assertEqual(t, [])
        self.assertIsNone(corr_rmsd)

    def test_the_private_folder_is_removed(self):
        self.aligner.impose_structure(protein("t1"), protein("ab"), self.ligand_dir)
        self.assertEqual(os.listdir(self.ligand_dir), [])


class PlasmaDispatchTest(unittest.TestCase):
    """find_protein_transformations passes PlasmaAligner the proteins and the ligand folder, like Dali and APoc."""

    def test_plasma_is_called_with_proteins_and_ligand_dir(self):
        sys.path.insert(0, os.path.join(REPO, "miners"))
        try:
            from miners.objects.protein import Protein
            from miners.objects.protein_pair import ProteinPair
        except ImportError as error:
            self.skipTest(f"miners dependencies are not installed: {error}")
        calls = []

        class StubAligner:
            name = "PlasmaAligner"

            def impose_structure(self, tar_protein, src_protein, ligand_dir):
                calls.append(ligand_dir)
                return [np.eye(4)], [np.zeros(3)], 0.5, []

        pair = types.SimpleNamespace(
            _tar_protein=types.SimpleNamespace(get_model=lambda *args: None),
            _src_protein=types.SimpleNamespace(get_model=lambda *args: None),
            _tar_model_idx=0, _src_model_idx=0, _ligand_dir="/ligands", _ligand_name="ATP",
            _save_transformed_models=False, _compute_ligand_rmsd=lambda R, t: 1.25,
        )
        original = Protein.get_residue_data
        Protein.get_residue_data = staticmethod(lambda chain: (np.zeros((3, 3)), "AAA", []))
        try:
            holder = types.SimpleNamespace()
            ProteinPair.find_protein_transformations(pair, holder, StubAligner())
        finally:
            Protein.get_residue_data = original
        self.assertEqual(calls, ["/ligands/ATP"])
        self.assertEqual(holder.PlasmaAligner_ligand_rmsd, 1.25)


if __name__ == "__main__":
    unittest.main()
