"""Inference without an atom cap: per-chain padding, batch size 1 and source/target of different sizes.

`models` is imported before `datasets` to follow the package's import order.
"""
import os
import sys
import tempfile
import types
import unittest

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "aligner_dl"))

import models  # noqa: F401
from datasets.scannet_dataset import ScanNetDataset
from models.loc_align import LocAlign
from models.recycling import RecyclingModule


def padded_length(
    max_atoms: int | None,
    min_atoms: int,
    n_atoms: int,
) -> int:
    """Call _padded_length without building a dataset."""
    holder = types.SimpleNamespace(_max_atoms=max_atoms, _min_atoms=min_atoms)
    return ScanNetDataset._padded_length(holder, n_atoms)


class PaddedLengthTest(unittest.TestCase):
    def test_capped_chains_are_padded_to_the_cap(self):
        self.assertEqual(padded_length(2000, 0, 500), 2000)
        self.assertEqual(padded_length(2000, 0, 3000), 2000)

    def test_uncapped_chains_keep_their_own_size(self):
        self.assertEqual(padded_length(None, 0, 12345), 12345)

    def test_uncapped_small_chains_are_padded_to_the_minimum(self):
        self.assertEqual(padded_length(None, 1001, 400), 1001)
        self.assertEqual(padded_length(None, 1001, 5000), 5000)

    def test_no_cap_requires_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "pairs.csv")
            pd.DataFrame([{"ligand_id": "ATP", "cath_degree": 1}]).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                ScanNetDataset(df_path=path, max_length=None, inference=False)
            ScanNetDataset(df_path=path, max_length=None, inference=True)


class RecyclingSizesTest(unittest.TestCase):
    def test_source_and_target_scores_have_their_own_sizes(self):
        torch.manual_seed(0)
        src = torch.randn(1, 30, 3) * 5
        tgt = torch.randn(1, 50, 3) * 5
        indices = torch.stack([torch.randint(0, 50, (1, 8)), torch.randint(0, 30, (1, 8))], dim=-1)
        values = torch.rand(1, 8)
        tgt_features, src_features, tgt_scalar, src_scalar = RecyclingModule(recycle_scalar=True)(src, tgt, values, indices)
        self.assertEqual(tgt_scalar.shape, (1, 50))
        self.assertEqual(src_scalar.shape, (1, 30))
        self.assertEqual(tgt_features.shape[1], 50)
        self.assertEqual(src_features.shape[1], 30)

    def test_thin_svd_gives_the_same_frame_as_the_full_svd(self):
        torch.manual_seed(0)
        coordinates = torch.randn(1, 200, 3) * 5 + 30.0
        padded = torch.cat([coordinates, torch.zeros(1, 100, 3)], dim=1)
        frame = RecyclingModule()._build_tarerence_frame(padded)
        centred = padded - padded.mean(dim=-2, keepdim=True)
        _, _, full_vh = torch.linalg.svd(centred)
        self.assertTrue(torch.allclose(frame[:, 1:], torch.swapaxes(full_vh, -2, -1), atol=1e-5))


def tiny_model() -> LocAlign:
    return LocAlign(
        loss=None,
        optimizer=None,
        metric=None,
        input_layer={"name": "EmbeddingBlock", "args": {"n_blocks": 1, "input_dim": 16, "hidden_dim": 8, "output_dim": 8, "norm_in_last_layer": True}},
        keypoints_selection={"name": "KeypointsSelection", "args": {"embedding_size": 8, "n_rbf_functions": 4, "top_k": 20}},
        denoiser={"name": "CDM", "args": {"n_nodes": 10, "n_gnn_layers": 2, "n_rbf_functions": 4, "with_angles": True}},
        n_iter_recycling=2,
    ).eval()


def tiny_batch(
    n_src: int,
    n_tar: int,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    batch = {}
    for key, n in (("src", n_src), ("tar", n_tar)):
        batch[f"{key}_pretrained_embeddings"] = torch.randn(1, n, 16)
        batch[f"{key}_frames"] = torch.randn(1, n, 4, 3) * 3
        batch[f"{key}_neighbors"] = torch.randint(0, n, (1, n, 6))
        batch[f"{key}_mask"] = torch.ones(1, n, dtype=torch.bool)
        batch[f"{key}_residue_indices"] = torch.arange(n).unsqueeze(0)
        batch[f"{key}_atom_original_indices"] = torch.arange(n).unsqueeze(0)
        batch[f"{key}_atom_types"] = torch.zeros(1, n, dtype=torch.long)
    return batch


class DifferentSizesTest(unittest.TestCase):
    def test_model_runs_with_source_and_target_of_different_sizes(self):
        model = tiny_model()
        for n_src, n_tar in ((40, 70), (70, 40), (25, 25)):
            with self.subTest(n_src=n_src, n_tar=n_tar), torch.no_grad():
                outputs, values, residues, atoms, types, _ = model._run_step(tiny_batch(n_src, n_tar), return_correspondences=True)
            self.assertEqual(len(outputs), 3)
            self.assertTrue(torch.isfinite(outputs[-1]["pred_R"]).all())
            self.assertTrue(torch.isfinite(values).all())
            self.assertLess(int(residues[..., 0].max()), n_src)
            self.assertLess(int(residues[..., 1].max()), n_tar)


if __name__ == "__main__":
    unittest.main()
