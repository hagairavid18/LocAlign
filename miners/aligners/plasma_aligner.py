"""Wrapper for the external PLASMA local aligner script.

This aligner invokes PLASMA with pre-computed ESM embeddings:

    /home/iscb/wolfson/hagairavid/PLASMA-Protein-Local-Alignment/align_pdb_chains.py \
      /path/to/pdb1.pdb A /path/to/pdb2.pdb B \
      --output-dir /path/to/output

It expects the non-ligand PDBs produced by the pipeline ("*_non_ligand_.ent") and
uses the provided ligand directory as the output directory. ESM embeddings are
extracted and cached before calling PLASMA to avoid redundant computation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
from Bio.PDB import PDBParser, PPBuilder

logger = logging.getLogger(__name__)


class PlasmaAligner:
    """Adapter to run the external PLASMA aligner with ESM embedding extraction."""

    PLASMA_SCRIPT = \
        "/home/iscb/wolfson/hagairavid/PLASMA-Protein-Local-Alignment/align_pdb_chains.py"

    def __init__(self, esm_model: str = "esm2_t33_650M_UR50D", esm_layer: int = 33) -> None:
        """Initialize PLASMA aligner with ESM model.
        
        Args:
            esm_model: Name of ESM model to use (default: esm2_t33_650M_UR50D)
            esm_layer: Layer to extract embeddings from (default: 33)
        """
        self.name = "PlasmaAligner"
        self._esm_model_name = esm_model
        self._esm_layer = esm_layer
        self._esm_model = None  # lazy to improve picklability
        self._esm_alphabet = None   # lazy to improve picklability
        self._batch_converter = None
        self._ppb_builder = None
        self._pdb_parser = None
        
        if not os.path.exists(self.PLASMA_SCRIPT):
            logger.warning("PLASMA script not found at %s", self.PLASMA_SCRIPT)
    
    def _init_esm_model(self):
        """Lazy initialization of ESM model."""
        if self._esm_model is not None:
            return
        
        import esm
        self._esm_model, self._esm_alphabet = getattr(esm.pretrained, self._esm_model_name)()
        self._batch_converter = self._esm_alphabet.get_batch_converter()
        self._esm_model = self._esm_model.eval()
        logger.debug(f"Loaded ESM model: {self._esm_model_name}")
    
    def _init_parsers(self):
        if self._pdb_parser is None:
            self._pdb_parser = PDBParser()
        if self._ppb_builder is None:
            self._ppb_builder = PPBuilder()

    def __getstate__(self):
        # Exclude non-picklable/heavy fields so we can pass aligner in mp.Pool
        state = {
            "name": self.name,
            "_esm_model_name": self._esm_model_name,
            "_esm_layer": self._esm_layer,
        }
        return state

    def __setstate__(self, state):
        self.name = state.get("name", "PlasmaAligner")
        self._esm_model_name = state.get("_esm_model_name", "esm2_t33_650M_UR50D")
        self._esm_layer = state.get("_esm_layer", 33)
        self._esm_model = None
        self._esm_alphabet = None
        self._batch_converter = None
        self._ppb_builder = None
        self._pdb_parser = None
    
    def _extract_and_cache_esm_embeddings(
        self,
        pdb_path: str,
        protein_name: str,
        chain_id: str,
        ligand_dir: str,
    ) -> str:
        """Extract ESM embeddings from PDB and save to PLASMA cache format.

        Args:
            pdb_path: Path to PDB file
            protein_name: Protein identifier (PDB ID or name)
            chain_id: Chain identifier
            ligand_dir: Directory to save embeddings

        Returns:
            Path to the saved embedding .pt file
        """
        # Create cache directory matching PLASMA's structure exactly
        cache_dir = Path(ligand_dir) / "data" / "embeddings" / "ESM2" / "AA_embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Use protein name + chain for cache key (simpler and more intuitive)
        cache_key = f"{self._esm_model_name}_{protein_name}{chain_id}"
        cache_path = cache_dir / f"{cache_key}.pt"

        # Fast path: if already cached - use it
        if cache_path.exists():
            logger.info("Using cached ESM embeddings: %s", cache_path)
            return str(cache_path)

        # Serialize concurrent writers via a simple dir lock
        lock_dir = cache_dir / (cache_key + ".lock")
        waited = 0.0
        while True:
            try:
                lock_dir.mkdir(exist_ok=False)
                break
            except FileExistsError:
                if cache_path.exists():
                    return str(cache_path)
                time.sleep(0.2)
                waited += 0.2
                if waited > 300:
                    logger.error("Timeout waiting for ESM lock %s", lock_dir)
                    # Give up; caller may decide how to proceed
                    return str(cache_path)

        try:
            # Initialize heavy deps lazily only when needed
            self._init_parsers()
            self._init_esm_model()

            # Parse structure and select chain
            structure = self._pdb_parser.get_structure("pdb", pdb_path)
            model = next(iter(structure))
            chain = None
            for ch in model.get_chains():
                if str(ch.id) == str(chain_id):
                    chain = ch
                    break
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found in {pdb_path}")

            # Build sequence
            peptides = self._ppb_builder.build_peptides(chain)
            if not peptides:
                raise ValueError(f"No peptide chains found in {pdb_path}")
            sequence = "".join(str(peptide.get_sequence()) for peptide in peptides)

            # Tokenize and run ESM
            data = [("sequence", sequence)]
            _, _, tokens = self._batch_converter(data)
            with torch.no_grad():
                results = self._esm_model(tokens, repr_layers=[self._esm_layer], return_contacts=False)
                embeddings = results["representations"][self._esm_layer][0, 1 : len(sequence) + 1].cpu()

            # Save to cache
            torch.save(embeddings, cache_path)
            logger.info("Saved ESM embeddings to %s", cache_path)
            return str(cache_path)
        finally:
            try:
                lock_dir.rmdir()
            except Exception:
                pass

    def impose_structure(self, tar_protein, src_protein, ligand_dir: str):
        """Run PLASMA on the target/source proteins with ESM embeddings.

        Extracts ESM embeddings for both proteins before calling PLASMA, then
        parses the output transformation matrices.
        """

        src_name, src_chain = src_protein._pdb_name, src_protein._chain_id
        tar_name, tar_chain = tar_protein._pdb_name, tar_protein._chain_id

        src_path = os.path.join(ligand_dir, f"{src_name}{src_chain}_non_ligand_.ent")
        tar_path = os.path.join(ligand_dir, f"{tar_name}{tar_chain}_non_ligand_.ent")

        try:
            # Extract ESM embeddings for both proteins
            logger.info(
                "Extracting ESM embeddings for %s%s and %s%s",
                src_name,
                src_chain,
                tar_name,
                tar_chain,
            )
            src_embed_path = self._extract_and_cache_esm_embeddings(src_path, src_name, src_chain, ligand_dir)
            tar_embed_path = self._extract_and_cache_esm_embeddings(tar_path, tar_name, tar_chain, ligand_dir)
        except Exception as e:
            logger.error("Failed to extract ESM embeddings: %s", e)
            return [], [], None, []

        # PLASMA aligns pdb1 to pdb2, pass src as pdb1 and tar as pdb2
        cmd = [
            sys.executable,
            self.PLASMA_SCRIPT,
            src_path,  # pdb1 (source - will be transformed)
            src_chain,  # chain1
            tar_path,  # pdb2 (target - reference)
            tar_chain,  # chain2
            "--output-dir",
            ligand_dir,
            "--embed1-path",
            src_embed_path,
            "--embed2-path",
            tar_embed_path,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("PLASMA aligner failed: %s", result.stderr)
                return [], [], None, []

            # Attempt to load rotation/translation matrices and metadata.
            rot_path = os.path.join(ligand_dir, "rotation_matrix.npy")
            trans_path = os.path.join(ligand_dir, "translation_vector.npy")
            align_path = os.path.join(ligand_dir, "soft_alignment.npy")
            meta_path = os.path.join(ligand_dir, "metadata.json")

            R: list[np.ndarray] = []
            t: list[np.ndarray] = []
            corr_rmsd: float | None = None

            if os.path.exists(rot_path) and os.path.exists(trans_path):
                try:
                    R_mat = np.load(rot_path)
                    t_vec = np.load(trans_path)
                    R = [np.array(R_mat)]
                    t = [np.array(t_vec)]
                except Exception as load_exc:  # defensive
                    logger.error("Failed to load PLASMA transforms: %s", load_exc)

            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    if "weighted_rmsd" in meta and meta["weighted_rmsd"] is not None:
                        corr_rmsd = float(meta["weighted_rmsd"])
                except Exception as load_exc:  # defensive
                    logger.error("Failed to load PLASMA metadata: %s", load_exc)

            if os.path.exists(align_path):
                try:
                    _ = np.load(align_path)
                except Exception as load_exc:  # defensive
                    logger.error("Failed to load PLASMA alignment matrix: %s", load_exc)

            return R, t, corr_rmsd, []
        except FileNotFoundError:
            logger.error("PLASMA script not found at %s", self.PLASMA_SCRIPT)
            return [], [], None, []
        except Exception as exc:  # defensive
            logger.error("Error running PLASMA aligner: %s", exc)
            return [], [], None, []
