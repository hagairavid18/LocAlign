import hashlib
import logging
import os
from dataclasses import dataclass, field

import numpy as np

from miners.aligners.tm_aligner import TMAligner
from miners.apo_discovery.candidate_structure import (
    download_raw_structure,
    impose_apo_on_holo,
    is_pocket_empty,
    save_model_to_pdb,
    transplant_ligand,
)
from miners.apo_discovery.constants import (
    APO_POCKET_EXCLUDED_RESNAMES,
    DEFAULT_MIN_OVERLAP,
    DEFAULT_POCKET_DISTANCE_THRESH,
    SYNTHETIC_CHAIN_ID,
)
from miners.apo_discovery.pdbe_client import get_best_structures, get_uniprot_accessions_for_chain

logger = logging.getLogger(__name__)


@dataclass
class ApoResult:
    ref_pdb_id: str
    ref_chain_id: str
    ligand_id: str
    apo_pdb_id: str | None = None
    apo_chain_id: str | None = None
    apo_original_chain_id: str | None = None
    apo_resolution: float | None = None
    tm_rmsd: float | None = None
    local_pdb_path: str | None = None
    synthetic_id: str | None = None
    status: str = "not_found"
    n_candidates_checked: int = 0
    rejection_log: list[str] = field(default_factory=list)


def _overlap_fraction(ref_start: int, ref_end: int, cand_start: int, cand_end: int) -> float:
    ref_len = ref_end - ref_start + 1
    if ref_len <= 0:
        return 0.0
    overlap = min(ref_end, cand_end) - max(ref_start, cand_start) + 1
    return max(0.0, overlap) / ref_len


def find_apo_structure(
    ref_pdb_id: str,
    ref_chain_id: str,
    ligand_id: str,
    holo_ligand_residue,
    holo_chain,
    *,
    structure_save_dir: str,
    pdbe_cache_dir: str,
    structure_cache_dir: str,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    distance_thresh: float = DEFAULT_POCKET_DISTANCE_THRESH,
    exclude_resnames: set[str] = APO_POCKET_EXCLUDED_RESNAMES,
    max_tm_rmsd: float = 15.0,
    aligner=None,
) -> ApoResult:
    """Find a genuine apo structure for (ref_pdb_id, ref_chain_id) with its bound
    ligand's pocket unoccupied, transplant the (transformed) ligand onto it, and save
    the result. See the approved plan at
    /home/iscb/wolfson/hagairavid/.claude/plans/gleaming-greeting-stallman.md for the
    full design rationale.
    """
    ref_pdb_id = ref_pdb_id.lower()
    result = ApoResult(ref_pdb_id=ref_pdb_id, ref_chain_id=ref_chain_id, ligand_id=ligand_id)
    aligner = aligner or TMAligner()

    ref_mappings = get_uniprot_accessions_for_chain(ref_pdb_id, ref_chain_id, pdbe_cache_dir)
    if not ref_mappings:
        result.status = "no_uniprot_mapping"
        return result

    candidates = []
    for mapping in ref_mappings:
        for cand in get_best_structures(mapping["accession"], pdbe_cache_dir):
            if cand.get("resolution") is None:
                continue
            if cand.get("pdb_id", "").lower() == ref_pdb_id:
                continue
            overlap = _overlap_fraction(
                mapping["unp_start"], mapping["unp_end"],
                cand.get("unp_start", 0), cand.get("unp_end", 0),
            )
            if overlap < min_overlap:
                continue
            candidates.append(cand)

    if not candidates:
        result.status = "no_candidate_passed_prefilter"
        return result

    candidates.sort(key=lambda c: c["resolution"])

    ref_ligand_coords = np.array([
        atom.coord for atom in holo_ligand_residue.get_atoms() if atom.element != "H"
    ])

    for cand in candidates:
        result.n_candidates_checked += 1
        cand_pdb_id = cand["pdb_id"].lower()
        cand_chain_id = cand["chain_id"]

        cand_model = download_raw_structure(cand_pdb_id, structure_cache_dir)
        if cand_model is None:
            result.rejection_log.append(f"{cand_pdb_id}{cand_chain_id}: download_failed")
            continue

        try:
            apo_chain = cand_model[cand_chain_id]
        except KeyError:
            result.rejection_log.append(f"{cand_pdb_id}{cand_chain_id}: chain_not_found")
            continue

        R, t, tm_rmsd = impose_apo_on_holo(holo_chain, apo_chain, aligner)
        if R is None or tm_rmsd is None or tm_rmsd > max_tm_rmsd:
            result.rejection_log.append(f"{cand_pdb_id}{cand_chain_id}: bad_alignment(rmsd={tm_rmsd})")
            continue

        transformed_ligand_coords = np.dot(ref_ligand_coords, R[:3, :3]) + t[:3]
        if not is_pocket_empty(cand_model, transformed_ligand_coords, exclude_resnames, distance_thresh):
            result.rejection_log.append(f"{cand_pdb_id}{cand_chain_id}: pocket_occupied")
            continue

        try:
            synthetic_id = "apo" + hashlib.sha1(
                f"{ref_pdb_id}{ref_chain_id}->{cand_pdb_id}{cand_chain_id}".encode()
            ).hexdigest()[:10]
            transplanted_model = transplant_ligand(apo_chain, holo_ligand_residue, R, t, ligand_id)
            local_pdb_path = os.path.join(structure_save_dir, ligand_id, f"{synthetic_id}.pdb")
            save_model_to_pdb(transplanted_model, local_pdb_path)
        except Exception as e:
            # One malformed candidate (e.g. an unexpected atom/residue shape) must
            # not abort the whole batch - log it and try the next candidate instead.
            result.rejection_log.append(f"{cand_pdb_id}{cand_chain_id}: transplant_failed({e})")
            continue

        result.status = "found"
        result.apo_pdb_id = cand_pdb_id
        result.apo_chain_id = SYNTHETIC_CHAIN_ID
        result.apo_original_chain_id = cand_chain_id
        result.apo_resolution = cand["resolution"]
        result.tm_rmsd = tm_rmsd
        result.local_pdb_path = local_pdb_path
        result.synthetic_id = synthetic_id
        return result

    result.status = "all_candidates_pocket_occupied"
    return result
