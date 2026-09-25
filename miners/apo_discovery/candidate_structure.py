import gzip
import logging
import os
import pickle
import subprocess
import uuid

import numpy as np
from Bio.PDB import MMCIFParser
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Residue import Residue
from scipy.spatial.distance import cdist

from miners.apo_discovery.constants import SYNTHETIC_CHAIN_ID
from miners.objects.protein import Protein, is_PDB_identifier

logger = logging.getLogger(__name__)


def download_raw_structure(pdb_id: str, cache_dir: str) -> Model | None:
    """Ligand-agnostic download+parse+cache of a raw PDB entry, for screening apo
    candidates before any ligand is known to exist in them. Mirrors
    Protein._init_structure's download/cache logic (files.rcsb.org + wget) but
    returns None on failure instead of an empty Structure, since callers here need
    to skip the candidate rather than proceed with it.

    Downloads and cache writes go through a per-call temp path + atomic os.replace,
    NOT a shared fixed path like Protein._init_structure uses - this function is
    called from many parallel workers that frequently target the SAME candidate
    pdb_id (many different reference chains can resolve to the same popular apo
    structure), so a fixed shared path would let concurrent writers corrupt each
    other's partial file. Atomic rename means any reader only ever sees either no
    file or a complete one.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{pdb_id}.pkl.gz")

    if os.path.exists(cache_file):
        try:
            with gzip.open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cached structure {pdb_id}: {e}")

    if not is_PDB_identifier(pdb_id):
        logger.error(f"{pdb_id} is not a valid PDB identifier")
        return None

    tmp_suffix = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    cif_path = os.path.join(cache_dir, f".tmp_{pdb_id}_{tmp_suffix}.cif")
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    try:
        subprocess.run(["wget", url, "-O", cif_path, "--quiet"], check=True, timeout=60)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error(f"Failed to download {pdb_id}.cif from {url}: {e}")
        _remove_if_exists(cif_path)
        return None

    try:
        structure = MMCIFParser().get_structure(pdb_id, cif_path)
    except Exception as e:
        logger.error(f"Failed to parse downloaded structure {pdb_id}: {e}")
        return None
    finally:
        _remove_if_exists(cif_path)

    try:
        tmp_cache_file = f"{cache_file}.tmp_{tmp_suffix}"
        with gzip.open(tmp_cache_file, "wb") as f:
            pickle.dump(structure, f)
        os.replace(tmp_cache_file, cache_file)  # atomic on the same filesystem
    except Exception as e:
        logger.warning(f"Failed to cache structure {pdb_id}: {e}")

    return structure[0]  # first model


def _remove_if_exists(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def get_heteroatom_coords(model: Model, exclude_resnames: set[str]) -> np.ndarray:
    """Heavy-atom coordinates of every heteroatom residue in `model` (all chains),
    excluding residues whose name is in `exclude_resnames`."""
    coords = []
    for chain in model:
        for residue in chain:
            if residue.id[0] == " ":  # standard AA/NA residue, not a heteroatom
                continue
            if residue.resname.strip() in exclude_resnames:
                continue
            for atom in residue.get_atoms():
                if atom.element != "H":
                    coords.append(atom.coord)
    return np.array(coords) if coords else np.empty((0, 3))


def is_pocket_empty(
    candidate_model: Model,
    transformed_ligand_coords: np.ndarray,
    exclude_resnames: set[str],
    distance_thresh: float = 4.0,
) -> bool:
    """True if no heteroatom (other than water/excluded ions) in candidate_model
    lies within distance_thresh of any transformed reference-ligand atom."""
    het_coords = get_heteroatom_coords(candidate_model, exclude_resnames)
    if het_coords.shape[0] == 0:
        return True
    dists = cdist(het_coords, transformed_ligand_coords)
    return bool(dists.min() >= distance_thresh)


def impose_apo_on_holo(holo_chain, apo_chain, aligner) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
    """TM-align apo onto holo, returning (R, t, rmsd) such that transforming
    HOLO-frame coordinates (e.g. the holo ligand) by R,t maps them into the APO frame.
    Follows the same fix/src convention as ProteinPair.find_protein_transformations
    (fix_points = the frame we're moving things INTO)."""
    apo_coord, apo_seq, _ = Protein.get_residue_data(apo_chain)
    holo_coord, holo_seq, _ = Protein.get_residue_data(holo_chain)
    R_list, t_list, rmsd, _ = aligner.impose_structure(apo_coord, holo_coord, apo_seq, holo_seq)
    if not R_list:
        return None, None, None
    return R_list[0], t_list[0], rmsd


def transplant_ligand(
    apo_chain: Chain,
    holo_ligand_residue: Residue,
    R: np.ndarray,
    t: np.ndarray,
    ligand_resname: str,
    output_chain_id: str = SYNTHETIC_CHAIN_ID,
) -> Model:
    """Build a fresh, minimal single-chain Model from apo_chain (renamed to
    output_chain_id) plus a new hetero residue holding holo_ligand_residue's atoms
    transformed by R,t. Rebuilding from scratch (rather than deep-copying the whole
    apo_model and keeping its original chain id) sidesteps two real problems:
    mmCIF chain ids can be >1 character (e.g. "A0"), which legacy-format PDBIO
    rejects outright, and the full biological assembly can carry many redundant
    chains we don't need - only the one aligned chain matters downstream."""
    new_chain = Chain(output_chain_id)
    for residue in apo_chain:
        # Only standard amino acids: matches Protein.create_non_ligand_model's own
        # filter, so incidental heteroatoms (waters, ions, other bound molecules)
        # get dropped here rather than surviving into a file that will just have
        # them stripped downstream anyway. This also sidesteps a real crash: some
        # candidates carry a modern 5-character PDB component code (e.g. "A1H2K"),
        # which silently overflows legacy PDB format's fixed-width resName column
        # and corrupts every field after it when the file is re-parsed.
        if not is_aa(residue, standard=False):
            continue
        new_residue = Residue(residue.id, residue.resname, residue.get_segid())
        for atom in residue.get_atoms():
            new_residue.add(atom.copy())
        new_chain.add(new_residue)

    existing_resseqs = [res.id[1] for res in new_chain]
    new_resseq = (max(existing_resseqs) if existing_resseqs else 0) + 1000
    ligand_residue = Residue(
        (f"H_{ligand_resname[:3]}", new_resseq, " "),
        ligand_resname[:3],
        holo_ligand_residue.get_segid(),
    )
    for atom in holo_ligand_residue.get_atoms():
        coord = np.dot(atom.coord, R[:3, :3]) + t[:3]
        new_atom = Atom(
            atom.get_name(), coord, atom.get_bfactor(), atom.get_occupancy(),
            atom.get_altloc(), atom.get_fullname(), atom.get_serial_number(), atom.element,
        )
        ligand_residue.add(new_atom)
    new_chain.add(ligand_residue)

    new_model = Model(0)
    new_model.add(new_chain)
    return new_model


def save_model_to_pdb(model: Model, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    io = PDBIO()
    io.set_structure(model)
    io.save(out_path)
