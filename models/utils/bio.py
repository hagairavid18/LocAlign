
import numpy as np

def get_residue_data(chain: Chain) -> tuple[np.ndarray, str, list[int]]:
        """
        Extracts CA coordinates, residue sequence, and residue indices from a chain.

        Args:
            chain (Bio.PDB.Chain): A chain from a PDB structure.

        Returns:
            tuple[np.ndarray, str, list[int]]: A tuple containing:
                - A NumPy array of CA atom coordinates.
                - A string of the protein sequence derived from the residues.
                - A list of residue indices corresponding to the coordinates.
        """
        coords = []
        seq = []
        residue_indices = []
        
        for residue in chain.get_residues():
            if "CA" in residue.child_dict and residue.resname in protein_letters_3to1:
                coords.append(residue.child_dict["CA"].coord)
                seq.append(protein_letters_3to1[residue.resname])
                residue_indices.append(residue.id[1])  # Extract residue index

        return np.vstack(coords), "".join(seq), residue_indices