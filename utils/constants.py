RESULTS_COLUMNS = ['Ligand_ID', 'ref_protein', 'mov_protein', 'ref_chain', 'mov_chain',
                    'n_transformations','rotations', 'translations', 'rmse', 'coverage', 'cath_degree',
                     'ref_ligand_n_atoms', 'mov_ligand_n_atoms', "failure message"]

NOT_ENOUGH_ATOMS_MESSAGE = "One of the ligands has less than 3 atoms"
LIGAND_RESIDUE_IS_MISSED_MESSAGE = "Could not find ligand residue for the given chain"
TOO_MUCH_RESIDUES_MESSAGE = "Too much residues to compute for a single pair"
N_ATOMS_RATIO_MESSAGE = "The lengths of the lignads differ significantly. The length ratio exceeds 1.2"
LIGAND_OVERLAP_MESSAGE = "The lignad atoms lack sufficient overlap, with less than 80% of the smaller one having corresponding atoms in the longer one."