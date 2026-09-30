# Water and monoatomic-ion residue names, deliberately narrow: for the apo binding-site
# occupancy check (reviewer comment #1), only these are excluded when deciding whether a
# candidate structure's pocket is "empty". Crystallization additives (PEG, glycerol, sulfate,
# etc. - see aligner_dl.utils.constants.CRYSTALIZATION_LIGANDS) are NOT excluded here: they
# still count as occupying the pocket, unlike in the primary-dataset ligand filter.
APO_POCKET_EXCLUDED_RESNAMES = {
    "HOH", "WAT", "DOD", "D8O",  # water forms
    "NA", "K", "LI", "CS", "RB",  # alkali metals
    "MG", "CA", "SR", "BA",  # alkaline earth metals
    "MN", "MN3", "FE", "FE2", "FE3", "CO", "NI", "NI2", "CU", "CU1", "CU3", "ZN", "ZN2", "CD", "HG",  # transition metals
    "AG", "AU", "AU3", "PT", "PD", "PB", "AL", "GA", "TL",  # other metals
    "LA", "CE", "SM", "EU", "EU3", "GD", "GD3", "TB", "YB", "LU",  # lanthanides
    "CL", "BR", "F", "IOD",  # monoatomic anions
}

PDBE_UNIPROT_MAPPING_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
PDBE_BEST_STRUCTURES_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{accession}"

DEFAULT_MIN_OVERLAP = 0.8
DEFAULT_POCKET_DISTANCE_THRESH = 4.0

# Chain id used for every saved synthetic apo(+transplanted ligand) PDB file. A
# candidate's real (mmCIF) chain id can be >1 character (e.g. "A0"), which the
# legacy PDB format PDBIO writes cannot represent - transplant_ligand always
# renames its output chain to this single character instead.
SYNTHETIC_CHAIN_ID = "A"
