# List of paths to folders and binaries. All folder paths should finish with slash (/)

# Paths required for prediction.
pipeline_folder = library_folder = 'ScanNet_mini/' # Where the Github Repo is located.
structures_folder = library_folder + 'PDB/' # Where pdb/mmCIF structures files are stored.
predictions_folder = library_folder + 'predictions/' # Output folder.
model_folder = library_folder + 'models/' # Where the networks as stored as pairs of files (.h5,.data).

# Additional paths required for prediction with evolutionary information. Can use either hhblits or mmseqs (switch the homology_search flag in preprocessing.pipelines.ScanNetPipeline)
MSA_folder = library_folder + 'MSA/' # Where multiple sequence alignments are stored.
path2hhblits = 'none' # Path to hhblits binary. Not required if using ScanNet_noMSA networks.
path2sequence_database = 'none' # Path to sequence database Not required if using ScanNet_noMSA networks. Example:

path2mmseqs = 'none' # Path to mmseqs binary. Not required if using ScanNet_noMSA networks. Example: '/opt/anaconda3/bin/mmseqs'
mmseqsdatabase = 'none' # Choice of sequence database for mmseqs. Not required if using ScanNet_noMSA networks.
path2mmseqsdatabases = 'none' # Path to mmseqs sequence database. Not required if using ScanNet_noMSA networks. Example: '/Users/jerometubiana/sequence_databases/'
path2mmseqstmp = 'none' # Path to mmseqs temporary folder. Not required if using ScanNet_noMSA networks. Example: '/Users/jerometubiana/tmp/'


# Additional paths for reproducing baselines.
path_to_dssp = 'none' # Path to dssp binary. Only for reproducing handcrafted features baseline performance.
path_to_msms = 'none' # Path to msms binary. Only for reproducing handcrafted features baseline performance.

path_to_multiprot = 'none'  # Path to multiprot executable. Only relevant for homology baseline.
homology_folder = 'none'  # Where files are stored for homology baseline.