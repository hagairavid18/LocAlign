import pickle, gzip
import os
import warnings
import numpy as np
import sys
sys.path.append(os.path.join(os.getcwd(), 'ScanNet_mini'))
from ScanNet_mini.preprocessing import PDBio, PDB_processing
import Bio.PDB
from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.filterwarnings("ignore", category=PDBConstructionWarning)


amino_acid_3to_1 = {
    'ALA': 'A',
    'CYS': 'C',
    'ASP': 'D',
    'GLU': 'E',
    'PHE': 'F',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LYS': 'K',
    'LEU': 'L',
    'MET': 'M',
    'ASN': 'N',
    'PRO': 'P',
    'GLN': 'Q',
    'ARG': 'R',
    'SER': 'S',
    'THR': 'T',
    'VAL': 'V',
    'TRP': 'W',
    'TYR': 'Y'
}

amino_acid_atom_order = {

    'A': ['N', 'CA', 'C', 'O', 'CB'],  # Alanine

    'C': ['N', 'CA', 'C', 'O', 'CB', 'SG'],  # Cysteine

    'D': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'],  # Aspartic Acid

    'E': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],  # Glutamic Acid

    'F': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],  # Phenylalanine

    'G': ['N', 'CA', 'C', 'O'],  # Glycine (no side chain)

    'H': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],  # Histidine

    'I': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'],  # Isoleucine

    'K': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],  # Lysine

    'L': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],  # Leucine

    'M': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],  # Methionine

    'N': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'],  # Asparagine

    'P': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],  # Proline

    'Q': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'],  # Glutamine

    'R': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],  # Arginine

    'S': ['N', 'CA', 'C', 'O', 'CB', 'OG'],  # Serine

    'T': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'],  # Threonine

    'V': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'],  # Valine

    'W': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],  # Tryptophan

    'Y': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH']  # Tyrosine

}
class SelectChain_without_ligand(Bio.PDB.Select):
    def __init__(self,selected_chains,excluded_ligand, mode='without_ligand', *args, **kwargs):
        self.selected_chains = selected_chains
        self.excluded_ligand = excluded_ligand
        self.mode = mode
        super().__init__(*args,**kwargs)
    def accept_model(self,model):
        if self.selected_chains == 'all':
            return 1
        elif model.id in [x[0] for x in self.selected_chains]:
            return 1
        else:
            return 0
    def accept_chain(self, chain):
        if self.selected_chains == 'all':
            return 1            
        elif (chain.get_full_id()[1],chain.get_full_id()[2]) in self.selected_chains:
            return 1
        else:
            return 0
    def accept_residue(self,residue):
        if self.mode == 'without_ligand':
            return int(residue.get_resname() != self.excluded_ligand)
        elif self.mode == 'only_ligand':
            return int(residue.get_resname() == self.excluded_ligand)
        else:
            raise ValueError(self.mode)

def extract_chains_andor_ligand_and_apply_transform(struct, chain_ids, ligand_id,final_file,mode='without_ligand'):
            
    with warnings.catch_warnings(record=True) as w:
        
        
        
        io = Bio.PDB.PDBIO()
        io.set_structure(struct)
        io.save(final_file, SelectChain_without_ligand(chain_ids,ligand_id, mode=mode))
    return final_file
        
        

def get_pocket(receptor_file,ligand_file):
    _,chains1 = PDBio.load_chains(receptor_file, verbose=False)
    _,chains2 = PDBio.load_chains(ligand_file, verbose=False)


    _, _, receptor_atom_coordinates, _, _ = PDB_processing.process_chain(chains1)
    
    ligand_atoms = Bio.PDB.Selection.unfold_entities(chains2,'A')
    ligand_atoms_coordinates = []
    for ligand_atom in ligand_atoms:
        if not ligand_atom.name == 'H':
            ligand_atoms_coordinates.append(ligand_atom.get_coord())
    ligand_atoms_coordinates = np.array(ligand_atoms_coordinates)
    
    in_contact = []
    for receptor_res in range(len(receptor_atom_coordinates)):    
        distances = np.sqrt( ((receptor_atom_coordinates[receptor_res][np.newaxis] - ligand_atoms_coordinates[:,np.newaxis])**2).sum(-1) )
        in_contact.append( distances.min() < 4)
    in_contact = np.array(in_contact)
        
    receptor_PDB_indices = PDB_processing.get_PDB_indices(chains1,return_model=False,return_chain=True)
    pocket_residues = np.array(receptor_PDB_indices)[in_contact]
    return pocket_residues
    

def make_pseudo_bond_files(
    output_file,
    template_receptor_file,
    template_ligand_file,
    transformed_query_receptor_file,
    transformed_query_ligand_file,
):
    _,template_receptor_chains = PDBio.load_chains(template_receptor_file)
    _,template_ligand_chains = PDBio.load_chains(template_ligand_file)
    _,transformed_query_receptor_chains = PDBio.load_chains(transformed_query_receptor_file)
    _,transformed_query_ligand_chains = PDBio.load_chains(transformed_query_ligand_file)
    
    
    template_ligand_atoms = Bio.PDB.Selection.unfold_entities(template_ligand_chains,'A')
    template_ligand_atoms_coordinates = []
    for ligand_atom in template_ligand_atoms:
        if not ligand_atom.name == 'H':
            template_ligand_atoms_coordinates.append(ligand_atom.get_coord())
    template_ligand_atoms_coordinates = np.array(template_ligand_atoms_coordinates)
    
    transformed_query_ligand_atoms = Bio.PDB.Selection.unfold_entities(transformed_query_ligand_chains,'A')
    transformed_query_ligand_atoms_coordinates = []
    for ligand_atom in transformed_query_ligand_atoms:
        if not ligand_atom.name == 'H':
            transformed_query_ligand_atoms_coordinates.append(ligand_atom.get_coord())
    transformed_query_ligand_atoms_coordinates = np.array(transformed_query_ligand_atoms_coordinates)
    
    template_receptor_atoms = Bio.PDB.Selection.unfold_entities(template_receptor_chains,'A')    
    template_pocket_atoms = []
    for template_atom in template_receptor_atoms:
        if template_atom.get_full_id()[-2][0] != ' ':
            continue
        coord = template_atom.get_coord()
        min_distance =  np.sqrt( ( (template_ligand_atoms_coordinates - coord)**2).sum(-1).min(0) )
        if min_distance < 4:
            template_pocket_atoms.append(template_atom)
    
    transformed_query_receptor_atoms = Bio.PDB.Selection.unfold_entities(transformed_query_receptor_chains,'A')    
    transformed_query_pocket_atoms = []
    for transformed_query_atom in transformed_query_receptor_atoms:
        if transformed_query_atom.get_full_id()[-2][0] != ' ':
            continue        
        coord = transformed_query_atom.get_coord()
        min_distance =  np.sqrt( ( (transformed_query_ligand_atoms_coordinates - coord)**2).sum(-1).min(0) )
        if min_distance < 4:
            transformed_query_pocket_atoms.append(transformed_query_atom)
                
    
    template_pocket_coords = np.array([atom.get_coord() for atom in template_pocket_atoms])
    transformed_query_pocket_coords = np.array([atom.get_coord() for atom in transformed_query_pocket_atoms])

    template_pocket_type = np.array([atom.get_id()[0] for atom in template_pocket_atoms])
    transformed_query_pocket_type = np.array([atom.get_id()[0] for atom in transformed_query_pocket_atoms])

    
    distances =  np.sqrt(  ( (template_pocket_coords[:,np.newaxis] - transformed_query_pocket_coords[np.newaxis])**2).sum(-1) )
    
    best_buddy_pairs = ( (distances == distances.min(0)[np.newaxis]) & (distances == distances.min(1)[:,np.newaxis]) )
    below_cutoff = (distances <= 2)
    same_type = (template_pocket_type[:,np.newaxis] == transformed_query_pocket_type[np.newaxis,:])
    
    corresponding_pairs = best_buddy_pairs & below_cutoff & same_type
    num_correspondences = corresponding_pairs.sum()
    ids_template,ids_query = np.nonzero(corresponding_pairs)
    
    print(f'Total number of template pocket atoms: {len(template_pocket_atoms)}')
    print(f'Total number of query pocket atoms: {len(transformed_query_pocket_atoms)}')
    print(f'Total number of correspondences: {num_correspondences}' )
    
    
    lines = [
    '; halfbond = false',
    '; color = black',
    '; radius = 0.25',
    '; dashes = 0'
    ]

    template_corr_atoms,query_corr_atoms = [],[]     
    for n in range(num_correspondences):
        template_atom = template_pocket_atoms[ids_template[n]].get_full_id()
        query_atom = transformed_query_pocket_atoms[ids_query[n]].get_full_id()        
        lines.append(f"#1/{template_atom[-3]}:{template_atom[-2][1]}@{template_atom[-1][0]} #2/{query_atom[-3]}:{query_atom[-2][1]}@{query_atom[-1][0]}")        
        template_corr_atoms.append(f'#1/{template_atom[-3]}:{template_atom[-2][1]}@{template_atom[-1][0]}')
        query_corr_atoms.append(f'#2/{query_atom[-3]}:{query_atom[-2][1]}@{query_atom[-1][0]}')
    
    template_corr_atoms = ' '.join(template_corr_atoms)    
    query_corr_atoms = ' '.join(query_corr_atoms)
    
    with open(output_file,'w') as f:
        for line in lines:
            f.write(line + '\n')    
    return output_file,template_corr_atoms,query_corr_atoms
    
    
def make_pseudo_bond_file_from_residue_indices(
    output_file: str,
    template_receptor_file: str,
    transformed_query_receptor_file: str,
    corr_residue_indices: np.ndarray,  # shape: (N, 2)
    corr_values: np.ndarray,           # shape: (N,)
    atom_indexes_list: np.ndarray | None = None
):
    """
    Create Chimera pseudobond file from residue correspondences.

    Parameters:
        output_file: str – path to save the pseudo bond file
        template_receptor_file: str – path to the template receptor PDB file
        transformed_query_receptor_file: str – path to the transformed query receptor PDB file
        corr_residue_indices: np.ndarray – Nx2 array of corresponding residue indices (template_idx, query_idx)
        corr_values: np.ndarray – Nx1 array of correspondence scores (same order)
    """
    parser = Bio.PDB.PDBParser(QUIET=True)

    template_structure = parser.get_structure("template", template_receptor_file)
    query_structure = parser.get_structure("query", transformed_query_receptor_file)

    template_chain = list(template_structure.get_chains())[0]
    query_chain = list(query_structure.get_chains())[0]

    lines = [
        "; halfbond = false",
        "; color = black",
        "; radius = 0.25",
        "; dashes = 0"
    ]
    def get_residue_by_number(chain, resnum):
        for residue in chain:
            het, rseq, icode = residue.get_id()
            if het.strip() == '' and rseq == resnum and icode.strip() == '':
                return residue
        return None
    
    template_corr_atoms, query_corr_atoms = [],[]

    for (query_idx, template_idx), score, (query_atom_index, template_atom_index) in zip(corr_residue_indices, corr_values, atom_indexes_list):
        try:
            template_residue = get_residue_by_number(template_chain, template_idx)
            query_residue = get_residue_by_number(query_chain, query_idx)

            # Pick CA atoms (or fallback to first atom)
            template_atom_type = amino_acid_atom_order[amino_acid_3to_1.get(template_residue.get_resname())][template_atom_index]
            query_atom_type = amino_acid_atom_order[amino_acid_3to_1.get(query_residue.get_resname())][query_atom_index]
            template_atom = template_residue[template_atom_type]
            query_atom = query_residue[query_atom_type]

            t_chain_id = template_atom.get_parent().get_parent().id  # Correct chain ID
            q_chain_id = query_atom.get_parent().get_parent().id

            t_atom_name = template_atom.get_name()
            q_atom_name = query_atom.get_name()
            
            # Compute distance between atoms
            t_coord = template_atom.get_coord()
            q_coord = query_atom.get_coord()
            distance = np.linalg.norm(t_coord - q_coord)

            # Only plot if distance is <= 1 Å
            if distance <= 2.0:
                lines.append(f"#1/{t_chain_id}:{template_idx}@{t_atom_name} #2/{q_chain_id}:{query_idx}@{q_atom_name}")
                
                template_corr_atoms.append(f'#1/{t_chain_id}:{template_idx}@{t_atom_name}')
                query_corr_atoms.append(f'#2/{q_chain_id}:{query_idx}@{q_atom_name}')

        except:
            print(f"Skipping correspondence ({template_idx}, {query_idx}) - index out of bounds")
            continue
        
    template_corr_atoms = ' '.join(template_corr_atoms)
    query_corr_atoms = ' '.join(query_corr_atoms)

    with open(output_file, 'w') as f:
        for line in lines:
            f.write(line + "\n")

    # print(f"Saved {len(corr_residue_indices)} pseudobonds to {output_file}")
    return output_file, template_corr_atoms, query_corr_atoms


def make_chimera_script(
                        output_folder,
                        ligand=None,
                        template_pocket_residues=None,
                        query_pocket_residues=None,
                        template_corr_atoms = None,
                        query_corr_atoms = None,
                        version = 'pocket',
                        ):
    
        
    assert version in ['pocket','motif','global']
    # Version pocket: Highlights the ligand and the pocket.
    # Version motif: Highlights the ligand, if present, and the learned alignment.
    
    show_ligand = ligand not in  [None,'general']
    show_template_ligand = show_ligand & os.path.exists(os.path.join(output_folder,'template_ligand.pdb') )
    show_query_ligand = show_ligand & os.path.exists(os.path.join(output_folder,'transformed_query_ligand.pdb'))
    # Use case where we want to show both ligands: finding common structural motif.
    # Use case where want to show neither: catalytic sites or unknown ligand, etc.
    # Use case where we want to show the template ligand, but query is unavailable: comparing unbound query against database of templates with bound ligands.
    # Last use case: a priori not needed.
    
    if version == 'pocket':    
        colors_and_transparency = {
            'template': {            
                'receptor': ('cornflower blue', 90),
                'ligand': ('dark blue',0),
                'pocket': ('blue',0),
                'keypoints': ('cyan',0)
            },
            
            'query': {            
                'receptor': ('orange red', 90),
                'ligand': ('dark red',0),
                'pocket': ('red',0),
                'keypoints': ('orange',0)
            }            
        }        
    elif version == 'motif':
        colors_and_transparency = {
            'template': {            
                'receptor': ('cornflower blue', 85),
                'ligand': ('dark blue',50),
                'keypoints': ('cyan',0)
            },
            
            'query': {            
                'receptor': ('orange red', 85),
                'ligand': ('dark red',50),
                'keypoints': ('orange',0)
            }
        }    
    elif version == 'global':
        colors_and_transparency = {
            'template': {            
                'receptor': ('cornflower blue', 30),
                'ligand': ('green',0),
                'keypoints': ('blue',0)
            },
            
            'query': {            
                'receptor': ('salmon', 30),
                'ligand': ('green',0),
                'keypoints': ('red',0)
            }
        }            

            
    show_template_pocket =  (template_pocket_residues is not None) & (version == 'pocket')
    show_query_pocket =  (query_pocket_residues is not None) & (version == 'pocket')
    
    show_template_keypoints = (template_corr_atoms is not None)
    show_query_keypoints = (query_corr_atoms is not None)
    
    
    model_ranks = {
        'template_receptor': 1,
        'query_receptor':2,
    }
    current_rank = 3
    

    list_commands = []
    
    
    for file in ['template_receptor','transformed_query_receptor']:
        list_commands.append( f'open {file}.pdb' )
        
    if show_template_ligand:
        list_commands.append( f'open template_ligand.pdb' )
        model_ranks['template_ligand'] = current_rank
        current_rank +=1
    if show_query_ligand:
        list_commands.append( f'open transformed_query_ligand.pdb' )
        model_ranks['query_ligand'] = current_rank
        current_rank +=1


    list_commands.append(f'dssp')
    list_commands.append(f"sel #{model_ranks['template_receptor']}")
    list_commands.append(f'hide sel atoms')
    list_commands.append(f"color sel {colors_and_transparency['template']['receptor'][0]} transparency {colors_and_transparency['template']['receptor'][1]}")
    
    list_commands.append(f"sel #{model_ranks['query_receptor']}")
    list_commands.append(f'hide sel atoms')
    list_commands.append(f"color sel {colors_and_transparency['query']['receptor'][0]} transparency {colors_and_transparency['query']['receptor'][1]}")
    
    if show_template_pocket:        
        template_pocket_residues_chimera_formatted = []
        for chain in np.unique(template_pocket_residues[:,0]):
            subset = (template_pocket_residues[:,0] == chain)
            indices = template_pocket_residues[subset,1]
            template_pocket_residues_chimera_formatted.append(f"#{model_ranks['template_receptor']}/{chain}:" + ','.join(indices))
        list_commands.append(f'sel ' + '| '.join(template_pocket_residues_chimera_formatted))
        list_commands.append(f'show sel atoms')
        list_commands.append(f'style sel stick')
        list_commands.append(f"color sel {colors_and_transparency['template']['pocket'][0]} transparency {colors_and_transparency['template']['pocket'][1]}")
        
    if show_query_pocket:
        query_pocket_residues_chimera_formatted = []
        for chain in np.unique(query_pocket_residues[:,0]):
            subset = (query_pocket_residues[:,0] == chain)
            indices = query_pocket_residues[subset,1]
            query_pocket_residues_chimera_formatted.append(f"#{model_ranks['query_receptor']}/{chain}:" + ','.join(indices))
        list_commands.append(f'sel ' + '| '.join(query_pocket_residues_chimera_formatted))
        list_commands.append(f'show sel atoms')
        list_commands.append(f'style sel stick')
        list_commands.append(f"color sel {colors_and_transparency['query']['pocket'][0]} transparency {colors_and_transparency['query']['pocket'][1]}")
        
    if show_template_ligand:
        list_commands.append(f"sel #{model_ranks['template_ligand']}")
        list_commands.append(f'show sel atoms')
        list_commands.append(f'style sel stick')
        list_commands.append(f"color sel {colors_and_transparency['template']['ligand'][0]} transparency {colors_and_transparency['template']['ligand'][1]}")        
        if not version == 'global':
            list_commands.append(f'color sel byhetero')
        
    if show_query_ligand:
        list_commands.append(f"sel #{model_ranks['query_ligand']}")
        list_commands.append(f'show sel atoms')
        list_commands.append(f'style sel stick')
        list_commands.append(f"color sel {colors_and_transparency['query']['ligand'][0]} transparency {colors_and_transparency['query']['ligand'][1]}")        
        if not version == 'global':        
            list_commands.append(f'color sel byhetero')
        
    
    if show_template_keypoints:            
        template_corr_residues = ' '.join( x.split('@')[0] for x in template_corr_atoms.split(' ') )
        list_commands.append(f"sel {template_corr_residues}")
        list_commands.append(f"color sel {colors_and_transparency['template']['keypoints'][0]} transparency {colors_and_transparency['template']['keypoints'][1]}")        
        if not version == 'global':        
            list_commands.append('show sel atoms')
            list_commands.append('hide sel cartoon')
            list_commands.append('style sel stick')
            list_commands.append(f'color sel byhetero')        
            list_commands.append(f"sel {template_corr_atoms}")
            list_commands.append('style sel ball')
        
    if show_query_keypoints:                
        query_corr_residues = ' '.join( x.split('@')[0] for x in query_corr_atoms.split(' ') )    
        list_commands.append(f'sel {query_corr_residues}')
        list_commands.append(f"color sel {colors_and_transparency['query']['keypoints'][0]} transparency {colors_and_transparency['query']['keypoints'][1]}")        
        if not version == 'global':
            list_commands.append('show sel atoms')
            list_commands.append('hide sel cartoon')
            list_commands.append('style sel stick')  
            list_commands.append(f'color sel byhetero')        
            list_commands.append(f"sel {query_corr_atoms}")
            list_commands.append('style sel ball')

    if (version == 'motif') & show_query_keypoints & show_template_keypoints:
        list_commands.append(f'sel {query_corr_atoms} {template_corr_atoms}')
        list_commands.append(f'view sel')
    
        
    list_commands.append('sel clear')
    list_commands.append('open correspondences.pb')

    # Common ions that might be ligands
    common_ions = ['ZN', 'MG', 'CA', 'FE', 'MN', 'CU', 'CO', 'NI', 'K', 'NA', 'CL', 'BR', 'I', 'F', 'FES']
    
    # Don't hide solvent if ligand is a solvent molecule or an ion
    if ligand not in common_ions:
        list_commands.append('hide solvent')
    
    # Apply spherical view if ligand is an ion
    if show_ligand and ligand in common_ions:
        ligand_models = []
        if show_template_ligand:
            ligand_models.append(f"#{model_ranks['template_ligand']}")
        if show_query_ligand:
            ligand_models.append(f"#{model_ranks['query_ligand']}")
        if ligand_models:
            list_commands.append(f"sel {' | '.join(ligand_models)}")
            list_commands.append('style sel sphere')
    
    list_commands.append('lighting soft')
    list_commands.append('set bgColor white')
    

    chimera_file = os.path.join(output_folder,f'chimera_script_{version}.cxc')
    with open(chimera_file,'w') as f:
        for command in list_commands:
            f.write(command + '\n')
    return chimera_file
    
def process_alignment(
    base_folder: str, 
    template: str, 
    template_ligand: str, 
    query: str,
    cache_dir: str,
    query_ligand: str | None = None,
    query_transformation: tuple[np.ndarray, np.ndarray] | None = None,
    corr_values: np.ndarray | None = None,
    corr_indices: np.ndarray | None = None,
    atom_indexes_list: np.ndarray | None = None
    ):
    """
    Process protein alignment and create visualization files for Chimera.
    
    Args:
        base_folder: Base folder for output files
        template: Template protein identifier
        template_ligand: Ligand identifier for the template
        query: Query protein identifier
        query_ligand: Ligand identifier for the query (defaults to template_ligand when None)
        scannet_dir: Directory containing ScanNet features
        query_transformation: Tuple of (R, t) for transformation
        corr_values: Correspondence values
        corr_indices: Correspondence indices
        atom_indexes_list: Atom index mappings
    """
    # Compute atom indexes if scannet_dir is provided and atom_indexes_list not provided
    corr_indices_atom = atom_indexes_list
    query_ligand = query_ligand or template_ligand

    tar_scannet_path = os.path.join(cache_dir, "scannet_embeddings", f"{template}_scannet_atoms.pkl")
    src_scannet_path = os.path.join(cache_dir, "scannet_embeddings", f"{query}_scannet_atoms.pkl")

    # Load tar and src scannet features
    with gzip.open(tar_scannet_path, 'rb') as f:
        tar_scannet = pickle.load(f)
    with gzip.open(src_scannet_path, 'rb') as f:
        src_scannet = pickle.load(f)
    
    atom_indexes_list = np.zeros((corr_indices_atom.shape[0], 2), dtype=int)
    for i in range(corr_indices.shape[0]):
        tar_res_idx = tar_scannet['sequence_indices_atom'][corr_indices_atom[i, 1]]
        src_res_idx = src_scannet['sequence_indices_atom'][corr_indices_atom[i, 0]]

        tar_atom_list = tar_scannet['aa_to_atom_indices'][tar_res_idx]
        src_atom_list = src_scannet['aa_to_atom_indices'][src_res_idx]

        tar_atom_index = np.where(tar_atom_list == corr_indices_atom[i, 1])[0][0]
        src_atom_index = np.where(src_atom_list == corr_indices_atom[i, 0])[0][0]

        atom_indexes_list[i, 0] = src_atom_index
        atom_indexes_list[i, 1] = tar_atom_index

    # Get parent parent folder
    pdb_folder = os.path.join(cache_dir, "pdb_files")

    folder = base_folder
    output_folder = folder
    os.makedirs(output_folder, exist_ok = True)

    template_chain_id = [(0, template[-1])]
    
    query_chain_id = [(0, query[-1])]

    parser = Bio.PDB.PDBParser(QUIET=True)
    

    template_non_ligand_struct = parser.get_structure('name', f"{pdb_folder}/{template_ligand}/{template}_non_ligand_.ent")

    extract_chains_andor_ligand_and_apply_transform(template_non_ligand_struct, template_chain_id, template_ligand,
                                os.path.join(output_folder, 'template_receptor.pdb')
                                ,mode='without_ligand')

    template_only_ligand_struct = parser.get_structure('name', f"{pdb_folder}/{template_ligand}/{template}_ligand.pdb")

    extract_chains_andor_ligand_and_apply_transform(template_only_ligand_struct, template_chain_id, template_ligand,
                                os.path.join(output_folder, 'template_ligand.pdb')
                                ,mode='only_ligand')

    query_non_ligand_struct = parser.get_structure('name', f"{pdb_folder}/{query_ligand}/{query}_non_ligand_.ent")
    rot,tran = query_transformation
    for atom in Bio.PDB.Selection.unfold_entities(query_non_ligand_struct,'A'):
        atom.set_coord( np.dot(atom.get_coord(), rot) + tran)

    query_only_ligand_struct = parser.get_structure('name', f"{pdb_folder}/{query_ligand}/{query}_ligand.pdb")
    rot,tran = query_transformation
    for atom in Bio.PDB.Selection.unfold_entities(query_only_ligand_struct,'A'):
        atom.set_coord( np.dot(atom.get_coord(), rot) + tran)

    extract_chains_andor_ligand_and_apply_transform(query_non_ligand_struct, query_chain_id, query_ligand,
                                os.path.join(output_folder, 'transformed_query_receptor.pdb')
                                ,mode='without_ligand')

    extract_chains_andor_ligand_and_apply_transform(query_only_ligand_struct, query_chain_id, query_ligand,
                                os.path.join(output_folder, 'transformed_query_ligand.pdb')
                                ,mode='only_ligand')

    ligand_tag = template_ligand if template_ligand == query_ligand else f"{template_ligand}_src-{query_ligand}"

    if template_ligand != 'general' and query_ligand != 'general':
        template_pocket_residues = get_pocket( os.path.join(output_folder, 'template_receptor.pdb'),
                            os.path.join(output_folder, 'template_ligand.pdb') )

        query_pocket_residues = get_pocket( os.path.join(output_folder, 'transformed_query_receptor.pdb'),
                            os.path.join(output_folder, 'transformed_query_ligand.pdb') )
    else:
        template_pocket_residues = None
        query_pocket_residues = None

    if corr_indices is not None: # In case we have correspondences from the model output
        _,template_corr_atoms,query_corr_atoms = make_pseudo_bond_file_from_residue_indices(
            os.path.join(output_folder, 'correspondences.pb'),
            os.path.join(output_folder, 'template_receptor.pdb'),
            os.path.join(output_folder, 'transformed_query_receptor.pdb'),
            corr_residue_indices=corr_indices,  # shape: (N, 2)
            corr_values= corr_values,  # shape: (N,
            atom_indexes_list=atom_indexes_list
        )
    else:
        _,template_corr_atoms,query_corr_atoms = make_pseudo_bond_files(
            os.path.join(output_folder, 'correspondences.pb'),    
            os.path.join(output_folder, 'template_receptor.pdb'),
            os.path.join(output_folder, 'template_ligand.pdb'),
            os.path.join(output_folder, 'transformed_query_receptor.pdb'),
            os.path.join(output_folder, 'transformed_query_ligand.pdb'),                         
        )

    for version in ['pocket','motif','global']:
        make_chimera_script(
                            output_folder,
                            ligand=ligand_tag,
                            template_pocket_residues=template_pocket_residues,
                            query_pocket_residues=query_pocket_residues,
                            template_corr_atoms = template_corr_atoms,
                            query_corr_atoms = query_corr_atoms,
                            version = version,
                            ) 

    return    

            

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process protein alignment and visualize pockets.")
    parser.add_argument("--base_folder", type=str, required=True, help="Base folder for output files.")
    parser.add_argument("--scannet_dir", type=str, default=None, help="Directory containing ScanNet features.")
    parser.add_argument("--model_output_path", type=str, default=None, help="Path to model output for correspondences.")
    parser.add_argument("--template", type=str, required=True, help="Template protein identifier.")
    parser.add_argument("--template_ligand", type=str, required=True, help="Ligand identifier for the template.")
    parser.add_argument("--query", type=str, required=True, help="Query protein identifier.")
    parser.add_argument("--query_ligand", type=str, default=None, help="Ligand identifier for the query (defaults to template_ligand).")
    
    args = parser.parse_args()

    # Call process_alignment with all arguments from command line
    process_alignment(
        base_folder=args.base_folder,
        template=args.template,
        template_ligand=args.template_ligand,
        query=args.query,
        query_ligand=args.query_ligand,
        scannet_dir=args.scannet_dir,
        model_output_path=args.model_output_path
    )
