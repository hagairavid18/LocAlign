import pickle
import os,sys,warnings
import numpy as np
path2libraries = '/home/iscb/wolfson/hagairavid/ScanNet_Ub'
sys.path.append(path2libraries)
from preprocessing import PDBio, PDB_processing
import Bio.PDB

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

def extract_chains_andor_ligand_and_apply_transform(file, chain_ids, ligand_id,final_file,mode='without_ligand',
                                                    transformation=None):
    class SelectChain_without_ligand(Bio.PDB.Select):
        def __init__(self,selected_chains,excluded_ligand,*args,**kwargs):
            self.selected_chains = selected_chains
            self.excluded_ligand = excluded_ligand
            return super().__init__(*args,**kwargs)
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
            if mode == 'without_ligand':
                return int(residue.get_resname() != self.excluded_ligand)
            elif mode == 'only_ligand':
                return int(residue.get_resname() == self.excluded_ligand)
            else:
                raise ValueError(mode)
            
    with warnings.catch_warnings(record=True) as w:
        if file[-4:] == '.cif':
            parser = Bio.PDB.MMCIFParser()
        else:
            parser = Bio.PDB.PDBParser()
        struct = parser.get_structure('name',file)
        if transformation is not None:
            rot,tran = transformation
            for atom in Bio.PDB.Selection.unfold_entities(struct,'A'):
                atom.set_coord( np.dot(atom.get_coord(), rot) + tran)
        
        
        io = Bio.PDB.PDBIO()
        io.set_structure(struct)
        io.save(final_file, SelectChain_without_ligand(chain_ids,ligand_id))
    return final_file
        
        

def get_pocket(receptor_file,ligand_file):
    _,chains1 = PDBio.load_chains(receptor_file)
    _,chains2 = PDBio.load_chains(ligand_file)
        
    
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

    template_corr_residues,query_corr_residues = [],[]     
    for n in range(num_correspondences):
        template_atom = template_pocket_atoms[ids_template[n]].get_full_id()
        query_atom = transformed_query_pocket_atoms[ids_query[n]].get_full_id()        
        lines.append(f"#1/{template_atom[-3]}:{template_atom[-2][1]}@{template_atom[-1][0]} #2/{query_atom[-3]}:{query_atom[-2][1]}@{query_atom[-1][0]}")        
        template_corr_residues.append(f'#1/{template_atom[-3]}:{template_atom[-2][1]}')
        query_corr_residues.append(f'#2/{query_atom[-3]}:{query_atom[-2][1]}')
    
    template_corr_residues = ' '.join(template_corr_residues)    
    query_corr_residues = ' '.join(query_corr_residues)
    
    with open(output_file,'w') as f:
        for line in lines:
            f.write(line + '\n')    
    return output_file,template_corr_residues,query_corr_residues
    
import numpy as np
from Bio.PDB import PDBParser
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
    parser = PDBParser(QUIET=True)

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
    
    template_corr_residues,query_corr_residues = [],[] 

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

            lines.append(f"#1/{t_chain_id}:{template_idx}@{t_atom_name} #2/{q_chain_id}:{query_idx}@{q_atom_name}")
            
            template_corr_residues.append(f'#1/{t_chain_id}:{template_idx}')
            query_corr_residues.append(f'#2/{q_chain_id}:{query_idx}@{q_atom_name}')

        except:
            print(f"Skipping correspondence ({template_idx}, {query_idx}) - index out of bounds")
            continue
        
    template_corr_residues = ' '.join(template_corr_residues)
    query_corr_residues = ' '.join(query_corr_residues)

    with open(output_file, 'w') as f:
        for line in lines:
            f.write(line + "\n")

    print(f"Saved {len(corr_residue_indices)} pseudobonds to {output_file}")
    return output_file,template_corr_residues,query_corr_residues

    
def process_alignment(
        base_folder: str, 
        template: str, 
        template_ligand: str, 
        query: str, 
        query_transformation: tuple[np.ndarray, np.ndarray] | None = None,
        corr_values: np.ndarray | None = None,
        corr_indices: np.ndarray | None = None,
        atom_indexes_list: np.ndarray | None = None
        ):
    query_ligand = template_ligand

    folder = base_folder
    # query = os.path.join(folder, f'RANSACAlligner_0_protein_0_0.pdb')
    output_folder = folder
    os.makedirs(output_folder, exist_ok = True)

    template_file, template_chain_id = PDBio.getPDB(template[:-1] + '_' + template[-1], biounit=False)
    query_file,query_chain_id = PDBio.getPDB(query[:-1] + '_' + query[-1], biounit=False)

    extract_chains_andor_ligand_and_apply_transform(template_file, template_chain_id, template_ligand,
                                os.path.join(output_folder, 'template_receptor.pdb')
                                ,mode='without_ligand',transformation=None)

    extract_chains_andor_ligand_and_apply_transform(template_file, template_chain_id, template_ligand,
                                os.path.join(output_folder, 'template_ligand.pdb')
                                ,mode='only_ligand',transformation=None)

    extract_chains_andor_ligand_and_apply_transform(query_file, query_chain_id, query_ligand,
                                os.path.join(output_folder, 'transformed_query_receptor.pdb')
                                ,mode='without_ligand',transformation=query_transformation)

    extract_chains_andor_ligand_and_apply_transform(query_file, query_chain_id, query_ligand,
                                os.path.join(output_folder, 'transformed_query_ligand.pdb')
                                ,mode='only_ligand',transformation=query_transformation)

    if query_ligand != 'general':
        template_pocket_residues = get_pocket( os.path.join(output_folder, 'template_receptor.pdb'),
                            os.path.join(output_folder, 'template_ligand.pdb') )

        query_pocket_residues = get_pocket( os.path.join(output_folder, 'transformed_query_receptor.pdb'),
                            os.path.join(output_folder, 'transformed_query_ligand.pdb') )
    else:
        template_pocket_residues = None
        query_pocket_residues = None

    if corr_indices is not None: # In case we have correspondences from the model output
        _,template_corr_residues,query_corr_residues = make_pseudo_bond_file_from_residue_indices(
            os.path.join(output_folder, 'correspondences.pb'),
            os.path.join(output_folder, 'template_receptor.pdb'),
            os.path.join(output_folder, 'transformed_query_receptor.pdb'),
            corr_residue_indices=corr_indices,  # shape: (N, 2)
            corr_values= corr_values,  # shape: (N,
            atom_indexes_list=atom_indexes_list
        )
    else:
        _,template_corr_residues,query_corr_residues = make_pseudo_bond_files(
            os.path.join(output_folder, 'correspondences.pb'),    
            os.path.join(output_folder, 'template_receptor.pdb'),
            os.path.join(output_folder, 'template_ligand.pdb'),
            os.path.join(output_folder, 'transformed_query_receptor.pdb'),
            os.path.join(output_folder, 'transformed_query_ligand.pdb'),                         
        )


    list_commands = []
    
    for file in ['template_receptor','transformed_query_receptor']:
        list_commands.append( f'open {file}.pdb' )
    
    if query_ligand != 'general':
        for file in ['template_ligand','transformed_query_ligand']:
            list_commands.append( f'open {file}.pdb' )
        list_commands.append(f'dssp')
        list_commands.append(f'sel #1')
        list_commands.append(f'hide sel atoms')
        list_commands.append(f'color sel cornflower blue transparency 90')

        if template_pocket_residues is not None:
            template_pocket_residues_chimera_formatted = []
            for chain in np.unique(template_pocket_residues[:,0]):
                subset = (template_pocket_residues[:,0] == chain)
                indices = template_pocket_residues[subset,1]
                template_pocket_residues_chimera_formatted.append(f'#1/{chain}:' + ','.join(indices))
            list_commands.append(f'sel ' + '| '.join(template_pocket_residues_chimera_formatted))
            list_commands.append(f'show sel atoms')
            list_commands.append(f'style sel stick')
            list_commands.append(f'color sel blue transparency 0')

        list_commands.append(f'sel #3')

        list_commands.append(f'show sel atoms')
        list_commands.append(f'style sel stick')
        list_commands.append(f'color sel cyan transparency 0')


        list_commands.append(f'sel #2')
        list_commands.append(f'hide sel atoms')
        list_commands.append(f'color sel orange red transparency 90')

        query_pocket_residues_chimera_formatted = []
        for chain in np.unique(query_pocket_residues[:,0]):
            subset = (query_pocket_residues[:,0] == chain)
            indices = query_pocket_residues[subset,1]
            query_pocket_residues_chimera_formatted.append(f'#2/{chain}:' + ','.join(indices))
        list_commands.append(f'sel ' + '| '.join(query_pocket_residues_chimera_formatted))
        list_commands.append(f'show sel atoms')
        list_commands.append(f'style sel stick')
        list_commands.append(f'color sel red transparency 0')
        # list_commands.append(f'color sel byhetero')


        list_commands.append(f'sel #4')
        list_commands.append(f'show sel atoms')
        list_commands.append(f'style sel stick')
        list_commands.append(f'color sel orange transparency 0')
        list_commands.append(f'color sel byhetero')
        list_commands.append('sel clear')
    list_commands.append('hide solvent')
    list_commands.append('lighting soft')
    list_commands.append('set bgColor white')
    
    list_commands.append(f'sel {template_corr_residues}')
    list_commands.append('color sel dark blue transparency 50') 
    list_commands.append('show sel atoms')
    list_commands.append('hide sel cartoon')
    list_commands.append('style sel ball')
    
    list_commands.append(f'sel {query_corr_residues}')
    list_commands.append('color sel dark red transparency 50') 
    list_commands.append('show sel atoms')
    list_commands.append('hide sel cartoon')
    list_commands.append('style sel ball')  
    list_commands.append('sel clear')    
    list_commands.append('open correspondences.pb')
    # for file in ['template_receptor','transformed_query_receptor']:
    #         list_commands.append( f'open {file}.pdb' )
    # list_commands.append("sel #6")
    # list_commands.append("color sel blue")
    # list_commands.append("sel clear")
    # list_commands.append("sel #7")
    # list_commands.append("color sel red")
    # list_commands.append("sel clear")

    chimera_file = os.path.join(output_folder,'chimera_script.cxc')
    with open(chimera_file,'w') as f:
        for command in list_commands:
            f.write(command + '\n')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process protein alignment and visualize pockets.")
    parser.add_argument("--base_folder", type=str, required=True, help="Base folder for output files.")
    parser.add_argument("--scannet_dir", type=str, default=None, help="Directory containing ScanNet features.")
    parser.add_argument("--model_output_path", type=str, default=None, help="Path to model output for correspondences.")
    parser.add_argument("--template", type=str, required=True, help="Template protein identifier.")
    parser.add_argument("--template_ligand", type=str, required=True, help="Ligand identifier for the template.")
    parser.add_argument("--query", type=str, required=True, help="Query protein identifier.")
    
    args = parser.parse_args()

    
            
    ref_scannet_path = os.path.join(args.scannet_dir, args.template_ligand, f"{args.template}_scannet_atoms.pkl")
    mov_scannet_path = os.path.join(args.scannet_dir, args.template_ligand, f"{args.query}_scannet_atoms.pkl")
    
    # load ref and mov scannet features
    with open(ref_scannet_path, 'rb') as f:
        ref_scannet = pickle.load(f)
    with open(mov_scannet_path, 'rb') as f:
        mov_scannet = pickle.load(f)
    
    if args.model_output_path:
        output_dict = np.load(args.model_output_path, allow_pickle=True)
        corr_values = output_dict.get('top_corr_values', None)
        corr_indices = output_dict.get('top_corr_indices', None)
        corr_indices_atom = output_dict.get('top_corr_indices_atom', None)
        atom_indexes_list = np.zeros((corr_indices_atom.shape[0],2),dtype=int)
        for i in range(corr_indices.shape[0]):
            ref_res_idx = ref_scannet['sequence_indices_atom'][corr_indices_atom[i,1]]
            mov_res_idx = mov_scannet['sequence_indices_atom'][corr_indices_atom[i,0]]

            ref_atom_list = ref_scannet['aa_to_atom_indices'][ref_res_idx]
            mov_atom_list = mov_scannet['aa_to_atom_indices'][mov_res_idx]

            ref_atom_index = np.where(ref_atom_list == corr_indices_atom[i,1])[0][0]
            mov_atom_index = np.where(mov_atom_list == corr_indices_atom[i,0])[0][0]

            atom_indexes_list[i,0] = mov_atom_index
            atom_indexes_list[i,1] = ref_atom_index

        R = output_dict['R']
        t = output_dict['t']
        process_alignment(
            args.base_folder, 
            args.template, 
            args.template_ligand, 
            args.query,
            query_transformation=(R, t),
            corr_values=corr_values,
            corr_indices=corr_indices,
            atom_indexes_list=atom_indexes_list
            )
    else:
        process_alignment(args.base_folder, args.template, args.template_ligand, args.query)
