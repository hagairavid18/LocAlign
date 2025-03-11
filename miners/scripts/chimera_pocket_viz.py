import os,sys,warnings
import numpy as np
path2libraries = '/home/iscb/wolfson/hagairavid/ScanNet_Ub'
sys.path.append(path2libraries)
from preprocessing import PDBio, PDB_processing
import Bio.PDB


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
    '; color = yellow',
    '; radius = 0.2',
    '; dashes = 3'
    ]
    
    for n in range(num_correspondences):
        template_atom = template_pocket_atoms[ids_template[n]].get_full_id()
        query_atom = transformed_query_pocket_atoms[ids_query[n]].get_full_id()        
        lines.append(f"#1/{template_atom[-3]}:{template_atom[-2][1]}@{template_atom[-1][0]} #3/{query_atom[-3]}:{query_atom[-2][1]}@{query_atom[-1][0]}")        
    
    with open(output_file,'w') as f:
        for line in lines:
            f.write(line + '\n')    
    return output_file
    
    
    
    
    
    
# cath_degree = '6'
# template = '1fdj_A'
# template_ligand = '2FP'
# query = '4ald_A'
# query_ligand = template_ligand

# cath_degree = '6'
# template = '1q78_A'
# template_ligand = '3AT'
# query = '4lt6_A'
# query_ligand = template_ligand

# cath_degree = '3'
# template = '1suw_A'
# template_ligand = 'NAP'
# query = '2jl1_A'
# query_ligand = template_ligand

# cath_degree = '3'
# template = '2gn4_A'
# template_ligand = 'UD1'
# query = '4wad_A'
# query_ligand = template_ligand

# cath_degree = '3'
# template = '1vpe_A'
# template_ligand = 'ANP'
# query = '3b7g_A'
# query_ligand = template_ligand

# cath_degree = '3'
# template = '3l92_A'
# template_ligand = 'COA'
# query = '5frd_A'
# query_ligand = template_ligand

# cath_degree = '3'
# template = '1ry2_A'
# template_ligand = 'AMP'
# query = '3uk2_A'
# query_ligand = template_ligand

# cath_degree = '3'
# template = '1qf6_A'
# template_ligand = 'AMP'
# query = '3iuy_A'
# query_ligand = template_ligand
def process_alignment(base_folder: str, template, template_ligand, query, query_transformation = None):
    query_ligand = template_ligand

    folder = base_folder
    query = os.path.join(folder, f'RANSACAlligner_0_protein_0_0.pdb')
    output_folder = folder
    os.makedirs(output_folder, exist_ok = True)

    template_file,template_chain_id = PDBio.getPDB(template,biounit=False)
    query_file,query_chain_id = PDBio.getPDB(query,biounit=False)

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


    template_pocket_residues = get_pocket( os.path.join(output_folder, 'template_receptor.pdb'),
                        os.path.join(output_folder, 'template_ligand.pdb') )

    query_pocket_residues = get_pocket( os.path.join(output_folder, 'transformed_query_receptor.pdb'),
                        os.path.join(output_folder, 'transformed_query_ligand.pdb') )


    make_pseudo_bond_files(
        os.path.join(output_folder, 'correspondences.pb'),    
        os.path.join(output_folder, 'template_receptor.pdb'),
        os.path.join(output_folder, 'template_ligand.pdb'),
        os.path.join(output_folder, 'transformed_query_receptor.pdb'),
        os.path.join(output_folder, 'transformed_query_ligand.pdb'),                         
    )


    list_commands = []
    for file in ['template_receptor','template_ligand','transformed_query_receptor','transformed_query_ligand']:
        list_commands.append( f'open {file}.pdb' )
    list_commands.append(f'dssp')
    list_commands.append(f'sel #1')
    list_commands.append(f'hide sel atoms')
    list_commands.append(f'color sel cornflower blue transparency 90')

    template_pocket_residues_chimera_formatted = []
    for chain in np.unique(template_pocket_residues[:,0]):
        subset = (template_pocket_residues[:,0] == chain)
        indices = template_pocket_residues[subset,1]
        template_pocket_residues_chimera_formatted.append(f'#1/{chain}:' + ','.join(indices))
    list_commands.append(f'sel ' + '| '.join(template_pocket_residues_chimera_formatted))
    list_commands.append(f'show sel atoms')
    list_commands.append(f'style sel stick')
    list_commands.append(f'color sel blue transparency 0')
    # list_commands.append(f'color sel byhetero')

    list_commands.append(f'sel #2')

    list_commands.append(f'show sel atoms')
    list_commands.append(f'style sel stick')
    list_commands.append(f'color sel cyan transparency 0')
    # list_commands.append(f'color sel byhetero')


    list_commands.append(f'sel #3')
    list_commands.append(f'hide sel atoms')
    list_commands.append(f'color sel orange red transparency 90')

    query_pocket_residues_chimera_formatted = []
    for chain in np.unique(query_pocket_residues[:,0]):
        subset = (query_pocket_residues[:,0] == chain)
        indices = query_pocket_residues[subset,1]
        query_pocket_residues_chimera_formatted.append(f'#3/{chain}:' + ','.join(indices))
    list_commands.append(f'sel ' + '| '.join(query_pocket_residues_chimera_formatted))
    list_commands.append(f'show sel atoms')
    list_commands.append(f'style sel stick')
    list_commands.append(f'color sel red transparency 0')
    # list_commands.append(f'color sel byhetero')


    list_commands.append(f'sel #4')
    list_commands.append(f'show sel atoms')
    list_commands.append(f'style sel stick')
    list_commands.append(f'color sel orange transparency 0')
    # list_commands.append(f'color sel byhetero')
    list_commands.append('sel clear')
    list_commands.append('hide solvent')
    list_commands.append('lighting soft')
    list_commands.append('set bgColor white')
    list_commands.append('open correspondences.pb')

    chimera_file = os.path.join(output_folder,'chimera_script.cxc')
    with open(chimera_file,'w') as f:
        for command in list_commands:
            f.write(command + '\n')