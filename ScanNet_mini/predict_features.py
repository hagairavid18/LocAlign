import predict_bindingsites
from preprocessing import PDBio
from utilities.paths import model_folder,structures_folder,MSA_folder
import numpy as np


def predict_features(list_queries,layer='SCAN_filter_activity_aa',
                     model='ScanNet_PPI_noMSA',
                     output_format='dictionary',
                     model_folder=model_folder,
                     biounit=False,
                     permissive=False,logfile=None):
    '''
    Usages:
     list_dictionary_features = predict_features(list_queries,output_format='dictionary')
     list_features, list_residueids = predict_features(list_queries,output_format='numpy')
    Example: 
    list_queries = ['1a3x_A','2p6b_AB','1a3y']
    list_dictionary_features = list of residues-level features, each element of the form Nresidues X Nfeatures.

    '''
    if not isinstance(list_queries,list):
        list_queries = [list_queries]
        return_one = True
        permissive = False
    else:
        return_one = False
    query_pdbs = []
    query_chain_ids = []
    nlayers = len(layer) if isinstance(layer,list) else 1

    for query in list_queries:
        pdb,chain_ids = PDBio.parse_str(query)
        query_pdbs.append(pdb)
        query_chain_ids.append(chain_ids)

    

    if 'noMSA' in model:
        pipeline = predict_bindingsites.pipeline_noMSA
        use_MSA = False
    else:
        pipeline = predict_bindingsites.pipeline_MSA
        use_MSA = True

    query_outputs = predict_bindingsites.predict_interface_residues(
    query_pdbs=query_pdbs,
    query_chain_ids=query_chain_ids,
    pipeline=pipeline,
    model=model,
    model_folder=model_folder,
    structures_folder=structures_folder,
    MSA_folder=MSA_folder,
    biounit=biounit,
    assembly=True,
    layer=layer,
    use_MSA=use_MSA,
    overwrite_MSA=False,
    Lmin=1,
    output_chimera=False,
    permissive=permissive,
    output_predictions=False,
    output_format = output_format,
    logfile=logfile
    )
    if output_format == 'numpy':
        query_pdbs, query_names, query_features, query_residue_ids, query_sequences = query_outputs

        if return_one:
            query_pdbs = query_pdbs[0]
            query_names = query_names[0]
            query_features = query_features[0]
            query_residue_ids = query_residue_ids[0]
            query_sequences = query_sequences[0]
        if permissive:
            return query_pdbs,query_features, query_residue_ids
        else:
            return query_features, query_residue_ids
    elif output_format == 'dictionary':
        query_pdbs, query_names, query_dictionary_features = query_outputs
        if return_one:
            query_pdbs = query_pdbs[0]
            query_names = query_names[0]
            query_dictionary_features = query_dictionary_features[0]
        if permissive:
            return query_pdbs,query_dictionary_features
        else:
            return query_dictionary_features




if __name__ == '__main__':
    model = 'ScanNet_ubiquitin_autoregressive_config9_noMSA_30_08_1' # Protein-protein binding site prediction model without evolutionary information.

    # layer_choices = [
    #     'SCAN_filter_activity_atom',
    #     'SCAN_filters_atom_aggregated_activity', # Atomic Neighborhood Embedding Module, *after* pooling. Atomic neighborhoods have radius of about 5 Angstrom.  Size: [Naa,64].
    #     'all_embedded_attributes_aa', # Embedded residue type or PWM (first 32 channels) + Atomic Neighborhood Embedding Module, *after* pooling (last 64 channels). Size: [Naa,96].
    #     'SCAN_filter_activity_aa', # Amino Acid Neighborhood Embedding Module. Amino acid neighborhoods have radius of about 11 Angstrom. Size: [Naa,128].
    #     'SCAN_filters_aa_embedded_1', # Non-linear, 32-dimensional projection of Amino Acid Neighborhood Embedding Module output. Input to the neighborhood attention module. Size: [Naa,32].
    #     None, # The binding site probabilities Size: ([Naa,])
    # ]
    
    layer_choices = [
        'SCAN_filter_activity_atom_1_normalization',
        'SCAN_filter_activity_aa_2_normalization',
        'classifier_output'
    ]

    output_format = 'numpy' #'dictionary' # 'numpy'


    # layer = layer_choices[2]
    # layer = [layer_choices[1],layer_choices[2],layer_choices[4]] # Multiple layers are supported.
    layer = [layer_choices[0],layer_choices[-3]]


    # if output_format == 'dictionary':
    #     list_names, list_dictionary_features = predict_features(['1a3x_A','1brs_A'],layer=layer,model=model,output_format=output_format,permissive=True)
    #     print('Dictionary format: Dictionary with residue ids as key and features as items.')
    #     for k in range(2):
    #         print('Query',list_names[k])
    #         for key,item in list(list_dictionary_features[k].items())[:10]:
    #             if isinstance(item,list):
    #                 list_shapes = [x.shape for x in item]
    #                 print('AA' ,key,'Features:',[item_[:5] for item_ in item],'Feature shapes',list_shapes)
    #             else:
    #                 print('AA',key, 'Features:',item[:5],'Feature shape',item.shape)
    # elif output_format == 'numpy':
    #     list_names,list_features, list_residue_ids = predict_features(['1a3x_A','1brs_A'],layer=layer,model=model,output_format='numpy',permissive=True)
    #     print('Numpy format: Numpy arrays with residue ids as key and features as items.')
    #     for k in range(2):
    #         print('Query',list_names[k])
    #         if isinstance(list_features[k],list):
    #             for feature_ in list_features[k]:
    #                 print('Features array', feature_[:10, :][:, :5], 'Shape',feature_.shape)
    #         else:
    #             print('Features array',list_features[k][:10,:][:,:5])
    #         print('Residue IDs array',list_residue_ids[k][:10])
            
            

    '''
    This code below is for manipulating atomic-level frames, embeddings, etc.
    
    
    atomic_plus_residue_embeddings = a matrix of size [Natoms , Nfeatures_atom+Nfeatures_aa],
    
    where for each row:
    - the first Nfeatures_atom are the atomic-level embedding of the atom, 
    - the last Nfeatures_aa are the amino-acid level embedding of the amino acid to which the atom belongs to.
    
    
     atomic-level embeddings, residue-level embeddings, and concatenate them at the atomic level
    '''
    
    model = 'ScanNet_ubiquitin_autoregressive_config9_noMSA_30_08_1' # Protein-protein binding site prediction model without evolutionary information.
    # model = 'ScanNet_PPI_noMSA'
    # list_layers = [
    #     'sequence_indices_atom', # The atom to amino acid index correspondence. For each atom, the index of the amino acid it belongs to.
    #     'frames_atom', # The frames attached to each atom. For each atom, a matrix of size [4,3], where the first row is the coordinates of the center (=the translation) of the frame and the rows 2-4 are the 3 normal vectors (=the rotation matrix).
    #     'SCAN_filter_activity_atom', # The atomic-level embeddings.
    #     'SCAN_filter_activity_aa', # The amino-acid level embeddings.
    # ]
    
    list_layers = [
        'atom_to_aa_indices',
        'frames_atom',
        'aa_to_atom_indices',
        'nearest_neighbor_search_atom',
        'SCAN_filter_activity_atom_1_normalization',
        'SCAN_filter_activity_aa_2_normalization',
        'classifier_output'
    ]
    
    list_inputs = ['1a3x_A','1brs_A','1pin_0-A']
    
    output_format = 'numpy' #'dictionary' # 'numpy'
    
    list_names,list_features, list_residue_ids = predict_features(list_inputs,layer=list_layers,model=model,output_format='numpy',permissive=True)
    
    idx = list_layers.index('atom_to_aa_indices')
    list_atom_to_residue_indices = [(list_features[k][idx] - list_features[k][idx][0])[:,0] for k in range(len(list_inputs))]
    
    idx = list_layers.index('aa_to_atom_indices')
    list_residues_to_atom_indices = [ np.maximum(list_features[k][idx] - list_features[k][idx][0,0],-1) for k in range(len(list_inputs))]
    
    idx = list_layers.index('nearest_neighbor_search_atom')
    list_nearest_neighbor_atoms = [(list_features[k][idx] - list_features[k][idx].min()) for k in range(len(list_inputs))]
    
    # Small modification here. In case we used the "protein serialization trick" described in the paper, there is an extra offset to be removed.
    idx = list_layers.index('frames_atom') 
    big_distance = 3000        
    list_atom_frames = []
    for k in range(len(list_inputs)):
        tmp = list_features[k][idx]
        offset = np.round( tmp[:,0,:].mean() / big_distance ) * big_distance
        tmp[:,0,:] -= offset
        # Small modification here. In case we used the "protein serialization trick" described in the paper, there is an extra offset to be removed.
        list_atom_frames.append(tmp)
                
    list_atomic_plus_residue_embeddings = []
    for k in range( len(list_names)):
        atom_to_residue_index = list_atom_to_residue_indices[k]
        atomic_embeddings = list_features[k][list_layers.index('SCAN_filter_activity_atom_1_normalization')]
        residue_embeddings = np.concatenate( (list_features[k][list_layers.index('SCAN_filter_activity_aa_2_normalization')],
                                              list_features[k][list_layers.index('classifier_output')] ),axis=-1)
        residue_embeddings_up_pooled = residue_embeddings[atom_to_residue_index]
        atomic_plus_residue_embedding = np.concatenate( (atomic_embeddings, residue_embeddings_up_pooled),axis=-1)
        print(k,atomic_plus_residue_embedding.shape)
        list_atomic_plus_residue_embeddings.append(atomic_plus_residue_embedding)
        
