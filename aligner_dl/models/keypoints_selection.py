import torch
import torch.nn as nn

from models.utils.math import euclidean_to_spherical
from models.correspondences_denoiser import LearnableRBFEncoding, EdgeWeightLearner
from models.layers.blocks import EmbeddingBlock


class KeypointsSelection(nn.Module):
    def __init__(
        self,
        embedding_size: int = 256,
        n_rbf_functions: int = 16,
        top_k: int = 1000,
        with_angles: bool = True
        ):
        super(KeypointsSelection, self).__init__()
        
        self._embedding_size = embedding_size
        self._top_k = top_k                                
        self._add_angle_features = with_angles  # Whether to include angle features                            
        pre_input_dim = n_rbf_functions + 4 if with_angles else n_rbf_functions
        self._embedding_block = EmbeddingBlock(input_dim=embedding_size, output_dim=3, n_blocks=2, dropout=0.0, bias=True)
        self._rbf_encoder = LearnableRBFEncoding(num_basis=n_rbf_functions, rbf_range=(0.0, 6.0), learn_gamma=True)
        self._edge_learner = EdgeWeightLearner(input_dim=pre_input_dim, hidden_dim=16)
        self._root_term = torch.nn.Parameter(torch.tensor(1.0))
        self.apply(self.init_weights)
                       
    @staticmethod
    def init_weights(m):
        """Custom weight initialization for stability"""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight,nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    import torch

    def safe_softmax(self, x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        """
        Softmax that safely handles zero denominators.

        Args:
            x: input tensor
            dim: dimension to apply softmax over
            eps: small constant to avoid division by zero
        """
        exps = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
        denom = exps.sum(dim=dim, keepdim=True)
        denom = torch.clamp(denom, min=eps)  # prevent division by zero
        return exps / denom

        
    def forward(
        self,
        embeddings: torch.Tensor,
        frames: torch.Tensor,
        neighbors: torch.Tensor,
        mask: torch.Tensor,
        previous_importance: torch.Tensor | None  = None,
        ) -> tuple[torch.Tensor,torch.Tensor]:
        
        B, N, embedding_size = embeddings.shape # Batch size, number of atoms, embedding size.
        K = neighbors.shape[-1] # Number of neigbhors
        # assert embedding_size == self._embedding_size
        neighbors = torch.clip(neighbors.type(torch.int64), 0, N-1) # Make sure that neighbors are not outside of max length.
        local_coordinates = self._get_local_coordinates(frames, neighbors) # B X N X K X 3 [3,theta,phi]
        local_coordinates = local_coordinates.view(B, N*K, 3) # Reshape before passing to edge learner.
        local_distances = local_coordinates[:,:,0]
        local_edges = self._rbf_encoder(local_distances) # Calculate scalar edges; same code as in correspondence solver module.
        if self._add_angle_features:
            local_angles = local_coordinates[:,:,1:]
            local_edges = torch.cat([local_edges, self._encode_angles(local_angles)], dim=-1)
        local_scalar_edges = self._edge_learner(local_edges).view(B,N,K)
                
        value_key_query = self._embedding_block(embeddings,mask) # value,key,queries for attention. Here, the value, query and key are scalars.
        value = value_key_query[:,:,0]
        if previous_importance is not None: # Previous importance, (either recycled from previous iteration or user-provided if using input motif)
            value += previous_importance
        key = value_key_query[:,:,1]
        query = value_key_query[:,:,2]
        query = query.masked_fill(~mask, -float('inf')) # Make sure that masked positions have no role in attention.
        local_value = value.gather(1, neighbors.view(B,N*K) ).view(B, N, K)
        local_query = query.gather(1, neighbors.view(B,N*K) ).view(B, N, K)
        local_attention = self.safe_softmax(key.unsqueeze(-1) * local_query + local_scalar_edges) # Scalar attention over neighbors.
        output_score = torch.sum(local_value * local_attention,axis=-1)  + value * self._root_term
        output_score = output_score.masked_fill(~mask, -float('inf'))  # Make sure that masked positions have no output_score.
        top_k_indices, top_k_score = self._get_rectified_top_k(output_score, mask)
        return top_k_indices, top_k_score
    
    def _get_local_coordinates(
        self,
        frames: torch.Tensor, 
        neighbors: torch.Tensor) -> torch.Tensor:
        B,N,K = neighbors.shape
        
        neighbor_coordinates = frames[:,:,0,:].gather(1, neighbors.view(B,N*K,-1).expand(-1,-1,3) ).view(B,N,K,3)        
        difference_vector = neighbor_coordinates - frames[:,:,0,:].unsqueeze(2)
        local_neighbor_coordinates = torch.einsum('bnkm, bnlm->bnkl', difference_vector,  frames[:,:,1:,:])
        return euclidean_to_spherical(local_neighbor_coordinates)

    def _encode_angles(
        self, 
        angles: torch.Tensor
        ) -> torch.Tensor:
        """
        Encode angles using sine and cosine transformations.
        """
        theta_sin = torch.sin(angles[..., 0])
        theta_cos = torch.cos(angles[..., 0])
        phi_sin = torch.sin(angles[..., 1])
        phi_cos = torch.cos(angles[..., 1])
        return torch.stack([theta_sin, theta_cos, phi_sin, phi_cos], dim=-1)
 

    def _get_rectified_top_k(
        self, 
        scalar_values: torch.Tensor, 
        mask: torch.Tensor
        ) -> tuple[torch.Tensor,torch.Tensor]:
        """
        Computes the top k scalar values and their indices, while ensuring that the mask is applied correctly.
        Args:
            scalar_values (torch.Tensor): Values to compute the top k from.
            mask (torch.Tensor): Mask to apply on the scalar values.

        Returns:
            tuple[torch.Tensor,torch.Tensor]: Top k indices and their corresponding scalar values.
        """        
        
        scalar_values = scalar_values.masked_fill(~mask, -float('inf'))
                
        top_k_plus_one_scalar, top_k_plus_one_indices = torch.topk(scalar_values, self._top_k + 1, dim=-1, largest=True) # I removed the train/eval condition for now...
        
        top_k_plus_one_mask = mask.gather(1, top_k_plus_one_indices) #
        top_k_plus_one_scalar.masked_fill_(~top_k_plus_one_mask, 0) # These two lines are to deal with the edge case where num_real_atoms < _top_k. In this case and without the fix, top_k_plus_one_scalar[:,:,_top_k] = -infty.
        
        top_k_scalar = top_k_plus_one_scalar[:, :self._top_k] - top_k_plus_one_scalar[:, self._top_k:]
        top_k_indices = top_k_plus_one_indices[:, :self._top_k]        
        return top_k_indices, top_k_scalar
    
    
if __name__ == '__main__':
    import sys,os
    import numpy as np
    sys.path.append('/Users/jerometubiana/GitHub/ScanNet_Ub')
    os.environ['KERAS_BACKEND'] = 'torch'
    from predict_features import predict_features
    
    def extract_features(list_inputs,model='ScanNet_ubiquitin_autoregressive_config9_noMSA_30_08_1'):
        list_layers = [
            'atom_to_aa_indices',
            'frames_atom',
            'aa_to_atom_indices',
            'nearest_neighbor_search_atom',
            'SCAN_filter_activity_atom_1_normalization',
            'SCAN_filter_activity_aa_2_normalization',
            'classifier_output'
        ]        
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
            list_atomic_plus_residue_embeddings.append(atomic_plus_residue_embedding)
                        
        dict_outputs = {
            'atom_embeddings': list_atomic_plus_residue_embeddings,
            'atom_to_residue_indices': list_atom_to_residue_indices,
            'residue_to_atom_indices': list_residues_to_atom_indices,
            'atom_nearest_neighbors': list_nearest_neighbor_atoms,
            'atom_frames': list_atom_frames
        }
        return dict_outputs
    
    
    
    def padd_array(array, N,value ):
        shape = list(array.shape)
        padding = max(N - shape[0],0)
        return torch.tensor(np.concatenate( ( array[:N], value* np.ones( [padding] + shape[1:] ,dtype=array.dtype) ),axis=0))
    
    src_id = '1qzz_A'
    tgt_id = '4mwz_A'
    
    dict_outputs = extract_features([src_id,tgt_id])
        
    embedding_size = dict_outputs['atom_embeddings'][0].shape[-1]
    N = max([atom_embeddings.shape[0] for atom_embeddings in dict_outputs['atom_embeddings']])
    
    embeddings = torch.stack([ padd_array(dict_outputs['atom_embeddings'][k], N,0.) for k in range(2)],axis=0)
    frames= torch.stack([ padd_array(dict_outputs['atom_frames'][k], N,0.) for k in range(2)],axis=0)
    nearest_neighbors= torch.stack([ padd_array(dict_outputs['atom_nearest_neighbors'][k], N,0) for k in range(2)],axis=0)    
    masks = torch.stack([  padd_array( np.ones( len(embeddings),dtype='bool'), N,False) for embeddings in dict_outputs['atom_embeddings']], axis=0)
        
    keypoints_selection = KeypointsSelection(embedding_size=embedding_size)
    previous_importance = torch.stack([  padd_array( np.concatenate( (np.zeros(2234,dtype=float), 100.*np.ones( 1,dtype=float)),axis=0), N,False) for _ in range(2)], axis=0)
    
    
    # previous_importance = None
        
    top_k_indices, top_k_scalar = keypoints_selection(embeddings,frames,nearest_neighbors,masks,previous_importance=previous_importance)