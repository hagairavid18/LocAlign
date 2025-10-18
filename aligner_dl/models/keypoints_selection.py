from models.utils.math import euclidean_to_spherical
from models.correspondences_denoiser import LearnableRBFEncoding,EdgeWeightLearner
from layers.blocks import EmbeddingBlock
import torch
import torch.nn as nn

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
        self._embedding_block = EmbeddingBlock(input_dim=embedding_size,output_dim=3,n_blocks=3,dropout=0.0,bias=False)
        self._rbf_encoder = LearnableRBFEncoding(num_basis=n_rbf_functions, rbf_range=(0.0, 6.0), learn_gamma=True)
        self._edge_learner = EdgeWeightLearner(input_dim=pre_input_dim, hidden_dim=16)
        self.apply(self.init_weights)
                       
         
    @staticmethod
    def init_weights(m):
        """Custom weight initialization for stability"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
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
        assert embedding_size == self._embedding_size
        
        local_coordinates = self._get_local_coordinates(frames,neighbors) # B X N X K X 3 [3,theta,phi]
        local_coordinates = local_coordinates.view(B,N*K,3) # Reshape before passing to edge learner.
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
        local_value = value.gather(1, neighbors.view(B,N*K) ).view(B,N,K)
        local_query = query.gather(1, neighbors.view(B,N*K) ).view(B,N,K)
        local_attention = torch.softmax( key.unsqueeze(-1) * local_query + local_scalar_edges,axis=-1) # Scalar attention over neighbors.
        output_score = torch.sum(local_value * local_attention,axis=-1)
        output_score = output_score.masked_fill(~mask, -float('inf'))  # Make sure that masked positions have no output_score.
        top_k_indices, top_k_score = self._get_rectified_top_k(output_score, mask)
        return top_k_indices, top_k_score
    
    def _get_local_coordinates(
        self,
        frames: torch.Tensor, 
        neighbors: torch.Tensor) -> torch.Tensor:
        B,N,K = neighbors.shape
        
        neighbor_coordinates = frames[:,:,0,:].gather(1, neighbors.view(B,N*K,-1) ).view(B,N,K,3)
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