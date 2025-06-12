import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv


import numpy as np

def euclidean_to_spherical(x, cut='2pi', eps=1e-8):
    r = torch.linalg.norm(x, dim=-1)
    theta = torch.acos(x[..., 2] / (r + eps))
    phi = torch.atan2(x[..., 1], x[..., 0] + eps)
    
    if cut == '2pi':
        phi = phi + (phi < 0).float() * (2 * np.pi)
    
    return torch.stack([r, theta, phi], dim=-1)

class CorrespondenceDenoisingModule(nn.Module):

    def __init__(self, k: int, n_gnn_layers: int = 3, n_rbf_functions: int = 16, with_angles: bool = True):
        super(CorrespondenceDenoisingModule, self).__init__()
        self.k = k  # Number of top correspondences to keep
        self.n_gnn_layers = n_gnn_layers  # Number of GNN layers
        self.gnn_layers = GraphConv(1, 1, aggr='sum')
        self.n_rbf_functions = n_rbf_functions  # Number of RBF functions
        self.with_angles = with_angles  # Whether to include angle features
        
        pre_input_dim = 2 * n_rbf_functions + 8 if with_angles else 2 * n_rbf_functions
        self.edge_learner = EdgeWeightLearner(input_dim=pre_input_dim, hidden_dim=64)  # Input: 2 * 16 (dist_A, dist_B)
        self.rbf_encoder = LearnableRBFEncoding(num_basis=n_rbf_functions, rbf_range=(0.0, 100.0), learn_gamma=True)
        

        self.apply(self.init_weights)
       
        for gnn_layer in self.gnn_layers:
            with torch.no_grad():
                gnn_layer.lin_rel.weight.fill_(0.05)
                gnn_layer.lin_rel.bias.fill_(1.0)
                gnn_layer.lin_root.weight.fill_(0.0)
            gnn_layer.lin_root.weight.requires_grad = False
            gnn_layer.lin_rel.bias.requires_grad = False
         
    @staticmethod
    def init_weights(m):
        """Custom weight initialization for stability"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
    def forward(self, soft_correspondences: torch.Tensor, src_frames: torch.Tensor, tgt_frames: torch.Tensor) -> torch.Tensor:
        B, N, _ = soft_correspondences.shape  # B: batch size, N: number of points

        top_k_values, top_k_indices = self.extract_top_k_correspondences(soft_correspondences)
        graph_data = self.build_correspondence_graph(top_k_values, top_k_indices, src_frames, tgt_frames)

        for i in range(self.n_gnn_layers):
            graph_data.x = (graph_data.x.reshape(B,self.k) / graph_data.x.reshape(B,self.k).sum(1, keepdim=True)).reshape(B *self.k,1)
            graph_data.x = self.gnn_layers(graph_data.x, graph_data.edge_index, graph_data.edge_attr) 

        graph_data.x = graph_data.x.relu()
        # Step 4: Update soft correspondences
        updated_correspondences = graph_data.x.squeeze(-1).to(soft_correspondences).view(B, -1) # Shape: [B, K]

        return updated_correspondences, top_k_indices

    def extract_top_k_correspondences(self, soft_correspondences: torch.Tensor) -> torch.Tensor:
        """
        Extract the top K correspondences from the entire matrix by flattening it.
        """
        B, N, _ = soft_correspondences.shape
        
        # Flatten the distance matrix to a 1D vector
        flat_correspondences = soft_correspondences.reshape(B, -1)  # Flatten each batch
        
        # Find the top K values and their indices across the entire matrix
        top_k_plus_one_values, top_k_plus_one_indices_flat = torch.topk(flat_correspondences, self.k + 1, dim=-1, largest=True)
        top_k_values = top_k_plus_one_values[:, :self.k] - top_k_plus_one_values[:, self.k:]
        top_k_indices_flat = top_k_plus_one_indices_flat[:, :self.k] 
        
        # Convert the flattened indices back to (i, j) pairs in the original 2D matrix
        top_k_indices = torch.stack((top_k_indices_flat // N, top_k_indices_flat % N), dim=-1)  # Convert to (i, j) index pairs

        return top_k_values, top_k_indices

    def _encode_angles(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Encode angles using sine and cosine transformations.
        """
        theta_sin = torch.sin(angles[..., 0])
        theta_cos = torch.cos(angles[..., 0])
        phi_sin = torch.sin(angles[..., 1])
        phi_cos = torch.cos(angles[..., 1])
        return torch.stack([theta_sin, theta_cos, phi_sin, phi_cos], dim=-1)
    
    def get_node_diffs(self, frames, indices):
        """
        Get the differences between the selected frames based on the indices.
        """
        selected_frames = frames.gather(1, indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, frames.size(2), frames.size(3)))
        diff = selected_frames[:, :, 0, :].unsqueeze(2) - selected_frames[:, :, 0, :].unsqueeze(1)  # (B, K, K, 3)
        # diff_invariant = torch.einsum('bklm,bknm->bkln', diff, selected_frames[:, :, 1:, :].transpose(-1, -2)) # TODO: test this line
        diff_invariant = torch.einsum('bklm,bknm->bkln', diff, selected_frames[:, :, 1:, :]) # TODO: test this line
        diff_r_theta_phi = euclidean_to_spherical(diff_invariant)
        return diff_r_theta_phi
    
    def build_correspondence_graph(self, top_k_values: torch.Tensor, top_k_indices: torch.Tensor, src_frames: torch.Tensor, tgt_frames: torch.Tensor):
        """
        Build a batch-aware graph using richer edge features including angles.
        """
        B, K, _ = top_k_indices.shape  # Batch size, Number of top correspondences

        batch_idx = torch.arange(B, device=top_k_indices.device).repeat_interleave(K)

        tgt_diff_r_theta_phi = self.get_node_diffs(tgt_frames, top_k_indices[..., 0])
        src_diff_r_theta_phi = self.get_node_diffs(src_frames, top_k_indices[..., 1])
        
        # Create edge indices
        i_idx_upper, j_idx_upper = torch.triu_indices(K, K, offset=1, device=top_k_indices.device)  # Upper triangle indices
        i_idx_lower, j_idx_lower = torch.tril_indices(K, K, offset=-1, device=top_k_indices.device)  # Lower triangle indices
        i_idx = torch.cat([i_idx_upper, i_idx_lower], dim=0)
        j_idx = torch.cat([j_idx_upper, j_idx_lower], dim=0)

        # Fix batching for edge indices
        edge_index_per_batch = torch.stack([i_idx, j_idx], dim=0)
        batch_offset = torch.arange(B, device=top_k_values.device).view(B, 1, 1) * K
        edge_index = edge_index_per_batch.unsqueeze(0).expand(B, -1, -1) + batch_offset
        edge_index = edge_index.permute(0, 2, 1).reshape(-1, 2).T  # (2, total_edges)

        # Stack all edge features together
        distance_features = torch.stack([src_diff_r_theta_phi[:, i_idx, j_idx, 0], tgt_diff_r_theta_phi[:, i_idx, j_idx, 0]], dim=-1)
        
        edge_features = self.rbf_encoder(distance_features).view(B, distance_features.shape[1], -1)
        if self.with_angles:
            angle_features = torch.cat([self._encode_angles(tgt_diff_r_theta_phi[:, i_idx, j_idx, 1:]),
                                        self._encode_angles(src_diff_r_theta_phi[:, i_idx, j_idx, 1:]),], dim=-1)
                
            edge_features = torch.cat([edge_features, angle_features], dim=-1)  # Concatenate RBF features
        # Pass through the edge learner
        edge_weight = self.edge_learner(edge_features).view(-1, 1)

        # Create PyG Data object
        data = Data(
            x=top_k_values.reshape(-1, 1),  # Node features (correspondence scores)
            edge_index=edge_index,  # Edge connectivity
            edge_attr=edge_weight,  # Enhanced edge attributes
            batch=batch_idx  # Batch assignment for each node
        )
        
        return data

        
class EdgeWeightLearner(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim=16):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Input: edge_attr
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, edge_attr):
        """
        Args:
            edge_attr: Tensor of shape (B * num_edges, input_dim)

        Returns:
            edge_weight: Tensor of shape (B * num_edges,)
        """
        if len(edge_attr.shape) == 3:
            # (B, num_edges, input_dim)
            edge_attr = self.norm(edge_attr)
            edge_attr = edge_attr.view(-1, edge_attr.size(-1))  # (B * num_edges, input_dim)
        elif len(edge_attr.shape) == 2:
            raise ValueError("Edge attribute is 2D, expected 3D for per-batch normalization.")
        else:
            raise ValueError(f"Unexpected edge_attr shape: {edge_attr.shape}")

        return self.mlp(edge_attr).view(-1)
    

class LearnableRBFEncoding(nn.Module):
    def __init__(self, num_basis=16, rbf_range=(0.0, 20.0), learn_gamma=True):
        super().__init__()
        centers = torch.linspace(rbf_range[0], rbf_range[1], num_basis)
        self.centers = nn.Parameter(centers)  # learnable centers

        if learn_gamma:
            self.gamma = nn.Parameter(torch.full((num_basis,), 1.0))  # learnable per-basis gamma
        else:
            delta = (rbf_range[1] - rbf_range[0]) / num_basis
            gamma = 1.0 / (delta ** 2)
            self.register_buffer('gamma', torch.full((num_basis,), gamma))  # fixed gamma

    def forward(self, distances):
        # distances: (...,) shape
        diff = distances.unsqueeze(-1) - self.centers  # (..., num_basis)
        return torch.exp(-self.gamma * diff ** 2)      # (..., num_basis)

