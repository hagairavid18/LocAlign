import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv



class CorrespondenceDenoisingModule(nn.Module):

    def __init__(self, k: int, n_gnn_layers: int = 3, n_rbf_functions: int = 16):
        super(CorrespondenceDenoisingModule, self).__init__()
        self.k = k  # Number of top correspondences to keep
        self._k_training = k
        self._k_inference = 2000
        self.n_gnn_layers = n_gnn_layers  # Number of GNN layers
        self.gnn_layers = nn.ModuleList([GraphConv(1, 1, aggr='sum') for _ in range(n_gnn_layers)])
        self.n_rbf_functions = n_rbf_functions  # Number of RBF functions
            
        self.edge_learner = EdgeWeightLearner(input_dim=2 * n_rbf_functions, hidden_dim=64)  # Input: 2 * 16 (dist_A, dist_B)
        self.rbf_encoder = LearnableRBFEncoding(num_basis=n_rbf_functions, rbf_range=(0.0, 50.0), learn_gamma=True)  # RBF encoding for distances
        

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
        
    def _force_consistency(self, soft_correspondences: torch.Tensor, src_coords: torch.Tensor, tgt_coords: torch.Tensor) -> torch.Tensor:
        self.k = self._k_training if self.training else self._k_inference
        B, N, _ = soft_correspondences.shape  # B: batch size, N: number of points

        top_k_values, top_k_indices = self.extract_top_k_correspondences(soft_correspondences)
        graph_data, edge_consistency = self.build_correspondence_graph(top_k_values, top_k_indices, src_coords, tgt_coords)

        for i in range(self.n_gnn_layers):
            graph_data.x = (graph_data.x.reshape(B,self.k) / graph_data.x.reshape(B,self.k).sum(1, keepdim=True)).reshape(B *self.k,1) # THESE TWO LINES CAN BE COMMENTED OUT
            graph_data.x = self.gnn_layers[i](graph_data.x, graph_data.edge_index, graph_data.edge_attr) # THESE TWO LINES CAN BE COMMENTED OUT

        graph_data.x = graph_data.x.relu()  # Apply ReLU activation to the node features
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
        top_k_indices_flat = top_k_plus_one_indices_flat[:,:self.k] 
        

        # Convert the flattened indices back to (i, j) pairs in the original 2D matrix
        top_k_indices = torch.stack(
            (top_k_indices_flat // N, top_k_indices_flat % N), dim=-1
        )  # Convert to (i, j) index pairs

        # Reshape the top_k_values to the correct shape
        top_k_values = top_k_values.view(B, self.k)
        top_k_indices = top_k_indices.view(B, self.k, 2)  # Shape: (B, K, 2) -> (source, target) pairs

        top_k_values = top_k_values.float()
        
        return top_k_values, top_k_indices

    def build_correspondence_graph(self, top_k_values: torch.Tensor, top_k_indices: torch.Tensor, src_coords: torch.Tensor, tgt_coords: torch.Tensor):
        """
        Build a batch-aware graph using richer edge features including angles.
        """
        B, K, _ = top_k_indices.shape  # Batch size, Number of top correspondences

        batch_idx = torch.arange(B, device=top_k_indices.device).repeat_interleave(K)

        # Gather selected points
        tgt_selected = tgt_coords.gather(1, top_k_indices[..., 0].unsqueeze(-1).expand(-1, -1, tgt_coords.size(-1)))
        src_selected = src_coords.gather(1, top_k_indices[..., 1].unsqueeze(-1).expand(-1, -1, src_coords.size(-1)))

        '''
        tgt_selected_frame = ... Same gather formula with extra expand
        src_selection_frame = ....  Same gather formula with extra expand
        
        tgt_diff = tgt_selected_frame[:,:,0,:].unsqueeze(2) - tgt_selected_frame[:,:,0,:].unsqueeze(1) # (B,K,K,3)
        src_diff -> same
        
        tgt_diff_invariant = einsum( 'bklm,bknm->bkln' tgt_diff,  tgt_selected_frame[:,:,1:,:]) # (B,K,K,3)
        
        tgt_diff_r_theta_phi = euclidean_to_spherical(tgt_diff_invariant)
        
        --> RBF embeddings for r, cosine/sine for theta/phi
                
        def euclidian_to_spherical(x,return_r=True,cut='2pi',eps=1e-8):
            r = ops.sqrt( ops.sum(x**2,axis=-1) )
            theta = ops.arccos(x[...,-1]/(r+eps) )
            phi = ops.atan2( x[...,1],x[...,0]+eps)
            if cut == '2pi':
                phi = phi + ops.cast(ops.greater(0.,phi), 'float32') * (2 * np.pi)
            if return_r:
                return ops.stack([r,theta,phi],axis=-1)
            else:
                return ops.stack([theta, phi], axis=-1)
        '''
        # Compute pairwise differences for source and target
        src_diff = src_selected.unsqueeze(2) - src_selected.unsqueeze(1)  # (B, K, K, 3)
        tgt_diff = tgt_selected.unsqueeze(2) - tgt_selected.unsqueeze(1)  # (B, K, K, 3)

        # Compute distances for source and target
        dist_A = torch.norm(src_diff, dim=-1)  # (B, K, K)
        dist_B = torch.norm(tgt_diff, dim=-1)  # (B, K, K)

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
        edge_features = torch.stack([dist_A[:, i_idx, j_idx], dist_B[:, i_idx, j_idx]], dim=-1)
        edge_consistency = torch.abs(edge_features[..., 0] - edge_features[..., 1])
        
        edge_features = self.rbf_encoder(edge_features).view(B, edge_features.shape[1], -1)
        # Pass through the edge learner
        edge_weight = self.edge_learner(edge_features)
        edge_weight = edge_weight.view(-1, 1)
        edge_consistency = edge_consistency.view(-1, 1)

        # Create PyG Data object
        data = Data(
            x=top_k_values.reshape(-1, 1),  # Node features (correspondence scores)
            edge_index=edge_index,  # Edge connectivity
            edge_attr=edge_weight,  # Enhanced edge attributes
            batch=batch_idx  # Batch assignment for each node
        )
        
        return data, edge_consistency

        
class EdgeWeightLearner(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim=16):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Input: edge_attr
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
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



if __name__ == "__main__":
    # Example Usage:
    # GNN model parameters
    input_dim = 1  # Example: Correspondence has 1 feature (e.g., distance)
    hidden_dim = 32
    output_dim = 1  # Output is the refined correspondence score or feature

    # Create the correspondence denoising module
    denoising_module = CorrespondenceDenoisingModule(k=10, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    # Example data (soft_correspondences, mask, src_coords, tgt_coords)
    soft_correspondences = torch.rand(8, 1200, 1200)  # Example correspondences
    mask = torch.ones(8, 1200, 1200)  # 2D Mask (B, N, N) for both source and target
    src_coords = torch.rand(8, 1200, 3)  # Example source coordinates (B, N, 3)
    tgt_coords = torch.rand(8, 1200, 3)  # Example target coordinates (B, N, 3)

    # Apply the correspondence denoising module
    updated_correspondences = denoising_module._force_consistency(soft_correspondences, mask, src_coords, tgt_coords)
    print(updated_correspondences)
