import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv


class CorrespondenceDenoisingModule(nn.Module):

    def __init__(self, k: int):
        super(CorrespondenceDenoisingModule, self).__init__()
        self.k = k  # Number of top correspondences to keep

        self.gnn_layer = GraphConv(1, 1, aggr='sum')
        self.edge_learner = EdgeWeightLearner(input_dim=3)
        

        self.apply(self.init_weights)
       
        with torch.no_grad():
            self.gnn_layer.lin_rel.weight.fill_(0.5)
            self.gnn_layer.lin_rel.bias.fill_(0.0)
            self.gnn_layer.lin_root.weight.fill_(0.5)

    @staticmethod
    def init_weights(m):
        """Custom weight initialization for stability"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
    def _force_consistency(self, soft_correspondences: torch.Tensor, src_coords: torch.Tensor, tgt_coords: torch.Tensor) -> torch.Tensor:
        B, N, _ = soft_correspondences.shape  # B: batch size, N: number of points

        top_k_values, top_k_indices = self.extract_top_k_correspondences(soft_correspondences)
        graph_data = self.build_correspondence_graph(top_k_values, top_k_indices, src_coords, tgt_coords)
        
        for i in range(1):
            graph_data.x = self.gnn_layer(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        

        print(f"lin_rel: {self.gnn_layer.lin_rel.weight} . lin_ root: {self.gnn_layer.lin_root.weight}")

        # Step 4: Update soft correspondences
        updated_correspondences = graph_data.x.squeeze(-1).to(soft_correspondences) # Shape: [B, K]
        updated_correspondences_flat = updated_correspondences.view(B, -1)  # Shape: [B, K]

        soft_correspondences_flat = soft_correspondences.clone().view(B, -1)  # Shape: [B, N * N]
        flat_indices = top_k_indices[..., 0] * N + top_k_indices[..., 1]  # Flattened indices in B X K

        soft_correspondences_flat.scatter_add_(1, flat_indices, updated_correspondences_flat)

        updated_soft_correspondences2 = soft_correspondences_flat.view(B, N, N)
        return updated_soft_correspondences2

    def extract_top_k_correspondences(self, soft_correspondences: torch.Tensor) -> torch.Tensor:
        """
        Extract the top K correspondences from the entire matrix by flattening it.
        """
        B, N, _ = soft_correspondences.shape
        
        # Flatten the distance matrix to a 1D vector
        flat_correspondences = soft_correspondences.view(B, -1)  # Flatten each batch
        
        # Find the top K values and their indices across the entire matrix
        top_k_values, top_k_indices_flat = torch.topk(flat_correspondences, self.k, dim=-1, largest=True)
        
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
        Build a batch-aware graph using richer edge features.
        """
        B, K, _ = top_k_indices.shape  # Batch size, Number of top correspondences

        batch_idx = torch.arange(B, device=top_k_indices.device).repeat_interleave(K)

        # Gather selected points
        tgt_selected = tgt_coords.gather(1, top_k_indices[..., 0].unsqueeze(-1).expand(-1, -1, tgt_coords.size(-1)))
        src_selected = src_coords.gather(1, top_k_indices[..., 1].unsqueeze(-1).expand(-1, -1, src_coords.size(-1)))

        # Compute pairwise differences
        src_diff = src_selected.unsqueeze(2) - src_selected.unsqueeze(1)  # (B, K, K, 3)
        tgt_diff = tgt_selected.unsqueeze(2) - tgt_selected.unsqueeze(1)  # (B, K, K, 3)

        # Compute distances
        dist_A = torch.norm(src_diff, dim=-1)  # (B, K, K)
        dist_B = torch.norm(tgt_diff, dim=-1)  # (B, K, K)

        # Compute angles via cosine similarity
        src_norm = torch.nn.functional.normalize(src_diff, dim=-1, eps=1e-6)
        tgt_norm = torch.nn.functional.normalize(tgt_diff, dim=-1, eps=1e-6)
        cosine_similarity = (src_norm * tgt_norm).sum(dim=-1)  # (B, K, K)

        # create edge indices
        i_idx_upper, j_idx_upper = torch.triu_indices(K, K, offset=1, device=top_k_indices.device)  # Upper triangle indices
        i_idx_lower, j_idx_lower = torch.tril_indices(K, K, offset=-1, device=top_k_indices.device)  # Lower triangle indices
        i_idx = torch.cat([i_idx_upper, i_idx_lower], dim=0)
        j_idx = torch.cat([j_idx_upper, j_idx_lower], dim=0)

        # Fix batching
        edge_index_per_batch = torch.stack([i_idx, j_idx], dim=0)
        batch_offset = torch.arange(B, device=top_k_values.device).view(B, 1, 1) * K
        edge_index = edge_index_per_batch.unsqueeze(0).expand(B, -1, -1) + batch_offset
        edge_index = edge_index.permute(0, 2, 1).reshape(-1, 2).T  # (2, total_edges)

        # Compute enhanced edge features
        angle_feature = cosine_similarity[:, i_idx, j_idx]

        # Stack features together
        edge_features = torch.stack([dist_A[:, i_idx, j_idx], dist_B[:, i_idx, j_idx], angle_feature], dim=-1)

        # Pass through the edge learner
        edge_weight = self.edge_learner(edge_features).view(-1, 1)
        print(f"mean edge weight {edge_weight.mean()} max {edge_weight.max()} min: {edge_weight.min()}")

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
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Input: edge_attr
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1), # Output: edge_weight
            nn.Sigmoid()  # Ensures output is between 0 and 1
        )

    def forward(self, edge_attr):
        """
        Args:
            edge_attr: Tensor of shape (B, num_edges_per_batch)

        Returns:
            edge_weight: Tensor of shape (B, num_edges_per_batch)
        """
        if len(edge_attr.shape) == 2:
            edge_attr = edge_attr.unsqueeze(-1)  # (B, num_edges, 1) for MLP
        edge_weight = self.mlp(edge_attr)  # Pass through MLP
        return edge_weight.squeeze(-1)  # Return shape (B, num_edges)
    
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
