import torch
import torch.nn as nn


class RecyclingModule(nn.Module):
    
    def __init__(self,
                 fourier_grid_size: int = 8,
                 fourier_min_length: float = 5.0,
                 fourier_n_rbf_functions: int = 8,
                 recycle_scalar: bool = False,
                 recycle_graph: bool = False,
                 recycle_coords: bool = True):

        super(RecyclingModule, self).__init__()

        # Create Fourier vectors
        fourier_grid_1d = torch.arange(fourier_grid_size) / (fourier_grid_size - 1) * 2 * torch.pi / fourier_min_length
        fourier_vectors = torch.stack(
            torch.meshgrid(fourier_grid_1d, fourier_grid_1d, fourier_grid_1d, indexing='ij'),
            dim=-1
        ).reshape(-1, 3).T[:, 1:]  # Transpose then drop first column

        fourier_norms = torch.sqrt(torch.sum(fourier_vectors ** 2, dim=0))
        fourier_vector_norms = torch.cat((fourier_norms, fourier_norms))

        # Register buffers (non-trainable, but device-aware)
        self.register_buffer("fourier_vectors", fourier_vectors)
        self.register_buffer("fourier_vector_norms", fourier_vector_norms)

        # Trainable RBF parameters
        self.rbf_scales = nn.Parameter(
            (fourier_min_length / (2 * torch.pi) * torch.linspace(0, 4, fourier_n_rbf_functions)) ** 2
        )
        self.rbf_weights = nn.Parameter(torch.full((fourier_n_rbf_functions,), 1.))

        # Flags
        self.recycle_scalar = recycle_scalar
        self.recycle_graph = recycle_graph
        self.recycle_coords = recycle_coords
        if self.recycle_scalar:
            self.scalar_scale = nn.Parameter(torch.full((1,), 1.))

        
        
    def _build_reference_frame(self, coordinates):
        mean = torch.mean(coordinates, axis=-2)
        U, eigenvalues, Vh = torch.linalg.svd(coordinates - mean.unsqueeze(-2))
        V = torch.swapaxes(Vh,-2,-1)
        return torch.cat([mean.unsqueeze(-2),V],axis=-2)
    
    def _global_to_local(self,coordinates, reference_frame):
        return  torch.einsum('ijk,ikl->ijl', (coordinates - reference_frame[:,0,:].unsqueeze(-2)), reference_frame[:,1:,:])    
                
    def _fourier_encode(self, coordinates):
        fourier_vectors_dot_coords = torch.matmul(coordinates,self.fourier_vectors)
        return torch.cat( (torch.cos(fourier_vectors_dot_coords),torch.sin(fourier_vectors_dot_coords) ),axis=-1)
    
    def _get_decay_function_in_fourier_space(self):
        '''
        Here, we are calculating the fourier transform of the distance decay function. We are explicitly parameterizing it as a sum of weighted gaussians.
        This way, the original decay function is also a weighted sum of gaussians. The non-negative coefficients ensure that the function goes to zero.
        
        '''
        return (torch.exp(-0.5*self.fourier_vector_norms.unsqueeze(1)**2 * self.rbf_scales.relu().unsqueeze(0) ) * self.rbf_weights.relu().unsqueeze(0) ).mean(1)
    
    
    def forward(self, src_coords: torch.Tensor, tgt_coords: torch.Tensor,
                top_corr_values: torch.Tensor | None = None, top_corr_indices: torch.Tensor | None=None):
                
        tgt_frame = self._build_reference_frame(tgt_coords)        
        tgt_coords_local = self._global_to_local(tgt_coords, tgt_frame)
        src_coords_local = self._global_to_local(src_coords, tgt_frame)
        
        tgt_fourier_embeddings = self._fourier_encode(tgt_coords_local).detach() # Detach to stop backpropagation here.
        src_fourier_embeddings = self._fourier_encode(src_coords_local).detach() # Detach to stop backpropagation here.                        
        fourier_scalings = self._get_decay_function_in_fourier_space()
        tgt_fourier_embeddings *= fourier_scalings.unsqueeze(0).unsqueeze(0)
        src_fourier_embeddings *= fourier_scalings.unsqueeze(0).unsqueeze(0)
        
        if self.recycle_scalar and (top_corr_values is not None) and (top_corr_indices is not None):
           
            B, N = src_coords.shape[:2]
            device = top_corr_indices.device

            # Per-pair weights (no batch-sum). Detach if you don't want gradients flowing back.
            vals = (self.scalar_scale * top_corr_values).detach()      # (B, K)

            # Indices (B, K)
            tgt_idx = top_corr_indices[:, :, 0].long()
            src_idx = top_corr_indices[:, :, 1].long()

            # Init outputs
            tgt_scalar = torch.zeros(B, N, device=device, dtype=vals.dtype)
            src_scalar = torch.zeros(B, N, device=device, dtype=vals.dtype)

            # Accumulate values (sums duplicates if they exist)
            tgt_scalar.scatter_add_(dim=1, index=tgt_idx, src=vals)  # (B, N)
            src_scalar.scatter_add_(dim=1, index=src_idx, src=vals)  # (B, N)

            # If you prefer a stable rule with duplicates, use amax instead:
            # tgt_scalar.scatter_reduce_(1, tgt_idx, vals, reduce='amax', include_self=False)
            # src_scalar.scatter_reduce_(1, src_idx, vals, reduce='amax', include_self=False)

            return tgt_fourier_embeddings, src_fourier_embeddings, tgt_scalar, src_scalar

        else:
            return tgt_fourier_embeddings, src_fourier_embeddings # Per-atom embeddings to be concatenated with the previous ones.
    

def scaled_dot_product(keys, queries):    
    return torch.einsum('ijk,ilk->ijl', keys,queries) / torch.sqrt( torch.tensor(keys.shape[-1]) )
    

def rescale_and_concat(keys1, keys2):
    '''
    This function is such that the scaled dot product of rescale_and_concat( keys1, keys2) equals the sum of the scaled dot product of keys1 and of keys2
            
    
    Example:    
    scaled_dot_product1 = scaled_dot_product(keys1, keys1)
    scaled_dot_product2 = scaled_dot_product(keys2, keys2)    
    
    concatenated_keys = rescale_and_concat( keys1, keys2)    
    scaled_dot_product_concat = scaled_dot_product(concatenated_keys,concatenated_keys)    
    print( (scaled_dot_product_concat - scaled_dot_product1 - scaled_dot_product2).max() )
    
    
    '''
    d1 = keys1.shape[-1]
    d2 = keys2.shape[-1]    
    scaling1 = torch.tensor(  ( (d1+d2)/d1)**(1./4) )
    scaling2 = torch.tensor(  ( (d1+d2)/d2)**(1./4) )    
    return torch.cat( (scaling1 * keys1, scaling2 * keys2) ,axis=-1)

if __name__ == '__main__':
    import matplotlib.pyplot as plt    
    recycling = RecyclingModule()
    
    B,N = 2,100 # Batch dimension, number of points    
    x1 = torch.randn(B,N,3) * 10
    x2 = x1+ torch.randn(B,N,3) * 2
    distances = torch.sqrt(torch.sum( (x1.unsqueeze(2) - x2.unsqueeze(1))**2,axis=-1) ) # B X N X N.    
    x1_embedded,x2_embedded = recycling(x1,x2)        
    dot_product_in_embedding_space = torch.sum(x1_embedded.unsqueeze(2) * x2_embedded.unsqueeze(1), axis=-1) / torch.sqrt( torch.tensor( x1_embedded.shape[-1] ))    
    plt.scatter(distances.detach().flatten(), dot_product_in_embedding_space.detach().flatten(),alpha=0.1)
    plt.show()
    
    

    keys1 = torch.randn(B,N,10)
    keys2 = torch.randn(B,N,10)
    
    scaled_dot_product_part_a = scaled_dot_product(keys1, keys2)
    scaled_dot_product_part_b = scaled_dot_product(x1_embedded, x2_embedded)    
    
    concatenated_keys_1 = rescale_and_concat( keys1, x1_embedded)    
    concatenated_keys_2 = rescale_and_concat( keys2, x2_embedded)    
    
    scaled_dot_product_concat = scaled_dot_product(concatenated_keys_1,concatenated_keys_2)    
    print( (scaled_dot_product_concat - scaled_dot_product_part_a - scaled_dot_product_part_b).max() )