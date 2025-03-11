from torch import nn
import torch


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class IdentityLayer(nn.Module):
    def __init__(self):
        super(IdentityLayer, self).__init__()

    def forward(self, x):
        return x  # Simply return the input as output

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.BatchNorm1d(output_dim)  # Using BatchNorm1d for normalization
        self.activation = nn.ReLU()  # Using ReLU for activation

    def forward(self, x):
        B, N, d = x.shape
        x = self.linear(x.view(-1, d))  
        x = self.norm(x)  
        x = self.activation(x)  
        return x.view(B, N, -1) 

    
class MaskedBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(MaskedBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mask=None):
        """
        Forward pass for masked batch normalization.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (B, N, C), where C is the number of channels/features.
        - mask (torch.Tensor, optional): A tensor of shape (B, N) indicating which elements are unmasked (1 for unmasked, 0 for masked).

        Returns:
        - torch.Tensor: Normalized tensor.
        """
        B, N, C = x.shape

        if mask is not None:
            # Ensure mask is of shape (B, N) and broadcast it to match the input shape (B, N, C)
            mask = mask.unsqueeze(-1)  # Shape (B, N, 1)
            x = x * mask  # Mask the input tensor (0 for masked positions)

            # Compute mean and variance for unmasked elements
            sum_mask = mask.sum(dim=(0, 1), keepdim=True)  # Sum of valid elements for each channel
            masked_mean = x.sum(dim=(0, 1), keepdim=True) / sum_mask
            masked_var = ((x - masked_mean) ** 2).sum(dim=(0, 1), keepdim=True) / sum_mask
        else:
            # Standard BN without mask
            masked_mean = x.mean(dim=(0, 1), keepdim=True)
            masked_var = x.var(dim=(0, 1), keepdim=True, unbiased=False)

        # Normalize
        x_normalized = (x - masked_mean) / torch.sqrt(masked_var + self.eps)
        return self.weight * x_normalized + self.bias


class MaskedLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-3):
        super(MaskedLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.to(x.dtype)
            mask = mask.unsqueeze(-1)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        var = torch.clamp(var, min=self.eps)
        normalized_x = ((x - mean) / torch.sqrt(var + self.eps))
        return self.gamma * normalized_x + self.beta

    

class FeatureBlock(nn.Module):
    def __init__(self, dim, dropout, hidden_dim=None):
        super(FeatureBlock, self).__init__()
        hidden_dim = hidden_dim or dim  # Default to same size if not provided
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),  # Ensure output_dim = input_dim for skip connection
            nn.ReLU()
        )
        self.norm = MaskedLayerNorm(dim)  # Apply LayerNorm

    def forward(self, x, mask=None):
        B, N, D = x.shape  # Batch size, number of tokens, feature dim
        x = x + self.mlp(x)  # Apply MLP
        x = self.norm(x, mask)  # Apply Masked LayerNorm
        return x


class EmbeddingBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, n_blocks: int = 1, hidden_dim: int | None = None, dropout: float = 0.0, bias: bool = True,
                norm_in_last_layer: bool = False, norm_first_layer: bool = False) -> None:
        """
        Feature transformation block with multiple layers.

        Args:
            input_dim (int): 
            output_dim (int):
            n_blocks (int, optional):n feature blocks to apply. Defaults to 1.
            hidden_dim (int | None, optional):. Defaults to None.
            dropout (float, optional): . Defaults to 0.0.
            bias (bool, optional): . Defaults to True.
            norm_in_last_layer (bool, optional): Defaults to False.
            norm_first_layer (bool, optional):  Defaults to False.
        """        
        super(EmbeddingBlock, self).__init__()
        
        # Stack multiple feature transformation blocks
        self.norm_first_layer = norm_first_layer
        if norm_first_layer:
            self._first_norm = MaskedLayerNorm(input_dim)
        
        self.blocks = nn.ModuleList([FeatureBlock(input_dim, hidden_dim=hidden_dim, dropout=dropout) for _ in range(n_blocks)])
        
        # Linear transformation if output_dim differs
        self.output_layer = nn.Linear(input_dim, output_dim, bias=bias) if input_dim != output_dim else nn.Identity()
        self._norm_in_last_layer = norm_in_last_layer
        if norm_in_last_layer:
            self.norm = MaskedLayerNorm(output_dim)
        
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        """Custom weight initialization for stability"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the embedding block."
        """
        
        if self.norm_first_layer:
            x = self._first_norm(x, mask)
        for block in self.blocks:
            x = block(x, mask)  # Pass the mask along with the input tensor
        x = self.output_layer(x)
        if self._norm_in_last_layer:
            x = self.norm(x, mask)
        return x
