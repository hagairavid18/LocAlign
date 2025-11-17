from torch import nn
import torch

from models.layers.norm import MaskedBatchNorm1d, MaskedLayerNorm


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
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        B, N, D = x.shape  # Batch size, number of tokens, feature dim
        x = x + self.mlp(x)  # Apply MLP
        x = self.norm(x)  # Apply Masked LayerNorm
        return x


class EmbeddingBlock(nn.Module):
    def __init__(
            self, 
            input_dim: int, 
            output_dim: int, 
            n_blocks: int = 1, 
            hidden_dim: int | None = None, 
            dropout: float = 0.0, 
            bias: bool = True,
            norm_in_last_layer: bool = False,
            last_norm_learnable: bool = True
            ) -> None:
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
        """        
        super(EmbeddingBlock, self).__init__()
        
        # Stack multiple feature transformation blocks
        self.blocks = nn.ModuleList([FeatureBlock(input_dim, hidden_dim=hidden_dim, dropout=dropout) for _ in range(n_blocks)])
        
        # Linear transformation if output_dim differs
        self.output_layer = nn.Linear(input_dim, output_dim, bias=bias) if input_dim != output_dim else nn.Identity()
        self._norm_in_last_layer = norm_in_last_layer
        self._last_norm_learnable = last_norm_learnable
        if norm_in_last_layer:
            if last_norm_learnable:
                self.norm = nn.LayerNorm(output_dim)
            else:
                # self.norm = MaskedLayerNorm(output_dim, learnable=last_norm_learnable)
                self.norm = MaskedBatchNorm1d(output_dim, learnable_weight=True, learnable_bias=False)
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
        
        for block in self.blocks:
            x = block(x, mask)  # Pass the mask along with the input tensor
        x = self.output_layer(x)
        if self._norm_in_last_layer:
            if self._last_norm_learnable:
                x= self.norm(x)
            else:
                x = self.norm(x, mask)
        return x