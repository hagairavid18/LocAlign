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
    def __init__(self, normalized_shape, eps=1e-5):
        """
        Masked LayerNorm for tensors of shape (B, N, C).

        Parameters:
        - normalized_shape (int): The size of the feature dimension (C).
        - eps (float): A small constant to avoid division by zero.
        """
        super(MaskedLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # Scale
        self.bias = nn.Parameter(torch.zeros(normalized_shape))   # Shift

    def forward(self, x, mask=None):
        """
        Forward pass for masked layer normalization.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (B, N, C), where C is the number of features.
        - mask (torch.Tensor, optional): Mask of shape (B, N), where 1 indicates unmasked and 0 indicates masked elements.

        Returns:
        - torch.Tensor: Normalized tensor of shape (B, N, C).
        """
        # Compute mean and variance along the last dimension (C) for each sample
        if mask is not None:
            # Reshape mask to broadcast along the feature dimension
            mask = mask.unsqueeze(-1)  # Shape: (B, N, 1)
            x = x * mask  # Mask the input tensor (masked positions are zeroed)

            # Compute mean and variance for unmasked elements along the last dimension
            sum_mask = mask.sum(dim=-2, keepdim=True)  # Sum across the sequence dimension (N)
            masked_mean = (x.sum(dim=-2, keepdim=True) / sum_mask).nan_to_num(0.0)
            masked_var = ((x - masked_mean) ** 2).sum(dim=-2, keepdim=True) / sum_mask
            masked_var = masked_var.nan_to_num(0.0)  # Replace NaNs due to zero division
        else:
            # Standard LayerNorm (normalize along the last dimension)
            masked_mean = x.mean(dim=-1, keepdim=True)
            masked_var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        x_normalized = (x - masked_mean) / torch.sqrt(masked_var + self.eps)

        # Apply learnable parameters (scale and shift)
        return self.weight * x_normalized + self.bias


class FeatureBlockGPT(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim: int | None = None, n_blocks=3, 
                 activation=nn.ReLU, normalization=None,
                 dropout=0.1, skip_connection=False, gated_skip=False, norm_in_last_layer: bool = True):
        """
        FeatureBlock with flexible configurations, including skip connections, dropout, gated skip, and mask compatibility.
        """
        super(FeatureBlockGPT, self).__init__()
        self.skip_connection = skip_connection
        self.gated_skip = gated_skip
        self.dropout = nn.Dropout(dropout)

        if isinstance(activation, str):
            activations_map = {
                "ReLU": nn.ReLU,
                "Tanh": nn.Tanh,
                "LeakyReLU": nn.LeakyReLU,
                "Sigmoid": nn.Sigmoid,
            }
            activation = activations_map.get(activation, nn.ReLU)  # Default to ReLU
        elif callable(activation):
            activation = activation
        
        if isinstance(normalization, str):
            normalization_map = {
                "LayerNorm": MaskedLayerNorm,
                "BatchNorm": MaskedBatchNorm1d,
            }
            normalization = normalization_map.get(normalization)
        elif callable(normalization):
            normalization = normalization

        # Create layers
        layers = []

        if hidden_dim is None:
            # Only input-to-output layer if hidden_dim is None
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # Input layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            if normalization:
                layers.append(normalization(hidden_dim))
            layers.append(activation())
            layers.append(self.dropout)

            # Hidden layers
            for _ in range(n_blocks - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if normalization:
                    layers.append(normalization(hidden_dim))
                layers.append(activation())
                layers.append(self.dropout)

            # Output layer
            layers.append(nn.Linear(hidden_dim, output_dim))

        # Combine layers into a sequential module
        self.model = nn.Sequential(*layers)
        self._initialize_weights()

        # Gating layer for gated skip connections
        if self.gated_skip:
            self.gate = nn.Linear(input_dim, output_dim)

        # Final normalization after skip connection (if enabled)
        self.last_norm = normalization(output_dim) if norm_in_last_layer and normalization else None

    def _initialize_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, features, mask=None):
        """
        Forward pass through the block of layers, with optional mask support.
        """
        B, N, _ = features.shape
        original_features = features  # Save for skip connection if needed

        # Flatten features to (B * N, input_dim)
        combined_output = features.view(B * N, -1)

        # Forward pass through layers
        for layer in self.model:
            if isinstance(layer, (MaskedBatchNorm1d, MaskedLayerNorm)):
                combined_output = combined_output.view(B, N, -1)  # Reshape to (B, N, hidden_dim)
                combined_output = layer(combined_output, mask=mask)  # Pass mask here
            else:
                combined_output = layer(combined_output)

        # Reshape back to (B, N, output_dim)
        combined_output = combined_output.view(B, N, -1)

        # Apply skip connection if enabled and dimensions match
        if self.skip_connection:
            if self.gated_skip:
                gate_value = torch.sigmoid(self.gate(original_features.view(B * N, -1)))
                gate_value = gate_value.view(B, N, -1)
                combined_output = combined_output * gate_value + original_features * (1 - gate_value)
            else:
                combined_output = combined_output + original_features

        # Apply last normalization **after** skip connection
        if self.last_norm:
            if isinstance(self.last_norm, (MaskedBatchNorm1d, MaskedLayerNorm)):
                combined_output = self.last_norm(combined_output, mask=mask)
            else:
                combined_output = self.last_norm(combined_output)

        return combined_output
