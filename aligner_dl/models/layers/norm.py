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
    def __init__(self, normalized_shape, eps=1e-3, learnable=True):
        super(MaskedLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        if learnable:
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_buffer('gamma', torch.ones(normalized_shape))
            self.register_buffer('beta', torch.zeros(normalized_shape))

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(-1)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        var = torch.clamp(var, min=self.eps)
        normalized_x = ((x - mean) / torch.sqrt(var + self.eps))
        return self.gamma * normalized_x + self.beta