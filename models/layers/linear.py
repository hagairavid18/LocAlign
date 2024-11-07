from torch import nn


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
    

class FeatureBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_blocks=3):
        """
        Initialize the FeatureBlock with configurable number of layers.

        Parameters:
        - input_dim: Dimension of the input features.
        - hidden_dim: Dimension of hidden layers.
        - output_dim: Dimension of the output features.
        - num_layers: Number of fully connected layers (default: 3).
        """
        super(FeatureBlock, self).__init__()
        
        self.num_layers = n_blocks

        # Create a list to hold layers
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.BatchNorm1d(hidden_dim))
        self.layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(n_blocks - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU())

        # Output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers.append(nn.BatchNorm1d(output_dim))
        self.layers.append(nn.ReLU())

        # Apply He Initialization
        self._initialize_weights()
        # self._initialize_weights_normal()

    def forward(self, features):
        """
        Forward pass through the block of layers.

        Parameters:
        - features: Input tensor of shape (B, N, input_dim).

        Returns:
        - Tensor of shape (B, N, output_dim) after passing through the layers.
        """
        B, N, _ = features.shape
        combined_output = features.view(B * N, -1)  # Flatten for Linear Layer

        # Pass through each layer in sequence
        for layer in self.layers:
            combined_output = layer(combined_output)

        combined_output = combined_output.view(B, N, -1)  # Reshape back to B x N x output_dim
        return combined_output

    def _initialize_weights(self):
        """
        Initialize weights of the linear layers with He initialization.
        """
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
    
    def _initialize_weights_normal(self):
        """
        Initialize weights of the linear layers with He initialization.
        """
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=1.0, std=0.01)

