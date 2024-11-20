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

class FeatureBlockOrig(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_blocks=3, skip_connection=True ):
        """
        Initialize the FeatureBlock with configurable number of layers.

        Parameters:
        - input_dim: Dimension of the input features.
        - hidden_dim: Dimension of hidden layers.
        - output_dim: Dimension of the output features.
        - num_layers: Number of fully connected layers (default: 3).
        """
        super(FeatureBlockOrig, self).__init__()
        
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
    
    def _initialize_weights(self):
        """
        Initialize weights of the linear layers with He initialization.
        """
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

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
        combined_output = combined_output + features
        return combined_output

class FeatureBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_blocks=3, 
                 activation=nn.ReLU, normalization=nn.BatchNorm1d, skip_connection=False):
        """
        Initialize the FeatureBlock with a configurable number of layers and optional skip connection.

        Parameters:
        - input_dim (int): Dimension of the input features.
        - hidden_dim (int): Dimension of hidden layers.
        - output_dim (int): Dimension of the output features.
        - n_blocks (int): Number of fully connected blocks (default: 3).
        - activation (nn.Module): Activation function (default: nn.ReLU).
        - normalization (nn.Module): Normalization function (default: nn.BatchNorm1d).
        - skip_connection (bool): Whether to use a skip connection (default: False).
        """
        super(FeatureBlock, self).__init__()

        self.skip_connection = skip_connection
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if normalization is not None:
            layers.append(normalization(hidden_dim))
        layers.append(activation())

        # Hidden layers
        for _ in range(n_blocks - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if normalization is not None:
                layers.append(normalization(hidden_dim))
            layers.append(activation())

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        # Combine layers into a sequential module
        self.model = nn.Sequential(*layers)

        # Apply weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, features):
        """
        Forward pass through the block of layers with optional skip connection.

        Parameters:
        - features (torch.Tensor): Input tensor of shape (B, N, input_dim).

        Returns:
        - torch.Tensor: Output tensor of shape (B, N, output_dim) after passing through the layers.
        """
        B, N, _ = features.shape
        combined_output = features.view(B * N, -1)  # Flatten for Linear Layer

        # Forward pass through layers
        combined_output = self.model(combined_output)

        # Reshape back to B x N x output_dim
        combined_output = combined_output.view(B, N, -1)

        # Apply skip connection if dimensions match and skip_connection is True
        if self.skip_connection and self.input_dim == self.output_dim:
            combined_output = combined_output + features

        return combined_output
    

class FeatureBlockGPT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_blocks=3, 
                 activation=nn.ReLU, normalization=nn.BatchNorm1d,
                 dropout=0.1, skip_connection=False, gated_skip=False):
        """
        FeatureBlock with flexible configurations, including skip connections, dropout, and gated skip.

        Parameters:
        - input_dim (int): Dimension of the input features.
        - hidden_dim (int): Dimension of hidden layers.
        - output_dim (int): Dimension of the output features.
        - n_blocks (int): Number of fully connected blocks (default: 3).
        - activation (nn.Module): Activation function (default: nn.ReLU).
        - normalization (nn.Module): Normalization function (default: nn.LayerNorm).
        - dropout (float): Dropout rate (default: 0.1).
        - skip_connection (bool): Whether to add a skip connection (default: False).
        - gated_skip (bool): Whether to use a learned gated skip connection.
        """
        super(FeatureBlockGPT, self).__init__()
        self.skip_connection = skip_connection
        self.gated_skip = gated_skip
        self.dropout = nn.Dropout(dropout)

        # Create layers
        layers = []

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
        layers.append(normalization(hidden_dim))
        layers.append(activation())

        # Combine layers into a sequential module
        self.model = nn.Sequential(*layers)

        # Apply weight initialization
        self._initialize_weights()

        # Gating layer for gated skip connections
        if self.gated_skip:
            self.gate = nn.Linear(input_dim, output_dim)

    def _initialize_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, features):
        """
        Forward pass through the block of layers.

        Parameters:
        - features (torch.Tensor): Input tensor of shape (B, N, input_dim).

        Returns:
        - torch.Tensor: Output tensor of shape (B, N, output_dim) after passing through the layers.
        """
        B, N, _ = features.shape
        original_features = features  # Save for skip connection if needed
        combined_output = features.view(B * N, -1)  # Flatten for Linear Layer

        # Forward pass through layers
        combined_output = self.model(combined_output)

        # Reshape back to B x N x output_dim
        combined_output = combined_output.view(B, N, -1)

        # Apply skip connection if enabled and dimensions match
        if self.skip_connection:
            if self.gated_skip:
                # Gated skip connection
                gate_value = torch.sigmoid(self.gate(original_features.view(B * N, -1)))
                gate_value = gate_value.view(B, N, -1)
                combined_output = combined_output * gate_value + original_features * (1 - gate_value)
            else:
                # Standard skip connection
                combined_output = combined_output + original_features

        return combined_output

