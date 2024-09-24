import torch
from torch import nn


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
    

class FeatureCoordinateBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeatureCoordinateBlock, self).__init__()
        
        self.fc1 = nn.Linear(input_dim + 3, hidden_dim)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.batch_norm2 = nn.BatchNorm1d(output_dim)

    def forward(self, features, coordinates):
        combined_input = torch.cat([features, coordinates], dim=-1) 
        B, N, _ = combined_input.shape
        combined_output = self.fc1(combined_input.view(B * N, -1))
        combined_output = self.batch_norm1(combined_output)
        combined_output = self.relu(combined_output)
        
        combined_output = self.fc2(combined_output)
        combined_output = self.batch_norm2(combined_output)
        combined_output = self.relu(combined_output)

        combined_output = combined_output.view(B, N, -1)
        
        return combined_output
