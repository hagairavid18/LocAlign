import torch
from torch import nn

class CNNCombiner(nn.Module):
    def __init__(self, channels=16):
        super(CNNCombiner, self).__init__()
        self.conv1 = nn.Conv2d(2, channels, kernel_size=3, padding=1)  # 2 input channels (A & B)
        self.conv2 = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.activation = nn.ReLU()

    def forward(self, A, B):
        x = torch.stack((A, B), dim=1)  # Shape: (B, 2, N, N)
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        return x.squeeze(1)  # Output: (B, N, N)