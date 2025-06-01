import torch.nn as nn
import torch

class SecureCNN(nn.Module):
    """FHE-compatible CNN model"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)  # Reduced from 32
        self.conv2 = nn.Conv2d(16, 32, 3, 1)  # Reduced from 64
        self.fc1 = nn.Linear(32 * 5 * 5, 64)  # 32 * 5 * 5 = 800, reduced from 128
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        return self.fc2(x)