import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvLayer(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=5, stride=1, padding=1):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
    def forward(self, x):
        return self.conv_layer(x)
    
class FCLayer(nn.Module):
    def __init__(self, in_features, num_actions):
        super().__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_actions)
        )
    
    def forward(self, x):
        return self.fc_layer(x)
    
class CNN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, 32, 64)
        self.conv2 = ConvLayer(64, 128, 256)
        # self.fc = FCLayer(256 * 53 * 53, num_actions)  # 224 x 224 input
        self.fc = FCLayer(256 * 29 * 29, num_actions) # 128 x 128 input
        # self.fc = FCLayer(256 * 13 * 13, num_actions) # 64 x 64 input
        self.num_actions = num_actions
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        rho = self.fc(x)  # Output the action values
        _, max_indices = torch.max(rho, dim=1)
        eta = F.one_hot(max_indices, num_classes=self.num_actions).int()
        return rho, eta