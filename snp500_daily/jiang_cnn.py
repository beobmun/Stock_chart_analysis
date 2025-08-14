import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(5, 3), padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1))
        )
        
    def forward(self, x):
        return self.conv_layer(x)
    
class FCLayer(nn.Module):
    def __init__(self, in_features, num_actions):
        super().__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features, num_actions)
        )
        
    def forward(self, x):
        return self.fc_layer(x)
    
class JiangCNN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, 64)
        self.conv2 = ConvLayer(64, 128)
        self.conv3 = ConvLayer(128, 256)
        self.conv4 = ConvLayer(256, 512)
        self.fc = FCLayer(512 * 14 * 224, num_actions)  # Assuming input size is 224x224, adjust if different
        self.num_actions = num_actions
        
        print("JiangCNN initialized")
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        rho = self.fc(x)  # Output the action values
        _, max_indices = torch.max(rho, dim=1)
        eta = F.one_hot(max_indices, num_classes=self.num_actions).float()
        return rho, eta