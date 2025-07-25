import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class DQN_ResNet50(nn.Module):
    def __init__(self, num_actions, pretrained=True):
        super().__init__()
        self.num_actions = num_actions
        if pretrained:
            self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
            print("Using pretrained ResNet50 weights.")
        else:
            self.resnet = resnet50()
            print("Using untrained ResNet50.")

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_actions)
        
        for param in self.resnet.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        rho = self.resnet(x)
        _, max_indices = torch.max(rho, dim=1)
        eta = F.one_hot(max_indices, num_classes=self.num_actions).float()
        return rho, eta