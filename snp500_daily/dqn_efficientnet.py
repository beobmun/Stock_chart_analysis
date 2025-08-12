import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

class DQN_EfficientNet(nn.Module):
    def __init__(self, num_actions, pretrained=True):
        super().__init__()
        self.num_actions = num_actions
        if pretrained:
            self.efficientnet = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
            print("Using pretrained EfficientNet-B4 weights.")
        else:
            self.efficientnet = efficientnet_b4(weights=None)
            print("Using untrained EfficientNet-B4.")
            
        in_features = self.efficientnet.classifier[-1].in_features
        self.efficientnet.classifier[-1] = nn.Linear(in_features, num_actions)

        for param in self.efficientnet.parameters():
            param.requires_grad = True

    def forward(self, x):
        rho = self.efficientnet(x)
        _, max_indices = torch.max(rho, dim=1)
        eta = F.one_hot(max_indices, num_classes=self.num_actions).float()
        return rho, eta