import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModel, AutoConfig
from models.internimage_g_22kto1k_512.modeling_internimage import InternImageModel

class DQN_InternImage(nn.Module):
    def __init__(self, num_actions, model_path="./models/internimage_g_22kto1k_512"):
        super().__init__()
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # self.backbone = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto", use_safetensors=True, local_files_only=True)
        self.backbone = InternImageModel(config)
        state_dict = {}
        if model_path.endswith("internimage_g_22kto1k_512"):
            for i in range(1, 4):
                part_path = os.path.join(model_path, f"model-0000{i}-of-00003.safetensors")
                part_dict = load_file(part_path)
                state_dict.update(part_dict)
        else:
            
            state_dict = load_file(f"{model_path}/model.safetensors")
        self.backbone.load_state_dict(state_dict, strict=False)

        dummy = torch.randn(1, 3, 224, 224) # Adjust input size if necessary
        with torch.no_grad():
            out = self.backbone(dummy)
            stage4 = out['hidden_states'][-1]
            hidden_size = stage4.shape[1]
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(hidden_size, num_actions)
        self.num_actions = num_actions
        
        for param in self.backbone.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        x = outputs['hidden_states'][-1]
        x = self.global_pool(x).squeeze(-1).squeeze(-1)
        logits = self.head(x)
        _, max_indices = torch.max(logits, dim=1)
        eta = F.one_hot(max_indices, num_classes=self.num_actions).float()
        return logits, eta