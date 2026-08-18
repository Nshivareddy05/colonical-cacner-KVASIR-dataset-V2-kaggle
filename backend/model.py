import torch
import torch.nn as nn
from timm import create_model

class HybridEfficientNetViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.efficientnet = create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=num_classes
        )

        self.vit = create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=num_classes
        )

        self.fc = nn.Sequential(
            nn.Linear(2 * num_classes, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        efficientnet_output = self.efficientnet(x)
        vit_output = self.vit(x)

        combined = torch.cat(
            (efficientnet_output, vit_output),
            dim=1
        )

        return self.fc(combined)

class EfficientNetBranch(nn.Module):
    def __init__(self, hybrid_model):
        super().__init__()

        self.efficientnet = hybrid_model.efficientnet
        self.vit = hybrid_model.vit
        self.fc = hybrid_model.fc

    def forward(self, x):
        efficientnet_output = self.efficientnet(x)
        vit_output = self.vit(x)

        combined = torch.cat(
            (efficientnet_output, vit_output),
            dim=1
        )

        return self.fc(combined)
