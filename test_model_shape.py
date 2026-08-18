import torch
import torch.nn as nn
from timm import create_model

class HybridEfficientNetViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.efficientnet = create_model("efficientnet_b3", pretrained=False, num_classes=num_classes)
        self.vit = create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
        self.fc = nn.Sequential(nn.Linear(2 * num_classes, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.fc(torch.cat((self.efficientnet(x), self.vit(x)), dim=1))

try:
    model8 = HybridEfficientNetViT(8)
    model8.load_state_dict(torch.load("best_efficientnet_vit_model.pth", map_location="cpu", weights_only=True))
    print("Model has 8 classes")
except Exception as e:
    print("Failed 8 classes:", e)

try:
    model2 = HybridEfficientNetViT(2)
    model2.load_state_dict(torch.load("best_efficientnet_vit_model.pth", map_location="cpu", weights_only=True))
    print("Model has 2 classes")
except Exception as e:
    pass

