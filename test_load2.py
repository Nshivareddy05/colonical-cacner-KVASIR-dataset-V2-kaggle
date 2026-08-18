import torch
import torch.nn as nn
from timm import create_model

class HybridEfficientNetViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.efficientnet = create_model("efficientnet_b3", pretrained=False, num_classes=num_classes)
        self.vit = create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
        fc_in_features = self.efficientnet.classifier.in_features + self.vit.head.in_features
        self.efficientnet.classifier = nn.Identity()
        self.vit.head = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(fc_in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
model = HybridEfficientNetViT(num_classes=6)
try:
    model.load_state_dict(torch.load('best_efficientnet_vit_model.pth', map_location='cpu'))
    print("Successfully loaded with 6 classes!")
except Exception as e:
    print("Error loading:", str(e))
