import torch
import torch.nn as nn
from torchvision import models

class AdamatchNet(nn.Module):
    """ ResNet backbone + Adamatch (encoder + classifier 분리) """

    def __init__(self, backbone="resnet18", num_classes=3, pretrained=True):
        super(AdamatchNet, self).__init__()

        arch_params = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
        }
        if backbone not in arch_params:
            raise ValueError(f"Unsupported architecture {backbone}")

        resnet = arch_params[backbone](pretrained=pretrained)
        num_features = resnet.fc.in_features

        # encoder
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        # classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x, return_features=False):
        features = self.encoder(x)               # [B, C, H, W]
        logits = self.classifier(features)       # [B, num_classes]
        if return_features:
            return logits, features
        return logits
