import torch
import torch.nn as nn
#from models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision import models


class VanilaResNet(nn.Module):
    def __init__(self, backbone = 'resnet18', num_classes = 1000):
        super(VanilaResNet, self).__init__()

        arch_params = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
        }

        if backbone not in arch_params:
            raise ValueError(f"Unsupported architecture {backbone}")

        self.resnet_model = arch_params[backbone](pretrained=True)
        
        num_features = self.resnet_model.fc.in_features
        
        self.resnet_model.fc = nn.Linear(num_features, num_classes)

    def forward(self, source): # computes activations for BOTH domains
        
        output = self.resnet_model(source)
        
        return 1, output