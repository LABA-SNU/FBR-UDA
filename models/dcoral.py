import torch
import torch.nn as nn
#from models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision import models
from models.layers.ReshapeLayer import ReshapeLayer  
from models.layers.ReverseLayerF import ReverseLayerF  


class DeepCORAL(nn.Module):
    """
    DeepCORAL network as defined in the paper.
    Network architecture based on following repository:
    https://github.com/SSARCandy/DeepCORAL/blob/master/models.py
    :param num_classes: int --> office dataset has 31 different classes
    """
    def __init__(self, backbone = "resnet18", num_classes = 1000):
        super(DeepCORAL, self).__init__()

        arch_params = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
        }

        if backbone not in arch_params:
            raise ValueError(f"Unsupported architecture {backbone}")

        self.sharedNetwork = arch_params[backbone](pretrained=True)
        
        num_features = self.sharedNetwork.fc.in_features
        
        self.sharedNetwork.fc = nn.Identity()
        self.fc8 = nn.Linear(num_features, num_classes) # fc8 activation
        
        # initiliaze fc8 weights according to the CORAL paper (N(0, 0.005))
        self.fc8.weight.data.normal_(0.0, 0.005)

    def forward(self, source, target): # computes activations for BOTH domains
        source_feat = self.sharedNetwork(source)
        source_output = self.fc8(source_feat)
        
        target_feat = self.sharedNetwork(target)
        target_output = self.fc8(target_feat)
        
        return source_feat, target_feat, source_output, target_output