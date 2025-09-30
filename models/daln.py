import torch
import torch.nn as nn
from torchvision import models
from models.layers.ReshapeLayer import ReshapeLayer  
from models.layers.ReverseLayerF import ReverseLayerF


class DALN(nn.Module):
    """ ResNet backbone + DALN (BN → DALNLayer) """
    
    def __init__(
        self, 
        backbone="resnet18",
        num_classes: int = 3,
    ):
        super(DALN, self).__init__()
        self.restored = False

        arch_params = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
        }

        if backbone not in arch_params:
            raise ValueError(f"Unsupported architecture {backbone}")

        model_resnet = arch_params[backbone](pretrained=True)

        num_features = model_resnet.fc.in_features 
        
        self.features = nn.Sequential(*list(model_resnet.children())[:-2])
        self.classifier = nn.Sequential()
        self.classifier.add_module('avgpool', nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.classifier.add_module('flatten', nn.Flatten(start_dim=1, end_dim=-1))
        self.classifier.add_module('fc', nn.Linear(num_features, num_classes))

    def forward(self, input_data):
        feature = self.features(input_data)
        class_output = self.classifier(feature)
        return class_output, feature
