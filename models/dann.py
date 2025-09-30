import torch
import torch.nn as nn
#from models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision import models
from models.layers.ReshapeLayer import ReshapeLayer  
from models.layers.ReverseLayerF import ReverseLayerF 
from collections import OrderedDict


class DANN(nn.Module):
    """ ResNet pretrained on imagenet"""

    def __init__(
        self, 
        backbone = 'resnet18',
        num_classes : int = 3,
        reshape = True
    ):
        
        super(DANN, self).__init__()
        self.restored = False
        self.reshape = reshape

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
        
        # Separate the features and the classifier
        self.features = nn.Sequential(*list(model_resnet.children())[:-2])
        #self.features = nn.Sequential(OrderedDict(list(model_resnet.named_children())[:-2]))

        self.classifier = nn.Sequential()
        self.classifier.add_module('avgpool', nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.classifier.add_module('flatten', nn.Flatten(start_dim=1, end_dim=-1))
        self.classifier.add_module('fc', nn.Linear(num_features, num_classes))
        
        # Discriminator can be added here if needed
        self.discriminator = nn.Sequential()
        
        if self.reshape:
            self.discriminator.add_module('reshape', ReshapeLayer(0,2,3,1))
        else:
            self.discriminator.add_module('avgpool', nn.AdaptiveAvgPool2d((1,1)))
            self.discriminator.add_module('flatten', nn.Flatten())
            
        self.discriminator.add_module('linear1', nn.Linear(num_features, int(num_features/2)))
        self.discriminator.add_module('relu', nn.ReLU(inplace = True))
        self.discriminator.add_module('linear2', nn.Linear(int(num_features/2), 2))

    
    def forward(self, input_data, alpha):
        feature = self.features(input_data)
        class_output = self.classifier(feature)

        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_output = self.discriminator(reverse_feature)


        return class_output, domain_output