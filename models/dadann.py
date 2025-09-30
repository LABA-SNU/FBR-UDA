import torch
import torch.nn as nn
#from models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision import models

from models.layers.ReshapeLayer import ReshapeLayer  
from models.layers.ReverseLayerF import ReverseLayerF
from models.layers.da_attention import _PositionAttentionModule, _ChannelAttentionModule


class DADANN(nn.Module):
    """ ResNet pretrained on imagenet"""

    def __init__(
        self, 
        backbone = 'resnet18',
        num_classes : int = 3,
        reshape = True
    ):
        
        super(DADANN, self).__init__()
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
        
        # Separate the features and the classifier
        self.features = nn.Sequential(*list(model_resnet.children())[:-2])

        self.pam = PAM_Module(num_features)
        self.cam = CAM_Module(num_features)
        
        self.classifier = nn.Sequential()
        self.classifier.add_module('avgpool', nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.classifier.add_module('flatten', nn.Flatten(start_dim=1, end_dim=-1))
        self.classifier.add_module('fc', nn.Linear(num_features, num_classes))

        # Discriminator can be added here if needed
        self.discriminator = nn.Sequential()
        
        if reshape:
            self.discriminator.add_module('reshape', ReshapeLayer(0,2,3,1))
        else:
            self.discriminator.add_module('avgpool', nn.AdaptiveAvgPool2d((1,1)))
            self.discriminator.add_module('flatten', nn.Flatten())
            
        self.discriminator.add_module('linear1', nn.Linear(num_features, int(num_features/2)))
        self.discriminator.add_module('relu', nn.ReLU(inplace = True))
        self.discriminator.add_module('linear2', nn.Linear(int(num_features/2), 2))

        # Initialize the classifier and discriminator weights
        #self._initialize_weights(self.classifier)
        #self._initialize_weights(self.discriminator)

    def _initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, input_data, alpha):
        feature = self.features(input_data)

        pam_feature = self.pam(feature)
        cam_feature = self.cam(feature)
        combined_feature = (pam_feature + cam_feature) / 2

        class_output = self.classifier(combined_feature)
        reverse_feature = ReverseLayerF.apply(combined_feature, alpha)
        domain_output = self.discriminator(reverse_feature)

        return class_output, domain_output