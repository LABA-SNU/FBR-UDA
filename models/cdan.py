import torch
import torch.nn as nn
#from models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision import models
from models.layers.ReverseLayerF import ReverseLayerF  
from models.layers.RandomLayer import RandomLayer

def compute_and_apply_entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    entropy = 1.0 + torch.exp(-entropy)
    entropy = entropy / torch.sum(entropy).detach().item()
    return input_ * entropy.unsqueeze(1)

class CDAN(nn.Module):
    """ ResNet pretrained on imagenet"""

    def __init__(
        self, 
        backbone = "resnet18",
        num_classes : int = 3,
        random_layer_dim : int = 1024, 
    ):

        self.num_classes = num_classes
        
        super(CDAN, self).__init__()
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

        self.classifier = nn.Sequential()
        self.classifier.add_module('avgpool', nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.classifier.add_module('flatten', nn.Flatten(start_dim=1, end_dim=-1))
        self.classifier.add_module('fc', nn.Linear(num_features, self.num_classes))
        
        self.discriminator = nn.Sequential()
        self.discriminator.add_module('linear1', nn.Linear(num_features*self.num_classes, num_features))
        self.discriminator.add_module('relu1', nn.ReLU(inplace = True))
        self.discriminator.add_module('linear2', nn.Linear(num_features, int(num_features/2)))
        self.discriminator.add_module('relu2', nn.ReLU(inplace = True))
        self.discriminator.add_module('fc', nn.Linear(int(num_features/2), 2))

        # Initialize RandomLayer
        self.random_layer = RandomLayer([num_features, num_classes], random_layer_dim)

    def forward(self, input_data, alpha, use_entropy = False):
        feature = self.features(input_data) # [1, 512, 7, 7]
        class_feature = self.classifier.avgpool(feature) # [1, 512, 1, 1]
        class_feature = self.classifier.flatten(class_feature) # [1, 512]

        feature_dim = class_feature.size(1)
        
        class_output = self.classifier.fc(class_feature)

        # CDAN: Feature-Classifier 상호 정보를 생성
        softmax_output = nn.functional.softmax(class_output, dim=1)

        # Entropy 계산 및 적용
        if use_entropy:
            softmax_output = compute_and_apply_entropy(softmax_output)

        # Dimension check for conditioning strategy
        if feature_dim * self.num_classes <= 4096:
            # Use multilinear map (tensor product)
            op_out = torch.bmm(softmax_output.unsqueeze(2), class_feature.unsqueeze(1))
            op_out = op_out.view(-1, feature_dim * self.num_classes)
        else:
            # Use randomized multilinear map
            random_output = self.random_layer([class_feature, softmax_output])
            op_out = random_output.view(-1, random_output.size(1))
        
        reverse_feature = ReverseLayerF.apply(op_out, alpha)
        domain_output = self.discriminator(reverse_feature)
        
        return class_output, domain_output