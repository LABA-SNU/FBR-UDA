import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import ResNet, BasicBlock
from models.layers.ReverseLayerF import ReverseLayerF   

class NuclearWassersteinDiscrepancy(nn.Module):
    def __init__(self, classifier: nn.Module):
        super(NuclearWassersteinDiscrepancy, self).__init__()
        self.classifier = classifier

    @staticmethod
    def n_discrepancy(y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        pre_s, pre_t = F.softmax(y_s, dim=1), F.softmax(y_t, dim=1)
        loss = (-torch.norm(pre_t, 'nuc') + torch.norm(pre_s, 'nuc')) / y_t.shape[0]
        return loss

    def forward(self, features_s: torch.Tensor, features_t: torch.Tensor, alpha: float) -> torch.Tensor:
        reverse_features_s = ReverseLayerF.apply(features_s, alpha)
        reverse_features_t = ReverseLayerF.apply(features_t, alpha)
        
        y_s = self.classifier(reverse_features_s)
        y_t = self.classifier(reverse_features_t)
        
        loss = self.n_discrepancy(y_s, y_t)
        return loss