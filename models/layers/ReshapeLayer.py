import torch
import torch.nn as nn
from torch.autograd import Function

class ReshapeLayer(nn.Module):
    def __init__(self, *dims):
        super(ReshapeLayer, self).__init__()
        self.dims = dims

    def forward(self, x):
        ch_num = x.size()[1]
        return x.permute(self.dims).reshape(-1, ch_num)
