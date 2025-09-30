import torch
import torch.nn as nn

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Saturday Feb 25 2020

@authors: Alan Preciado, Santosh Muthireddy
"""

def DDC_loss(source_activation, target_activation):
    """
    From the paper, the loss used is the maximum mean discrepancy (MMD)
    :param source: torch tensor: source data (Ds) with dimensions DxNs
    :param target: torch tensor: target data (Dt) with dimensons DxNt
    """
    """
    min_length = min(source_activation.size(0), target_activation.size(0))

    source_activation_trimmed = source_activation[:min_length, :]
    target_activation_trimmed = target_activation[:min_length, :]
    
    diff_domains = source_activation_trimmed - target_activation_trimmed
    
    loss = torch.mean(torch.mm(diff_domains, torch.transpose(diff_domains, 0, 1)))
    """
    diff_domains = torch.mean(source_activation,0) - torch.mean(target_activation,0)
    loss = torch.dot(diff_domains, diff_domains.T)
    return loss


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Calculate the Gaussian kernel between two samples.

    :param source: torch tensor: source data (DxNs)
    :param target: torch tensor: target data (DxNt)
    :param kernel_mul: float: a multiplicative factor for the kernel bandwidth
    :param kernel_num: int: number of Gaussian kernels to use
    :param fix_sigma: float: optional fixed bandwidth for the kernels
    :return: torch tensor: the calculated kernel value
    """
    n_samples = source.size(0) + target.size(0)
    total = torch.cat([source, target], dim=0)
    
    # Calculate the pairwise L2 distance
    L2_distance = torch.cdist(total, total, p=2)
    
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    
    kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
    return sum(kernel_val)

def MMD_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Calculate the Maximum Mean Discrepancy (MMD) loss between two domains.

    :param source: torch tensor: source data (Ds) with dimensions DxNs
    :param target: torch tensor: target data (Dt) with dimensions DxNt
    :param kernel_mul: float: a multiplicative factor for the kernel bandwidth
    :param kernel_num: int: number of Gaussian kernels to use
    :param fix_sigma: float: optional fixed bandwidth for the kernels
    :return: torch tensor: the MMD loss
    """
    source_kernel = gaussian_kernel(source, source, kernel_mul, kernel_num, fix_sigma)
    target_kernel = gaussian_kernel(target, target, kernel_mul, kernel_num, fix_sigma)
    cross_kernel = gaussian_kernel(source, target, kernel_mul, kernel_num, fix_sigma)
    
    # Calculate the means
    source_mean = torch.mean(source_kernel)
    target_mean = torch.mean(target_kernel)
    cross_mean = torch.mean(cross_kernel)
    
    # MMD loss
    loss = source_mean + target_mean - 2 * cross_mean
    return loss