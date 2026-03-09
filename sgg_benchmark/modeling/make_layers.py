# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Utility functions for creating layers.

Uses Hydra/OmegaConf configuration format.
"""
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Callable
from omegaconf import DictConfig

from sgg_benchmark.layers import Conv2d

def get_group_gn(dim: int, dim_per_gp: int, num_groups: int) -> int:
    """
    Get number of groups used by GroupNorm, based on number of channels.
    
    Args:
        dim: Total number of channels
        dim_per_gp: Channels per group (mutually exclusive with num_groups)
        num_groups: Number of groups (mutually exclusive with dim_per_gp)
        
    Returns:
        Number of groups for GroupNorm
    """
    assert dim_per_gp == -1 or num_groups == -1, \
        "GroupNorm: can only specify G or C/G."

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0, \
            "dim: {}, dim_per_gp: {}".format(dim, dim_per_gp)
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0, \
            "dim: {}, num_groups: {}".format(dim, num_groups)
        group_gn = num_groups

    return group_gn


def group_norm(cfg: DictConfig, out_channels: int, affine: bool = True, divisor: int = 1) -> nn.GroupNorm:
    """
    Create GroupNorm layer with parameters from config.
    
    Args:
        cfg: Hydra configuration
        out_channels: Number of output channels
        affine: Whether to use learnable affine parameters
        divisor: Divisor for reducing channels
        
    Returns:
        GroupNorm layer
    """
    out_channels = out_channels // divisor
    
    # Get from Hydra config
    dim_per_gp = cfg.model.group_norm.dim_per_gp // divisor
    num_groups = cfg.model.group_norm.num_groups // divisor
    eps = cfg.model.group_norm.epsilon
    
    return torch.nn.GroupNorm(
        get_group_gn(out_channels, dim_per_gp, num_groups),
        out_channels,
        eps,
        affine
    )


def make_conv3x3(
    cfg: DictConfig = None,
    in_channels: int = None,
    out_channels: int = None,
    dilation: int = 1,
    stride: int = 1,
    use_gn: bool = False,
    use_relu: bool = False,
    kaiming_init: bool = True
) -> Union[Conv2d, nn.Sequential]:
    """
    Create a 3x3 convolution layer with optional GroupNorm and ReLU.
    
    Args:
        cfg: Hydra configuration (required if use_gn=True)
        in_channels: Number of input channels
        out_channels: Number of output channels
        dilation: Dilation rate
        stride: Stride
        use_gn: Whether to use GroupNorm
        use_relu: Whether to add ReLU activation
        kaiming_init: Whether to use Kaiming initialization
        
    Returns:
        Conv2d layer or Sequential module with conv + optional norm/activation
    """
    conv = Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False if use_gn else True
    )
    if kaiming_init:
        nn.init.kaiming_normal_(
            conv.weight, mode="fan_out", nonlinearity="relu"
        )
    else:
        torch.nn.init.normal_(conv.weight, std=0.01)
    if not use_gn:
        nn.init.constant_(conv.bias, 0)
    module = [conv,]
    if use_gn:
        if cfg is None:
            raise ValueError("cfg is required when use_gn=True")
        module.append(group_norm(cfg, out_channels))
    if use_relu:
        module.append(nn.ReLU(inplace=True))
    if len(module) > 1:
        return nn.Sequential(*module)
    return conv


def make_fc(cfg: DictConfig = None, dim_in: int = None, hidden_dim: int = None, use_gn: bool = False) -> Union[nn.Linear, nn.Sequential]:
    """
    Create a fully connected layer with optional GroupNorm.
    
    Caffe2 implementation uses XavierFill, which corresponds to kaiming_uniform_ in PyTorch.
    
    Args:
        cfg: Hydra configuration (required if use_gn=True)
        dim_in: Input dimension
        hidden_dim: Hidden dimension (output)
        use_gn: Whether to use GroupNorm
        
    Returns:
        Linear layer or Sequential module with linear + GroupNorm
    """
    if use_gn:
        if cfg is None:
            raise ValueError("cfg is required when use_gn=True")
        fc = nn.Linear(dim_in, hidden_dim, bias=False)
        nn.init.kaiming_uniform_(fc.weight, a=1)
        return nn.Sequential(fc, group_norm(cfg, hidden_dim))
    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=1)
    nn.init.constant_(fc.bias, 0)
    return fc


def conv_with_kaiming_uniform(cfg: DictConfig = None, use_gn: bool = False, use_relu: bool = False) -> Callable:
    """
    Create a conv layer factory function with Kaiming uniform initialization.
    
    Args:
        cfg: Hydra configuration (required if use_gn=True)
        use_gn: Whether to use GroupNorm
        use_relu: Whether to add ReLU activation
        
    Returns:
        Function that creates conv layers with specified properties
    """
    def make_conv(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1
    ) -> Union[Conv2d, nn.Sequential]:
        """
        Create a conv layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size
            stride: Stride
            dilation: Dilation rate
            
        Returns:
            Conv2d layer or Sequential module
        """
        conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            bias=False if use_gn else True
        )
        # Caffe2 implementation uses XavierFill, which in fact
        # corresponds to kaiming_uniform_ in PyTorch
        nn.init.kaiming_uniform_(conv.weight, a=1)
        if not use_gn:
            nn.init.constant_(conv.bias, 0)
        module = [conv,]
        if use_gn:
            if cfg is None:
                raise ValueError("cfg is required when use_gn=True")
            module.append(group_norm(cfg, out_channels))
        if use_relu:
            module.append(nn.ReLU(inplace=True))
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv

    return make_conv
