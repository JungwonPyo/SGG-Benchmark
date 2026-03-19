# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Backbone network builders for the Scene Graph Generation model.
This module provides factory functions to build various backbone architectures.
Supports Hydra/OmegaConf configuration only.
"""
from collections import OrderedDict
from torch import nn
import torch
from omegaconf import DictConfig

from sgg_benchmark.modeling import registry
from sgg_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module
from . import resnet
from .yolo import YoloModel
from .yoloworld import YoloWorldModel
from .yoloe import YOLOEDetectionModel

@registry.BACKBONES.register("dinov2")
def build_dinov2_backbone(cfg: DictConfig) -> nn.Module:
    """Build DINOv2 ViT-B/14 backbone.
    
    Args:
        cfg: Hydra/OmegaConf DictConfig
        
    Returns:
        DINOv2 model with out_channels attribute
    """
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model.out_channels = 768
    return model


@registry.BACKBONES.register("yolo")
def build_yolo_backbone(cfg: DictConfig) -> nn.Module:
    """Build YOLO backbone (YOLO11, YOLO12, etc.).
    
    Args:
        cfg: Hydra/OmegaConf DictConfig
        
    Returns:
        YOLO model with out_channels attribute
    """
    nc = cfg.model.roi_box_head.num_classes - 1
    model = YoloModel(cfg, nc=nc)
    model.out_channels = cfg.model.yolo.out_channels
    return model


@registry.BACKBONES.register("yoloworld")
def build_yoloworld_backbone(cfg: DictConfig) -> nn.Module:
    """Build YOLO-World backbone (open-vocabulary detection).
    
    Args:
        cfg: Hydra/OmegaConf DictConfig
        
    Returns:
        YOLO-World model with out_channels attribute
    """
    nc = cfg.model.roi_box_head.num_classes - 1
    model = YoloWorldModel(cfg, nc=nc)
    model.out_channels = cfg.model.yolo.out_channels
    return model


@registry.BACKBONES.register("yoloe")
def build_yoloe_backbone(cfg: DictConfig) -> nn.Module:
    """Build YOLO-E backbone.
    
    Args:
        cfg: Hydra/OmegaConf DictConfig
        
    Returns:
        YOLO-E model with out_channels attribute
    """
    nc = cfg.model.roi_box_head.num_classes - 1
    model = YOLOEDetectionModel(cfg, nc=nc)
    model.out_channels = cfg.model.yolo.out_channels
    return model


@registry.BACKBONES.register("yolov5")
def build_yolov5_backbone(cfg: DictConfig) -> nn.Module:
    """Build YOLOv5 backbone.
    
    Args:
        cfg: Hydra/OmegaConf DictConfig
        
    Returns:
        YOLOv5 model with out_channels attribute
    """
    nc = cfg.model.roi_box_head.num_classes - 1
    model = YoloModel(cfg, nc=nc)
    model.out_channels = cfg.model.yolo.out_channels
    return model

@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg: DictConfig) -> nn.Module:
    """Build ResNet backbone without FPN.
    
    Args:
        cfg: Hydra/OmegaConf DictConfig
        
    Returns:
        ResNet model with out_channels attribute
    """
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.model.resnets.backbone_out_channels
    return model


@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg: DictConfig) -> nn.Module:
    """Build ResNet backbone with Feature Pyramid Network (FPN).
    
    Args:
        cfg: Hydra/OmegaConf DictConfig
        
    Returns:
        ResNet+FPN model with out_channels attribute
    """
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.model.resnets.res2_out_channels
    out_channels = cfg.model.resnets.backbone_out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg, cfg.model.fpn.use_gn, cfg.model.fpn.use_relu
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


def build_backbone(cfg: DictConfig) -> nn.Module:
    """Build backbone network based on configuration.
    
    Main entry point for backbone construction. Supports:
    - YOLO (all variants: YOLO11, YOLO12, YOLOv5, YOLO-World, YOLO-E)
    - ResNet (with/without FPN)
    - DINOv2
    
    Args:
        cfg: Hydra/OmegaConf DictConfig
        
    Returns:
        Backbone model with out_channels attribute
        
    Raises:
        AssertionError: If backbone type not registered
    """
    if cfg.model.backbone.type is not None:
        assert cfg.model.backbone.type in registry.BACKBONES, \
            f"cfg.model.backbone.type: {cfg.model.backbone.type} is not registered in registry. " \
            f"Available: {list(registry.BACKBONES.keys())}"
        return registry.BACKBONES[cfg.model.backbone.type](cfg)
