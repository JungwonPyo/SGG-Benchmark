# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Detection model builders for Scene Graph Generation.

Uses Hydra/OmegaConf configuration format.
"""
import torch.nn as nn
from omegaconf import DictConfig

from .generalized_rcnn import GeneralizedRCNN
from .generalized_yolo import GeneralizedYOLO
from .simrel_rcnn import SimrelRCNN


_DETECTION_META_ARCHITECTURES = {
    "GeneralizedRCNN": GeneralizedRCNN,
    "GeneralizedYOLO": GeneralizedYOLO,
    "SparseRCNN": SimrelRCNN
}


def build_detection_model(cfg: DictConfig) -> nn.Module:
    """
    Build detection model based on meta-architecture type.
    
    Supports:
    - GeneralizedRCNN: Faster R-CNN with RPN
    - GeneralizedYOLO: YOLO-based detection
    - SparseRCNN: Sparse R-CNN variant
    
    Args:
        cfg: Hydra configuration with model.meta_architecture key
        
    Returns:
        Detection model (nn.Module)
        
    Raises:
        KeyError: If meta-architecture not found
        
    Example:
        >>> model = build_detection_model(cfg)  # cfg.model.meta_architecture = "GeneralizedYOLO"
    """
    meta_arch_name = cfg.model.meta_architecture
    
    if meta_arch_name not in _DETECTION_META_ARCHITECTURES:
        raise KeyError(
            f"Meta-architecture '{meta_arch_name}' not found. "
            f"Available: {list(_DETECTION_META_ARCHITECTURES.keys())}"
        )
    
    meta_arch = _DETECTION_META_ARCHITECTURES[meta_arch_name]
    return meta_arch(cfg)

