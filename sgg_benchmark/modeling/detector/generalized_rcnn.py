# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework for Scene Graph Generation.

Uses Hydra/OmegaConf configuration format.
"""
import torch
from torch import nn
from typing import List, Optional, Dict, Union, Any
from omegaconf import DictConfig

from sgg_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and relations.
    Combines the following components:
    - Backbone
    - RPN
    - ROI heads (box, relation, attribute)
    
    This is the top-level wrapper that creates a complete model for detecting
    objects and their relationships (scene graphs).
    
    The full pipeline consists of:
    1. Backbone extracts feature maps from input images
    2. RPN proposes regions of interest
    3. ROI heads process proposals to generate final
        detections / relations from it.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize Generalized R-CNN model.
        
        Args:
            cfg: Hydra configuration
        """
        super(GeneralizedRCNN, self).__init__()
        
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(
        self,
        images: Union[List[torch.Tensor], 'ImageList'],
        targets: Optional[List[Dict[str, Any]]] = None,
        logger = None,
        return_attention: bool = False,
    ) -> Union[List[Dict[str, Any]], Dict[str, torch.Tensor]]:
        """
        Forward pass of Generalized R-CNN.
        
        Arguments:
            images: Images to be processed (list of tensors or ImageList)
            targets: Ground-truth boxes present in the image (optional)
            logger: Logger for tracking metrics (optional)
            return_attention: Whether to return attention maps (optional)

        Returns:
            During training: dict[Tensor] containing the losses
            During testing: list[Dict] with additional fields like `pred_scores`, `pred_labels`, etc.
        """
        if self.roi_heads.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
            
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets, logger, return_attention=return_attention)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}
        
        if self.roi_heads.training:
            losses = {}
            losses.update(detector_losses)
            
            # During the relationship training stage, the rpn_head should be fixed, and no loss
            if not self.cfg.model.relation_on:
                losses.update(proposal_losses)
            return losses

        return result
