# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Combined ROI Heads for Scene Graph Generation.

Uses Hydra/OmegaConf configuration format.
"""
import torch
from typing import List, Dict, Tuple, Optional, Any
from omegaconf import DictConfig

from .box_head.box_head import build_roi_box_head
from .attribute_head.attribute_head import build_roi_attribute_head
# from .relation_head.relation_head_dsformer import build_roi_relation_head
from .relation_head.relation_head import build_roi_relation_head


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction, attributes, relations) 
    into a single head.
    
    The combined head manages:
    - Box head: Object detection and classification
    - Attribute head: Object attribute prediction (optional)
    - Relation head: Scene graph relation prediction (optional)
    """

    def __init__(self, cfg: DictConfig, heads: List[Tuple[str, torch.nn.Module]]):
        """
        Initialize combined ROI heads.
        
        Args:
            cfg: Hydra configuration
            heads: List of (name, module) tuples for each head
        """
        super(CombinedROIHeads, self).__init__(heads)
        
        self.cfg = cfg

    def forward(
        self,
        features: List[torch.Tensor],
        proposals: List[Dict[str, Any]],
        targets: Optional[List[Dict[str, Any]]] = None,
        logger = None,
        detections: Optional[List[Dict[str, Any]]] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, torch.Tensor]]:
        """
        Forward pass through all ROI heads.
        
        Args:
            features: Feature maps from backbone
            proposals: Proposed boxes from RPN or detector
            targets: Ground-truth boxes (optional, for training)
            logger: Logger for tracking metrics (optional)
            detections: Pre-computed detections (optional)
            
        Returns:
            Tuple of (features, detections, losses):
            - features: Final feature representation
            - detections: Processed detections with predictions
            - losses: Dictionary of losses (empty during inference)
        """
        losses = {}
        
        # Get config values
        box_head_on = self.cfg.model.box_head
        relation_on = self.cfg.model.relation_on
        attribute_on = self.cfg.model.attribute_on
        
        # Box head processing
        if box_head_on or detections is None:
            x, detections, loss_box = self.box(features, proposals, targets)
        
        if not relation_on:
            # During the relationship training stage, the bbox_proposal_network should be fixed, and no loss. 
            losses.update(loss_box)

        # Attribute head processing
        if attribute_on:
            # Attribute head don't have a separate feature extractor
            z, detections, loss_attribute = self.attribute(features, detections, targets)
            losses.update(loss_attribute)

        # Relation head processing
        if relation_on:
            # it may be not safe to share features due to post processing
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_relation = self.relation(features, detections, targets, return_attention=return_attention)
            losses.update(loss_relation)

        return x, detections, losses


def build_roi_heads(cfg: DictConfig, in_channels) -> CombinedROIHeads:
    """
    Build combined ROI heads based on configuration.
    
    Constructs individual heads (box, mask, relation, attribute) and combines them
    into a single CombinedROIHeads module.
    
    Args:
        cfg: Hydra configuration
        in_channels: Number of input channels from backbone (int or list of ints for multi-scale)
        
    Returns:
        CombinedROIHeads module with requested heads
        
    Example:
        >>> roi_heads = build_roi_heads(cfg, 256)  # cfg.model.relation_on = True
        >>> roi_heads = build_roi_heads(cfg, [256, 512, 512])  # multi-scale
    """
    # individually create the heads, that will be combined together afterwards
    roi_heads = []
    
    # Get config values
    box_head_on = cfg.model.box_head
    rpn_only = cfg.model.rpn_only
    relation_on = cfg.model.relation_on
    attribute_on = cfg.model.attribute_on
    
    # Extract a single channel value for box head from multi-scale input if needed
    # The box head will get multi-scale features directly from the model
    # and uses the config's yolo.out_channels to handle different input channels
    # Convert to int explicitly to handle ListConfig objects
    try:
        if hasattr(in_channels, '__getitem__'):
            box_in_channels = int(in_channels[0])
        else:
            box_in_channels = int(in_channels)
    except (TypeError, KeyError):
        box_in_channels = int(in_channels)

    # Build individual heads
    if box_head_on and not rpn_only:
        roi_heads.append(("box", build_roi_box_head(cfg, box_in_channels)))
    
    if relation_on:
        roi_heads.append(("relation", build_roi_relation_head(cfg, in_channels)))
    
    if attribute_on:
        roi_heads.append(("attribute", build_roi_attribute_head(cfg, box_in_channels)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads

