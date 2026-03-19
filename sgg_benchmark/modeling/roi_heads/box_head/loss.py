import torch
from torch.nn import functional as F
from typing import List, Dict, Any

from sgg_benchmark.layers import smooth_l1_loss
from sgg_benchmark.modeling.box_coder import BoxCoder
from sgg_benchmark.modeling.matcher import Matcher
from sgg_benchmark.structures.box_ops import box_iou
from sgg_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from sgg_benchmark.modeling.utils import cat


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(self, cls_agnostic_bbox_reg=False):
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def assign_label_to_proposals(self, proposals: List[Dict[str, Any]], targets: List[Dict[str, Any]], attris):
        for img_idx, (target, proposal) in enumerate(zip(targets, proposals)):
            match_quality_matrix = box_iou(target["boxes"], proposal["boxes"], target["mode"], proposal["mode"])
            matched_idxs = self.proposal_matcher(match_quality_matrix)
            
            labels_per_image = target["labels"][matched_idxs.clamp(min=0)].clone().to(dtype=torch.int64)
            labels_per_image[matched_idxs < 0] = 0
            proposals[img_idx]["labels"] = labels_per_image

            if attris and "attributes" in target:
                attris_per_image = target["attributes"][matched_idxs.clamp(min=0)].clone().to(dtype=torch.int64)
                attris_per_image[matched_idxs < 0, :] = 0
                proposals[img_idx]["attributes"] = attris_per_image
        return proposals


    def __call__(self, class_logits, box_regression, proposals: List[Dict[str, Any]]):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])
            proposals (list[Dict])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        labels = cat([proposal["labels"] for proposal in proposals], dim=0)
        regression_targets = cat([proposal["regression_targets"] for proposal in proposals], dim=0)

        classification_loss = F.cross_entropy(class_logits, labels.long())

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3], device=device)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss


def make_roi_box_loss_evaluator(cfg):
    cls_agnostic_bbox_reg = cfg.model.cls_agnostic_bbox_reg

    loss_evaluator = FastRCNNLossComputation(cls_agnostic_bbox_reg)

    return loss_evaluator
