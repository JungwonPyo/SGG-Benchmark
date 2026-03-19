import torch
from torch.nn import functional as F
from typing import List, Dict, Any, Tuple, Optional

from sgg_benchmark.modeling.box_coder import BoxCoder
from sgg_benchmark.modeling.matcher import Matcher
from sgg_benchmark.structures.box_ops import box_iou, filter_instances
from sgg_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from sgg_benchmark.modeling.utils import cat


class FastRCNNSampling(object):
    """
    Sampling RoIs
    """
    def __init__(
        self,
        proposal_matcher,
        fg_bg_sampler,
        box_coder,
        use_attris=False,
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.use_attributes = use_attris

    def match_targets_to_proposals(self, proposal: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
        match_quality_matrix = box_iou(target["boxes"], proposal["boxes"], target["mode"], proposal["mode"])
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        
        # get the targets corresponding GT for each proposal
        matched_targets = {
            "boxes": target["boxes"][matched_idxs.clamp(min=0)],
            "labels": target["labels"][matched_idxs.clamp(min=0)],
            "image_size": target["image_size"],
            "mode": target["mode"],
            "matched_idxs": matched_idxs
        }
        if "attributes" in target:
            matched_targets["attributes"] = target["attributes"][matched_idxs.clamp(min=0)]
            
        return matched_targets

    def prepare_targets(self, proposals: List[Dict[str, Any]], targets: List[Dict[str, Any]]):
        labels = []
        attributes = []
        regression_targets = []
        matched_idxs = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs_per_image = matched_targets["matched_idxs"]
            labels_per_image = matched_targets["labels"].clone().to(dtype=torch.int64)
            
            # Label background (below the low threshold)
            bg_inds = matched_idxs_per_image == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0
            
            if self.use_attributes and "attributes" in matched_targets:
                attris_per_image = matched_targets["attributes"].clone().to(dtype=torch.int64)
                attris_per_image[bg_inds,:] = 0
                attributes.append(attris_per_image)

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs_per_image == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets["boxes"], proposals_per_image["boxes"]
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
            matched_idxs.append(matched_idxs_per_image)

        return labels, regression_targets, matched_idxs, attributes

    def subsample(self, proposals: List[Dict[str, Any]], targets: List[Dict[str, Any]]):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        """
        labels, regression_targets, matched_idxs, attributes = self.prepare_targets(proposals, targets)

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)

        if attributes != []:
            for labels_per_image, attributes_per_image, regression_targets_per_image, matched_idxs_per_image, proposals_per_image in zip(
                labels, attributes, regression_targets, matched_idxs, proposals
            ):
                proposals_per_image["labels"] = labels_per_image
                proposals_per_image["regression_targets"] = regression_targets_per_image
                proposals_per_image["matched_idxs"] = matched_idxs_per_image
                proposals_per_image["attributes"] = attributes_per_image
        else:
            for labels_per_image, regression_targets_per_image, matched_idxs_per_image, proposals_per_image in zip(
                labels, regression_targets, matched_idxs, proposals
            ):
                proposals_per_image["labels"] = labels_per_image
                proposals_per_image["regression_targets"] = regression_targets_per_image
                proposals_per_image["matched_idxs"] = matched_idxs_per_image

        # distributed sampled proposals
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals[img_idx] = filter_instances(proposals[img_idx], img_sampled_inds)

        return proposals

    def assign_label_to_proposals(self, proposals: List[Dict[str, Any]], targets: List[Dict[str, Any]]):
        for img_idx, (target, proposal) in enumerate(zip(targets, proposals)):
            match_quality_matrix = box_iou(target["boxes"], proposal["boxes"], target["mode"], proposal["mode"])
            matched_idxs = self.proposal_matcher(match_quality_matrix)
            
            matched_labels = target["labels"][matched_idxs.clamp(min=0)].clone().to(dtype=torch.int64)
            matched_labels[matched_idxs < 0] = 0
            proposals[img_idx]["labels"] = matched_labels

            if self.use_attributes and "attributes" in target:
                attris_per_image = target["attributes"][matched_idxs.clamp(min=0)].clone().to(dtype=torch.int64)
                attris_per_image[matched_idxs < 0, :] = 0
                proposals[img_idx]["attributes"] = attris_per_image
            
        return proposals


def make_roi_box_samp_processor(cfg):
    matcher = Matcher(
        cfg.model.roi_heads.fg_iou_threshold,
        cfg.model.roi_heads.bg_iou_threshold,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.model.roi_heads.bbox_reg_weights
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.model.roi_heads.batch_size_per_image, cfg.model.roi_heads.positive_fraction
    )

    samp_processor = FastRCNNSampling(
        matcher,
        fg_bg_sampler,
        box_coder,
        cfg.model.attribute_on, # use_attris
    )

    return samp_processor
