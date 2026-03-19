import torch
import torch.nn.functional as F
from torch import nn
from typing import List, Dict, Any, Tuple, Optional

from sgg_benchmark.structures.box_ops import box_clip, box_nms, cat_instances
from sgg_benchmark.modeling.box_coder import BoxCoder

class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        post_nms_per_cls_topn=300,
        nms_filter_duplicates=True,
        detections_per_img=100,
        test_detections_per_img=100,
        box_coder=None,
        cls_agnostic_bbox_reg=False,
        bbox_aug_enabled=False,
        save_proposals=False,
        custum_eval=False
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            test_detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.post_nms_per_cls_topn = post_nms_per_cls_topn
        self.nms_filter_duplicates = nms_filter_duplicates
        self.detections_per_img = detections_per_img
        self.test_detections_per_img = test_detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.bbox_aug_enabled = bbox_aug_enabled
        self.save_proposals = save_proposals
        self.custum_eval = custum_eval

    def forward(self, x, boxes: List[Dict[str, Any]], relation_mode=False):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[Dict]): bounding boxes that are used as
                reference, one for each image

        Returns:
            results (list[Dict]): one Dict for each image, containing
                the extra fields labels and scores
        """
        features, class_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)

        # TODO think about a representation of batch of boxes
        image_shapes = [box["image_size"] for box in boxes]
        boxes_per_image = [len(box["boxes"]) for box in boxes]
        concat_boxes = torch.cat([a["boxes"] for a in boxes], dim=0)

        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -4:]
        # add rpn regression offset to the original proposals
        proposals = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        ) # tensor of size (num_box, 4*num_cls)
        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1]

        features = features.split(boxes_per_image, dim=0)
        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        nms_features = []
        for i, (prob, boxes_per_img, image_shape) in enumerate(zip(
            class_prob, proposals, image_shapes
        )):
            instance = self.prepare_instance(boxes_per_img, prob, image_shape, boxes[i]["mode"])
            instance = box_clip(instance)
            assert self.bbox_aug_enabled == False
            if not self.bbox_aug_enabled:  # If bbox aug is enabled, we will do it later
                instance, orig_inds, boxes_per_cls = self.filter_results(instance, num_classes)
            # add 
            instance = self.add_important_fields(i, boxes, orig_inds, instance, boxes_per_cls, relation_mode)
            
            results.append(instance)
            nms_features.append(features[i][orig_inds])
        
        nms_features = torch.cat(nms_features, dim=0)
        return nms_features, results

    def add_important_fields(self, i, boxes, orig_inds, instance, boxes_per_cls, relation_mode=False):
        if relation_mode:
            if not self.custum_eval:
                gt_labels = boxes[i]['labels'][orig_inds]
                instance['labels'] = gt_labels

            predict_logits = boxes[i]['predict_logits'][orig_inds]
            instance['boxes_per_cls'] = boxes_per_cls
            instance['predict_logits'] = predict_logits

        return instance

    def prepare_instance(self, boxes, scores, image_shape, mode):
        """
        Returns instance Dict from `boxes` and adds probability scores information
        as an extra field
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        instance = {
            "boxes": boxes,
            "image_size": image_shape,
            "mode": mode,
            "pred_scores": scores
        }
        return instance

    def filter_results(self, instance, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        boxes = instance["boxes"].reshape(-1, num_classes * 4)
        boxes_per_cls = instance["boxes"].reshape(-1, num_classes, 4)
        scores = instance["pred_scores"].reshape(-1, num_classes)

        device = scores.device
        result = []
        orig_inds = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            
            instance_j = {
                "boxes": boxes_j,
                "image_size": instance["image_size"],
                "mode": instance["mode"],
                "pred_scores": scores_j
            }
            
            instance_j, keep = box_nms(
                instance_j, self.nms, max_proposals=self.post_nms_per_cls_topn, score_field='pred_scores'
            )
            inds = inds[keep]
            num_labels = len(instance_j["boxes"])
            instance_j["pred_labels"] = torch.full((num_labels,), j, dtype=torch.int64, device=device)
            result.append(instance_j)
            orig_inds.append(inds)

        if self.nms_filter_duplicates or self.save_proposals:
            assert len(orig_inds) == (num_classes - 1)
            # set all bg to zero
            inds_all[:, 0] = 0 
            for j in range(1, num_classes):
                inds_all[:, j] = 0
                orig_idx = orig_inds[j-1]
                inds_all[orig_idx, j] = 1
            dist_scores = scores * inds_all.float()
            scores_pre, labels_pre = dist_scores.max(1)
            final_inds = scores_pre.nonzero()
            assert final_inds.dim() != 0
            final_inds = final_inds.squeeze(1)

            scores_pre = scores_pre[final_inds]
            labels_pre = labels_pre[final_inds]

            result = {
                "boxes": boxes_per_cls[final_inds, labels_pre],
                "image_size": instance["image_size"],
                "mode": instance["mode"],
                "pred_scores": scores_pre,
                "pred_labels": labels_pre
            }
            orig_inds = final_inds
        else:
            result = cat_instances(result)
            orig_inds = torch.cat(orig_inds, dim=0)
        
        number_of_detections = len(result["boxes"])
        # Limit to max_per_image detections **over all classes**
        # During evaluation, use test_detections_per_img for better precision/latency trade-off
        limit = self.detections_per_img if self.training else self.test_detections_per_img
        if number_of_detections > limit > 0:
            cls_scores = result["pred_scores"]
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - limit + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            
            # Simple manual filter for the result dict
            result["boxes"] = result["boxes"][keep]
            result["pred_scores"] = result["pred_scores"][keep]
            result["pred_labels"] = result["pred_labels"][keep]
            orig_inds = orig_inds[keep]
            
        return result, orig_inds, boxes_per_cls[orig_inds]


def make_roi_box_post_processor(cfg):
    bbox_reg_weights = cfg.model.roi_heads.bbox_reg_weights
    box_coder = BoxCoder(weights=bbox_reg_weights)

    score_thresh = cfg.model.roi_heads.score_thresh
    nms_thresh = cfg.model.roi_heads.nms
    detections_per_img = cfg.model.roi_heads.detections_per_img
    test_detections_per_img = cfg.test.detections_per_img
    cls_agnostic_bbox_reg = cfg.model.cls_agnostic_bbox_reg
    bbox_aug_enabled = cfg.test.bbox_aug.enabled
    post_nms_per_cls_topn = cfg.model.roi_heads.post_nms_per_cls_topn
    nms_filter_duplicates = cfg.model.roi_heads.nms_filter_duplicates
    save_proposals = cfg.test.save_proposals
    custum_eval = cfg.test.custum_eval

    postprocessor = PostProcessor(
        score_thresh,
        nms_thresh,
        post_nms_per_cls_topn,
        nms_filter_duplicates,
        detections_per_img,
        test_detections_per_img,
        box_coder,
        cls_agnostic_bbox_reg,
        bbox_aug_enabled,
        save_proposals,
        custum_eval
    )
    return postprocessor
