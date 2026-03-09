# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import scipy.linalg
import numpy as np

from sgg_benchmark.layers import nms as _box_nms

def box_union(boxes1, boxes2):
    """
    Computes the union region of two sets of boxes.
    Boxes should be [N, 4] and in 'xyxy' mode.
    """
    return torch.cat((
        torch.min(boxes1[:, :2], boxes2[:, :2]),
        torch.max(boxes1[:, 2:], boxes2[:, 2:])
    ), dim=1)

def box_intersection(boxes1, boxes2):
    """
    Computes the intersection region of two sets of boxes.
    Boxes should be [N, 4] and in 'xyxy' mode.
    """
    inter_box = torch.cat((
        torch.max(boxes1[:, :2], boxes2[:, :2]),
        torch.min(boxes1[:, 2:], boxes2[:, 2:])
    ), dim=1)
    invalid_bbox = (inter_box[:, 0] >= inter_box[:, 2]) | (inter_box[:, 1] >= inter_box[:, 3])
    inter_box[invalid_bbox] = 0
    return inter_box

def cat_instances(instances_list):
    """
    Concatenates a list of instance dictionaries.
    """
    if not instances_list:
        return {}
    if len(instances_list) == 1:
        return instances_list[0]

    res = {
        "image_size": instances_list[0]["image_size"],
        "mode": instances_list[0]["mode"],
    }
    
    keys = instances_list[0].keys()
    for k in keys:
        if k in ["image_size", "mode"]:
            continue
        
        vals = [inst[k] for inst in instances_list]
        if isinstance(vals[0], torch.Tensor):
            if k == "boxes":
                res[k] = torch.cat(vals, dim=0)
            elif k in ("relation", "relation_map"):
                # Relation matrices are square [N, N] — combine with block_diag
                # so that indices remain valid within each image's sub-block.
                triplet_list = [v.detach().cpu().numpy() for v in vals]
                data = torch.from_numpy(scipy.linalg.block_diag(*triplet_list)).to(vals[0].device)
                res[k] = data
            else:
                res[k] = torch.cat(vals, dim=0)
        else:
            # For non-tensor fields, just take the first or handle specially
            res[k] = vals[0]
            
    return res

# transpose constants
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

def box_nms(boxes, scores=None, nms_thresh=0.5, max_proposals=-1, score_field="pred_scores"):
    """
    Performs non-maximum suppression on boxes.
    Boxes can be a Tensor or a dictionary (instance).
    """
    if isinstance(boxes, dict):
        res = boxes.copy()
        if scores is None:
            scores = boxes.get(score_field)
            if scores is None:
                # fall back to 'scores' if 'pred_scores' not found
                scores = boxes.get("scores")
        
        keep = box_nms(boxes["boxes"], scores, nms_thresh, max_proposals)
        return filter_instances(res, keep), keep

    if nms_thresh <= 0:
        return torch.arange(len(boxes), device=boxes.device)
    
    keep = _box_nms(boxes, scores, nms_thresh)
    if max_proposals > 0:
        keep = keep[:max_proposals]
    return keep

def box_iou(boxes1, boxes2):
    """
    Computes IoU between two sets of boxes.
    Both should be in 'xyxy' mode.
    """
    area1 = box_area(boxes1, mode="xyxy")
    area2 = box_area(boxes2, mode="xyxy")

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt + 1).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou

def box_convert(boxes, in_mode, out_mode):
    """
    Converts boxes from in_mode to out_mode.
    Supported modes: 'xyxy', 'xywh'
    """
    if in_mode == out_mode:
        return boxes
    
    if in_mode == "xyxy" and out_mode == "xywh":
        TO_REMOVE = 1
        xmin, ymin, xmax, ymax = boxes.split(1, dim=-1)
        return torch.cat(
            (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1
        )
    elif in_mode == "xywh" and out_mode == "xyxy":
        TO_REMOVE = 1
        xmin, ymin, w, h = boxes.split(1, dim=-1)
        return torch.cat(
            (
                xmin,
                ymin,
                xmin + (w - TO_REMOVE).clamp(min=0),
                ymin + (h - TO_REMOVE).clamp(min=0),
            ),
            dim=-1,
        )
    else:
        raise ValueError("Unsupported conversion: {} to {}".format(in_mode, out_mode))

def box_resize(boxes, old_size, new_size, mode="xyxy"):
    """
    Resizes boxes from old_size to new_size.
    old_size, new_size: (width, height)
    Boxes can be a Tensor or a dictionary (proposal).
    """
    if isinstance(boxes, dict):
        res = boxes.copy()
        res["boxes"] = box_resize(boxes["boxes"], old_size, new_size, mode=boxes.get("mode", mode))
        res["image_size"] = new_size
        return res

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, old_size))
    ratio_width, ratio_height = ratios
    
    if ratio_width == ratio_height:
        return boxes * ratio_width
    
    if mode == "xywh":
        boxes = box_convert(boxes, "xywh", "xyxy")
        
    xmin, ymin, xmax, ymax = boxes.split(1, dim=-1)
    scaled_xmin = xmin * ratio_width
    scaled_xmax = xmax * ratio_width
    scaled_ymin = ymin * ratio_height
    scaled_ymax = ymax * ratio_height
    scaled_box = torch.cat(
        (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1
    )
    
    if mode == "xywh":
        scaled_box = box_convert(scaled_box, "xyxy", "xywh")
        
    return scaled_box

def box_transpose(boxes, image_size, method, mode="xyxy"):
    """
    Transpose bounding box (flip or rotate in 90 degree steps)
    """
    if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
        raise NotImplementedError(
            "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
        )

    image_width, image_height = image_size
    
    is_xywh = mode == "xywh"
    if is_xywh:
        boxes = box_convert(boxes, "xywh", "xyxy")
        
    xmin, ymin, xmax, ymax = boxes.split(1, dim=-1)
    if method == FLIP_LEFT_RIGHT:
        TO_REMOVE = 1
        transposed_xmin = image_width - xmax - TO_REMOVE
        transposed_xmax = image_width - xmin - TO_REMOVE
        transposed_ymin = ymin
        transposed_ymax = ymax
    elif method == FLIP_TOP_BOTTOM:
        transposed_xmin = xmin
        transposed_xmax = xmax
        transposed_ymin = image_height - ymax
        transposed_ymax = image_height - ymin

    transposed_boxes = torch.cat(
        (transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1
    )
    
    if is_xywh:
        transposed_boxes = box_convert(transposed_boxes, "xyxy", "xywh")
        
    return transposed_boxes

def box_clip(boxes, image_size, mode="xyxy"):
    """
    Clips boxes to image boundaries.
    image_size: (width, height)
    """
    if mode == "xywh":
        boxes = box_convert(boxes, "xywh", "xyxy")
        
    boxes = boxes.clone()
    boxes[:, 0] = boxes[:, 0].clamp(min=0, max=image_size[0])
    boxes[:, 1] = boxes[:, 1].clamp(min=0, max=image_size[1])
    boxes[:, 2] = boxes[:, 2].clamp(min=0, max=image_size[0])
    boxes[:, 3] = boxes[:, 3].clamp(min=0, max=image_size[1])
    
    if mode == "xywh":
        boxes = box_convert(boxes, "xyxy", "xywh")
    return boxes

def box_remove_empty(boxes, mode="xyxy"):
    """
    Returns a mask of boxes that are not empty.
    """
    if mode == "xywh":
        # boxes[:, 2] and boxes[:, 3] are width and height
        return (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
    else:
        # xyxy
        return (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])

def box_area(boxes, mode="xyxy"):
    """
    Computes the area of a set of boxes.
    """
    if mode == "xyxy":
        return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    elif mode == "xywh":
        return boxes[:, 2] * boxes[:, 3]
    else:
        raise ValueError("Unsupported mode: {}".format(mode))
def squeeze_tensor(tensor):
    tensor = torch.squeeze(tensor)
    try:
        len(tensor)
    except TypeError:
        tensor.unsqueeze_(0)
    return tensor
def remove_small_boxes(boxes, min_size, mode="xyxy"):
    """
    Only keep boxes with both sides >= min_size
    """
    if mode == "xyxy":
        TO_REMOVE = 1
        ws = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        hs = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
    elif mode == "xywh":
        ws = boxes[:, 2]
        hs = boxes[:, 3]
    else:
        raise ValueError("Unsupported mode: {}".format(mode))
    
    keep = (ws >= min_size) & (hs >= min_size)
    return keep.nonzero().squeeze(1)

def filter_instances(instances, keep):
    """
    Filters fields in an instance dictionary based on a boolean mask or index tensor.
    Handles [N] fields and [N, N] triplet fields.
    """
    new_instances = instances.copy()
    new_instances["boxes"] = instances["boxes"][keep]
    
    # We need to know which fields are triplet fields.
    # We can infer it from the shape or store a list.
    # For now, let's check shapes.
    N = instances["boxes"].shape[0]
    
    for k, v in instances.items():
        if k in ["boxes", "image_size", "mode"]:
            continue
        if isinstance(v, torch.Tensor):
            if v.dim() == 1 and v.shape[0] == N:
                new_instances[k] = v[keep]
            elif k == "relation" and v.dim() == 2 and v.shape[0] == N and v.shape[1] == N:
                # Square N×N relation matrix — must use bilateral indexing
                new_instances[k] = v[keep][:, keep]
            elif v.dim() > 0 and v.shape[0] == N:
                # All other per-instance fields (lb_boxes, pred_scores, features…)
                # — index only along dim 0, regardless of shape[1].
                # NOTE: do NOT use the shape-heuristic square check here: when N==4
                # a (N,4) lb_boxes tensor would spuriously match (N,N) and get
                # doubly-indexed, producing a wrong (N_rel, N_rel) tensor.
                new_instances[k] = v[keep]
                
    return new_instances

def split_instances(instances, split_indices):
    """
    Splits an instances dictionary into multiple dictionaries based on split indices.
    split_indices: list/tuple of end indices for each split.
    Example: split_indices=(10, 25) for N=25 -> [0:10], [10:25]
    """
    res = []
    start = 0
    for end in split_indices:
        # Create a mask or use slicing
        indices = torch.arange(start, end, device=instances["boxes"].device)
        res.append(filter_instances(instances, indices))
        start = end
    return res
