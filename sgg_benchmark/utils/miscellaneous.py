# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import errno
import json
import logging
import os
from typing import List, Any
from .comm import is_main_process
import numpy as np
import torch

from sgg_benchmark.structures.box_ops import box_iou
import sgg_benchmark

# Hydra/OmegaConf imports (for new config system)
try:
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = Any  # Type hint fallback


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_path() -> str:
    """Get the root path of the project."""
    return os.path.dirname(sgg_benchmark.__file__).split('sgg_benchmark')[0]

def mkdir(path: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_labels(dataset_list: List, output_dir: str) -> None:
    """
    Save dataset label mappings to JSON file.
    
    Args:
        dataset_list: List of datasets with categories
        output_dir: Directory to save labels.json
    """
    if is_main_process():
        try:
            from loguru import logger
        except ImportError:
            logger = logging.getLogger(__name__)

        ids_to_labels = {}
        for dataset in dataset_list:
            if hasattr(dataset, 'categories'):
                ids_to_labels.update(dataset.categories)

        if ids_to_labels:
            labels_file = os.path.join(output_dir, 'labels.json')
            logger.info("Saving labels mapping into {}".format(labels_file))
            with open(labels_file, 'w') as f:
                json.dump(ids_to_labels, f, indent=2)

def save_config(cfg: DictConfig, path: str) -> None:
    """
    Save configuration to YAML file.
    
    Supports both YACS (old) and Hydra/OmegaConf (new) configs.
    
    Args:
        cfg: Configuration to save (YACS CfgNode or OmegaConf DictConfig)
        path: File path to save config
    """
    if not is_main_process():
        return
    
    with open(path, 'w') as f:
        # Check if it's a Hydra/OmegaConf config
        if HYDRA_AVAILABLE and isinstance(cfg, DictConfig):
            # New way: OmegaConf YAML
            OmegaConf.save(cfg, f)
        elif hasattr(cfg, 'dump'):
            # Old way: YACS dump()
            f.write(cfg.dump())
        else:
            # Fallback: try to convert to YAML
            import yaml
            yaml.dump(dict(cfg), f, default_flow_style=False)


def intersect_2d(x1, x2):
    """
    Given two arrays [m1, n], [m2,n], returns a [m1, m2] array where each entry is True if those
    rows match.
    :param x1: [m1, n] numpy array
    :param x2: [m2, n] numpy array
    :return: [m1, m2] bool array of the intersections
    """
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Input arrays must have same #columns")

    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    res = (x1[..., None] == x2.T[None, ...]).all(1)
    return res

def argsort_desc(scores):
    """
    Returns the indices that sort scores descending in a smart way
    :param scores: Numpy array of arbitrary size
    :return: an array of size [numel(scores), dim(scores)] where each row is the index you'd
             need to get the score.
    """
    return np.column_stack(np.unravel_index(np.argsort(-scores.ravel()), scores.shape))

def bbox_overlaps(boxes1, boxes2):
    """
    Parameters:
        boxes1 (m, 4) [List or np.array] : bounding boxes of (x1,y1,x2,y2)
        boxes2 (n, 4) [List or np.array] : bounding boxes of (x1,y1,x2,y2)
    Return:
        iou (m, n) [np.array]
    """
    boxes1 = torch.as_tensor(boxes1, dtype=torch.float32)
    boxes2 = torch.as_tensor(boxes2, dtype=torch.float32)
    iou = box_iou(boxes1, boxes2).cpu().numpy()
    return iou
