"""
Utilities to infer number of classes from dataset files.

This module provides functions to automatically determine num_classes and num_predicates
from COCO-format dataset JSON files, eliminating the need to manually specify these
parameters in configuration files.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


def infer_num_classes_from_dataset(
    annotation_file: str,
) -> Tuple[int, int]:
    """
    Infer the number of object classes and predicates from a COCO-format dataset file.
    
    Args:
        annotation_file: Path to COCO-format JSON annotation file
        
    Returns:
        Tuple of (num_object_classes, num_predicate_classes)
        Includes background class, so min values are 2 and 2
    """
    annotation_file = Path(annotation_file)
    
    if not annotation_file.exists():
        logger.warning(f"Annotation file not found: {annotation_file}")
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    
    try:
        with open(annotation_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file {annotation_file}: {e}")
        raise ValueError(f"Invalid JSON file: {annotation_file}") from e
    
    # Extract number of object classes
    # COCO format: categories list with 'id' and 'name' fields
    categories = data.get('categories', [])
    if not categories:
        logger.warning(f"No categories found in {annotation_file}, assuming 151 object classes (VG default)")
        num_obj_classes = 151
    else:
        # Get max category id and add 1 (for 0-based indexing)
        max_cat_id = max(cat.get('id', 0) for cat in categories)
        num_obj_classes = max_cat_id + 1
    
    # Add 1 for background class if not already included (common in COCO format)
    # Background is typically at index 0
    if categories and min(cat.get('id', 0) for cat in categories) > 0:
        num_obj_classes += 1
    
    # Extract number of predicate classes
    # Custom field in SGG: rel_categories list
    rel_categories = data.get('rel_categories', [])
    if not rel_categories:
        logger.warning(f"No rel_categories found in {annotation_file}, assuming 51 predicate classes (VG default)")
        num_rel_classes = 51
    else:
        # rel_categories is typically a list of relationship names
        # Add 1 for background predicate (typically at index 0)
        num_rel_classes = len(rel_categories) + 1  # +1 for background
    
    logger.info(f"Inferred from {annotation_file.name}:")
    logger.info(f"  - Object classes: {num_obj_classes}")
    logger.info(f"  - Predicate classes: {num_rel_classes}")
    
    return num_obj_classes, num_rel_classes


def infer_num_classes_from_config(cfg: Any) -> Tuple[Optional[int], Optional[int]]:
    """
    Infer num_classes and num_predicate_classes from configuration.
    
    This function attempts to:
    1. Use explicit config values if provided (and >= 0)
    2. Infer from dataset annotation files if available
    3. Return None for values that cannot be inferred (will use config defaults)
    
    Args:
        cfg: Configuration object (OmegaConf DictConfig)
        
    Returns:
        Tuple of (num_obj_classes, num_rel_classes), where either may be None
    """
    num_obj_classes = None
    num_rel_classes = None
    
    # Try to get annotation files from config
    try:
        # Look for dataset configuration
        datasets_cfg = cfg.get('datasets', {})
        
        # Try to find annotation file for training data
        ann_file = None
        if hasattr(datasets_cfg, 'catalog'):
            catalog = datasets_cfg.catalog
            for dataset_name in ['custom_dataset', 'vg150', 'psg', 'gqa', 'indoorvg']:
                if hasattr(catalog, dataset_name):
                    dataset_cfg = catalog[dataset_name]
                    if hasattr(dataset_cfg, 'ann_file'):
                        ann_file = dataset_cfg.ann_file
                        break
        
        if ann_file:
            try:
                num_obj_classes, num_rel_classes = infer_num_classes_from_dataset(ann_file)
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Could not infer classes from dataset file: {e}")
                num_obj_classes = None
                num_rel_classes = None
    
    except (AttributeError, TypeError) as e:
        logger.warning(f"Could not access dataset configuration: {e}")
    
    return num_obj_classes, num_rel_classes


def resolve_num_classes(cfg: Any) -> Dict[str, Dict[str, int]]:
    """
    Resolve final num_classes values using config values and inference.
    
    Returns a dict with resolved values that can be applied to config using OmegaConf.
    
    num_classes signal values:
    - 0 or not set: Auto-infer from dataset (default)
    - > 0: Use explicit value (skip inference)
    
    Args:
        cfg: Configuration object
        
    Returns:
        Dictionary with structure:
        {
            'model': {
                'roi_box_head': {'num_classes': int},
                'roi_relation_head': {'num_classes': int}
            }
        }
    """
    result = {'model': {'roi_box_head': {}, 'roi_relation_head': {}}}
    
    try:
        inferred_obj, inferred_rel = infer_num_classes_from_config(cfg)
        
        # Get current config values
        current_obj_classes = getattr(cfg.model.roi_box_head, 'num_classes', 0)
        current_rel_classes = getattr(cfg.model.roi_relation_head, 'num_classes', 0)
        
        # Use inferred values if current values are <= 0 (signal for auto-inference)
        if current_obj_classes <= 0 and inferred_obj:
            result['model']['roi_box_head']['num_classes'] = inferred_obj
            logger.info(f"Using inferred num_classes for roi_box_head: {inferred_obj}")
        else:
            result['model']['roi_box_head']['num_classes'] = current_obj_classes or inferred_obj or 151
        
        if current_rel_classes <= 0 and inferred_rel:
            result['model']['roi_relation_head']['num_classes'] = inferred_rel
            logger.info(f"Using inferred num_classes for roi_relation_head: {inferred_rel}")
        else:
            result['model']['roi_relation_head']['num_classes'] = current_rel_classes or inferred_rel or 51
    
    except Exception as e:
        logger.warning(f"Failed to resolve num_classes, using defaults: {e}")
        result['model']['roi_box_head']['num_classes'] = getattr(cfg.model.roi_box_head, 'num_classes', 0) or 151
        result['model']['roi_relation_head']['num_classes'] = getattr(cfg.model.roi_relation_head, 'num_classes', 0) or 51
    
    return result
