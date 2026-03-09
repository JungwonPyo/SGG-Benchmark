# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import importlib.util
import os
from pathlib import Path
import bisect
import copy
import logging

import torch
import torch.utils.data
from sgg_benchmark.utils.comm import get_world_size
from sgg_benchmark.utils.miscellaneous import save_labels

from . import datasets as D
from . import samplers

from .collate_batch import BatchCollator, BBoxAugCollator
from .transforms import build_transforms

from .datasets.data import RelationDataset


# ---------------------------------------------------------------------------
# Default data directory mapping  (name, type) → relative path from project root
# ---------------------------------------------------------------------------
_DEFAULT_DATA_DIRS: dict[tuple[str, str], str] = {
    ("VG150",    "h5"):   "datasets/VG150",
    ("VG150",    "coco"): "datasets/VG150/VG150_coco_format",
    ("IndoorVG", "coco"): "datasets/IndoorVG/IndoorVG_coco_format",
    ("PSG",      "coco"): "datasets/PSG/coco_format",
}

# Project root — two levels above sgg_benchmark/data/build.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _get_default_data_dir(ds_name: str, ds_type: str) -> str:
    key = (ds_name, ds_type)
    rel = _DEFAULT_DATA_DIRS.get(key, "")
    return str(_PROJECT_ROOT / rel) if rel else ""


def _resolve_path(p: str) -> str:
    """Return an absolute path, resolving relative paths against the project root."""
    if not p:
        return p
    if os.path.isabs(p):
        return p
    return str(_PROJECT_ROOT / p)


def _auto_download(ds_name: str, output_dir: str) -> None:
    """Download *ds_name* from HuggingFace Hub using tools/download_from_hub.py."""
    try:
        logger = logging.getLogger(__name__)
    except Exception:
        logger = logging.getLogger("sgg_benchmark")

    hub_script = _PROJECT_ROOT / "tools" / "download_from_hub.py"
    if not hub_script.exists():
        raise RuntimeError(
            f"Dataset '{ds_name}' not found at '{output_dir}' and the download "
            f"script was not found at '{hub_script}'.\n"
            f"Please run:  python tools/download_from_hub.py --dataset {ds_name}"
        )

    logger.info(f"Auto-downloading '{ds_name}' from HuggingFace Hub → {output_dir} …")
    spec = importlib.util.spec_from_file_location("download_from_hub", hub_script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.download_dataset(ds_name, output_dir=Path(output_dir), save_images=False)


def _cfg_to_dict(cfg):
    """Convert OmegaConf / Config wrapper to a plain Python dict."""
    try:
        from omegaconf import OmegaConf, DictConfig
        from sgg_benchmark.config import Config
        if isinstance(cfg, Config):
            cfg = cfg._cfg
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass
    return cfg


def get_dataset_info(name_or_split: str, cfg):
    """
    Build a ``{factory, args}`` dict for one dataset split.

    The *name_or_split* argument is used only to extract the split token
    (``train`` / ``val`` / ``test``) from its suffix (e.g. ``"VG150_train"``
    → split ``"train"``).  The actual dataset identity comes from
    ``cfg.datasets.{name, type, data_dir}``.

    Supported types
    ---------------
    ``type: h5``   – legacy VG150 format with ``.h5`` annotation files
                     → ``VGDataset`` factory
    ``type: coco`` – COCO-format with ``_annotations.coco.json`` files
                     → ``RelationDataset`` factory
                     If *data_dir/split* is missing, the dataset is
                     auto-downloaded from HuggingFace Hub.
    """
    cfg = _cfg_to_dict(cfg)

    # ---- extract split ----
    p = name_or_split.rfind("_")
    split = name_or_split[p + 1:] if (p >= 0 and name_or_split[p + 1:] in {"train", "val", "test"}) else "train"

    # ---- read simplified datasets config ----
    datasets_cfg = cfg.get("datasets", {})
    ds_name     = datasets_cfg.get("name", "")
    ds_type     = datasets_cfg.get("type", "coco")
    ds_data_dir = _resolve_path(datasets_cfg.get("data_dir", "") or _get_default_data_dir(ds_name, ds_type))

    model_cfg            = cfg.get("model", {})
    roi_relation_head_cfg = model_cfg.get("roi_relation_head", {})
    flip_aug             = model_cfg.get("flip_aug", False)

    # ---- route to factory ----
    if ds_type == "h5":
        return _get_h5_dataset_info(ds_name, ds_data_dir, split, model_cfg, roi_relation_head_cfg, flip_aug)
    else:
        return _get_coco_dataset_info(ds_name, ds_data_dir, split, flip_aug)


def _get_h5_dataset_info(ds_name, data_dir, split, model_cfg, roi_head_cfg, flip_aug):
    """Build VGDataset args for a legacy h5-format dataset."""
    data_dir_path = Path(data_dir)
    datasets_root = data_dir_path.parent  # e.g.  datasets/

    KNOWN_H5 = {
        "VG150": {
            "roidb_file":      "VG-SGG-with-attri.h5",
            "dict_file":       "VG-SGG-dicts-with-attri.json",
            "image_file":      str(datasets_root / "vg" / "image_data.json"),
            "zeroshot_file":   "zeroshot_triplet.pytorch",
            "img_dir":         str(datasets_root / "VG_100K"),
        },
    }

    if ds_name not in KNOWN_H5:
        raise RuntimeError(
            f"h5 type is only supported for: {list(KNOWN_H5)}. "
            f"Got '{ds_name}'. Use type=coco instead."
        )

    h5 = KNOWN_H5[ds_name]
    args = {
        "img_dir":          h5["img_dir"],
        "roidb_file":       str(data_dir_path / h5["roidb_file"]),
        "dict_file":        str(data_dir_path / h5["dict_file"]),
        "image_file":       h5["image_file"],
        "zeroshot_file":    str(data_dir_path / h5["zeroshot_file"]),
        "informative_file": "",
        "capgraphs_file":   "",
        "ann_file":         "",
        "split":            split,
        "filter_empty_rels": True,
        "filter_duplicate_rels": True,
        "filter_non_overlap": (
            not roi_head_cfg.get("use_gt_box", True)
            and model_cfg.get("relation_on", False)
            and roi_head_cfg.get("require_box_overlap", True)
        ),
        "flip_aug": flip_aug,
    }
    return dict(factory="VGDataset", args=args)


def _get_coco_dataset_info(ds_name, data_dir, split, flip_aug):
    """Build RelationDataset args for a COCO-format dataset, auto-downloading if needed."""
    split_dir = os.path.join(data_dir, split)
    ann_file  = os.path.join(split_dir, "_annotations.coco.json")

    # Auto-download from HuggingFace if the annotation file is missing
    if not os.path.exists(ann_file):
        if ds_name in ("PSG", "VG150", "IndoorVG"):
            _auto_download(ds_name, data_dir)
        else:
            raise RuntimeError(
                f"Annotation file not found: {ann_file}\n"
                f"Dataset '{ds_name}' does not support auto-download. "
                f"Please place the data manually."
            )

    args = {
        "annotation_file":      ann_file,
        "img_dir":              split_dir,
        "filter_empty_rels":    True,
        "filter_duplicate_rels": True,
        "filter_non_overlap":   False,
        "flip_aug":             flip_aug,
    }
    return dict(factory="RelationDataset", args=args)
    


# by Jiaxin
def get_dataset_statistics(cfg, save=True):
    """
    Get dataset statistics (e.g., frequency bias) from training data.
    Called to help construct the FrequencyBias module.
    """
    try:
        from loguru import logger
    except ImportError:
        logger = logging.getLogger(__name__)

    cfg = _cfg_to_dict(cfg)

    logger.info('-' * 100)
    logger.info('get dataset statistics...')

    datasets_cfg = cfg.get('datasets', {})
    ds_name = datasets_cfg.get('name', '')

    # Cache file: stored in output_dir, keyed by dataset name
    data_statistics_name = f"{ds_name}_statistics" if ds_name else "dataset_statistics"
    save_file = os.path.join(cfg.get('output_dir', '.'), f"{data_statistics_name}.cache")

    if os.path.exists(save_file):
        logger.info('Loading data statistics from: ' + str(save_file))
        logger.info('-' * 100)
        return torch.load(save_file, map_location=torch.device("cpu"), weights_only=False)
    else:
        logger.info('Unable to load data statistics from: ' + str(save_file))

    # Build training dataset and extract statistics
    train_name = f"{ds_name}_train" if ds_name else "dataset_train"
    data = get_dataset_info(train_name, cfg)
    factory = getattr(D, data["factory"])
    dataset = factory(**data["args"])
    statistics = dataset.get_statistics()

    result = {
        'fg_matrix': statistics['fg_matrix'],
        'pred_dist': statistics['pred_dist'],
        'obj_classes': statistics['obj_classes'], # must be exactly same for multiple datasets
        'rel_classes': statistics['rel_classes'],
        'predicate_new_order': statistics['predicate_new_order'], # for GCL
        'predicate_new_order_count': statistics['predicate_new_order_count'],
        'pred_freq': statistics['pred_freq'],
        'triplet_freq': statistics['triplet_freq'],
        'pred_weight': statistics['pred_weight'],
    }

    if save:
        logger.info('Save data statistics to: ' + str(save_file))
        logger.info('-'*100)
        torch.save(result, save_file)
    return result


def build_dataset(cfg, dataset_list, transforms, is_train=True):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_train, coco_2014_val, etc
        transforms (callable): transforms to apply to each (image, target) sample
        is_train (bool): whether to setup the dataset for training or testing
    """
    # Convert OmegaConf ListConfig to plain list if needed
    try:
        from omegaconf import ListConfig
        if isinstance(dataset_list, ListConfig):
            dataset_list = list(dataset_list)
    except ImportError:
        pass
    
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {} of type {}".format(dataset_list, type(dataset_list))
        )
    datasets = []
    for dataset_name in dataset_list:
        # Use new dataset factory
        data = get_dataset_info(dataset_name, cfg)
        factory = getattr(D, data["factory"])
        args = data["args"]
        args["transforms"] = transforms

        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset]


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
    dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_loader(cfg, mode='train', is_distributed=False, start_iter=0, dataset_to_test=None, num_iters=None):
    assert mode in {'train', 'val', 'test'}
    if dataset_to_test == "":
        dataset_to_test = None
    assert dataset_to_test in {'train', 'val', 'test', None}

    # this variable enable to run a test on any data split, even on the training dataset
    # without actually flagging it for training....
    if dataset_to_test is None:
        dataset_to_test = mode

    num_gpus = get_world_size()
    is_train = mode == 'train'
    
    solver_cfg = cfg.get('solver', {})
    test_cfg = cfg.get('test', {})
    dataloader_cfg = cfg.get('dataloader', {})
    datasets_cfg = cfg.get('datasets', {})
    model_cfg = cfg.get('model', {})
    
    if is_train:
        images_per_batch = solver_cfg.get('ims_per_batch', 1)
        assert (
            images_per_batch % num_gpus == 0
        ), "solver.ims_per_batch ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
    else:
        images_per_batch = test_cfg.get('ims_per_batch', 1)
        assert (
            images_per_batch % num_gpus == 0
        ), "test.ims_per_batch ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if dataloader_cfg.get('aspect_ratio_grouping', False) else []

    # Get dataset list based on mode
    if dataset_to_test == 'train':
        dataset_list = datasets_cfg.get('train', [])
    elif dataset_to_test == 'val':
        dataset_list = datasets_cfg.get('val', [])
    else:
        dataset_list = datasets_cfg.get('test', [])

    # Fallback: if the list is empty, derive a single entry from datasets.name
    if not dataset_list:
        ds_name = datasets_cfg.get('name', '')
        if ds_name:
            dataset_list = [f"{ds_name}_{dataset_to_test}"]

    # If bbox aug is enabled in testing, simply set transforms to None and we will apply transforms later
    bbox_aug_enabled = test_cfg.get('bbox_aug', {}).get('enabled', False)

    transforms = None if not is_train and bbox_aug_enabled else build_transforms(cfg, is_train)

    datasets = build_dataset(cfg, dataset_list, transforms, is_train)

    if is_train:
        # save category_id to label name mapping
        output_dir = cfg.get('output_dir', '.')
        save_labels(datasets, output_dir)

    data_loaders = []
    for dataset in datasets:
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
        )
        
        bbox_aug_enabled = test_cfg.get('bbox_aug', {}).get('enabled', False)
        size_divisibility = dataloader_cfg.get('size_divisibility', 0)
        num_workers = dataloader_cfg.get('num_workers', 4)
        
        collator = BBoxAugCollator() if not is_train and bbox_aug_enabled else \
            BatchCollator(size_divisibility)
        
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            pin_memory=True,
        )
        
        data_loaders.append(data_loader)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders
