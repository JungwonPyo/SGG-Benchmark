#!/usr/bin/env python3
"""
Evaluation script for Scene Graph Generation with Hydra configs.

Key fix:
- Keep cfg.output_dir pointing to the training checkpoint directory so
  dataset statistics are loaded from the same cache/convention used in training.
- Save evaluation artifacts to a separate eval_output_dir.
"""

import argparse
import os
import sys
from pathlib import Path

import torch


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sgg_benchmark.config.hydra_config import Config, load_config_from_file
from sgg_benchmark.data import make_data_loader, get_dataset_statistics
from sgg_benchmark.engine.inference import inference
from sgg_benchmark.modeling.detector import build_detection_model
from sgg_benchmark.utils.checkpoint import DetectronCheckpointer
from sgg_benchmark.utils.comm import synchronize, get_rank
from sgg_benchmark.utils.logger import setup_logger
from sgg_benchmark.utils.miscellaneous import mkdir, set_seed


def get_cfg_dict(cfg):
    if hasattr(cfg, "cfg"):
        return cfg.cfg
    if hasattr(cfg, "_cfg"):
        return cfg._cfg
    return cfg


def find_config_file(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    config_candidates = [
        checkpoint_dir / "config.yaml",
        checkpoint_dir / "hydra_config.yaml",
        checkpoint_dir / ".hydra" / "config.yaml",
    ]

    for config_path in config_candidates:
        if config_path.exists():
            print(f"Found config at: {config_path}")
            return str(config_path)

    raise FileNotFoundError(
        f"Could not find config file in {checkpoint_dir}. "
        f"Looked in: {[str(p) for p in config_candidates]}"
    )


def find_checkpoint_file(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)

    checkpoint_candidates = [
        checkpoint_dir / "model_best.pth",
        checkpoint_dir / "model_final.pth",
    ]

    for ckpt in checkpoint_candidates:
        if ckpt.exists():
            print(f"Found checkpoint: {ckpt}")
            return str(ckpt)

    epoch_ckpts = sorted(checkpoint_dir.glob("*model_epoch_*.pth"))
    if epoch_ckpts:
        latest = epoch_ckpts[-1]
        print(f"Found latest epoch checkpoint: {latest}")
        return str(latest)

    raise FileNotFoundError(
        f"Could not find any checkpoint in {checkpoint_dir}. "
        f"Looked for: model_best.pth, model_final.pth, model_epoch_*.pth"
    )


def load_hydra_config(config_file):
    hydra_cfg = load_config_from_file(config_file)
    cfg = Config(hydra_cfg)
    return cfg, hydra_cfg


def assert_task_mode(cfg, task):
    cfg_dict = get_cfg_dict(cfg)

    if task == "sgdet":
        cfg_dict.model.roi_relation_head.use_gt_box = False
        cfg_dict.model.roi_relation_head.use_gt_object_label = False
        print(f"Task={task}: Full detection mode")
    elif task == "sgcls":
        cfg_dict.model.roi_relation_head.use_gt_box = True
        cfg_dict.model.roi_relation_head.use_gt_object_label = False
        print(f"Task={task}: Using GT boxes")
    elif task == "predcls":
        cfg_dict.model.roi_relation_head.use_gt_box = True
        cfg_dict.model.roi_relation_head.use_gt_object_label = True
        print(f"Task={task}: Using GT boxes and labels")


def enable_inplace_relu(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.ReLU):
            setattr(model, name, torch.nn.ReLU(inplace=True))
        else:
            enable_inplace_relu(module)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SGG Model")

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Directory containing checkpoint and config (auto-detects best/final)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Specific checkpoint file path",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Config file (if not in checkpoint dir)",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="sgdet",
        choices=["predcls", "sgcls", "sgdet"],
        help="Evaluation task mode",
    )
    parser.add_argument(
        "--test-split",
        type=str,
        default=None,
        help="Dataset split to evaluate (default: use config)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for evaluation logs/predictions",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save prediction results",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations",
    )

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--debug-stats", action="store_true", help="Print dataset statistics info")

    args = parser.parse_args()

    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

        checkpoint_path = args.checkpoint or find_checkpoint_file(checkpoint_dir)
        config_file = args.config_file or find_config_file(checkpoint_dir)

    elif args.checkpoint:
        checkpoint_path = args.checkpoint
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint_dir = Path(checkpoint_path).parent
        config_file = args.config_file or find_config_file(checkpoint_dir)
    else:
        raise ValueError("Must provide either --checkpoint-dir or --checkpoint")

    print("=" * 80)
    print("EVALUATION SETUP")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config:     {config_file}")
    print(f"Task:       {args.task}")
    print("=" * 80)

    cfg, hydra_cfg = load_hydra_config(config_file)
    cfg_dict = get_cfg_dict(cfg)

    assert_task_mode(cfg, args.task)

    if args.test_split:
        cfg_dict.datasets.test = [args.test_split]
        print(f"Overriding test split: {args.test_split}")

    if args.output_dir:
        eval_output_dir = args.output_dir
    else:
        eval_output_dir = str(checkpoint_dir / f"inference_{args.task}")
    mkdir(eval_output_dir)

    # IMPORTANT:
    # Keep stats/cache resolution aligned with training-time checkpoint directory.
    cfg_dict.output_dir = str(checkpoint_dir)

    seed = cfg.SEED if hasattr(cfg, "SEED") else (cfg.seed if hasattr(cfg, "seed") else 42)
    set_seed(seed)

    verbose = (
        cfg.VERBOSE
        if hasattr(cfg, "VERBOSE")
        else (cfg.verbose if hasattr(cfg, "verbose") else "INFO")
    )
    logger = setup_logger(
        "sgg_benchmark",
        eval_output_dir,
        get_rank(),
        filename=f"eval_{args.task}.log",
        verbose=verbose,
    )
    logger.info(f"Using {num_gpus} GPUs")
    logger.info(f"Evaluating: {checkpoint_path}")
    logger.info(f"Task mode: {args.task}")
    logger.info(f"Training stats/checkpoint dir: {checkpoint_dir}")
    logger.info(f"Evaluation output dir: {eval_output_dir}")

    if args.debug_stats:
        print(f"[DEBUG] cfg.output_dir used for statistics/model build: {cfg_dict.output_dir}")

    logger.info("Building model...")
    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    enable_inplace_relu(model)

    logger.info("Loading checkpoint...")
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=eval_output_dir)
    _ = checkpointer.load(checkpoint_path)

    if args.debug_stats:
        stats = get_dataset_statistics(cfg)
        obj_classes = stats.get("obj_classes", [])
        rel_classes = stats.get("rel_classes", [])
        fg_matrix = stats.get("fg_matrix", None)
        print(f"[DEBUG] len(obj_classes) = {len(obj_classes)}")
        print(f"[DEBUG] len(rel_classes) = {len(rel_classes)}")
        print(f"[DEBUG] fg_matrix shape = {None if fg_matrix is None else tuple(fg_matrix.shape)}")

    if "world" in cfg.MODEL.BACKBONE.TYPE.lower():
        logger.info("Loading text embeddings for YOLO World...")
        stats = get_dataset_statistics(cfg)
        obj_classes = stats["obj_classes"]
        if obj_classes and str(obj_classes[0]).lower() in {"__background__", "__bg__", "background"}:
            obj_classes = obj_classes[1:]
        model.backbone.load_txt_feats(obj_classes)

    model.backbone.eval()
    model.roi_heads.eval()

    iou_types = ("bbox",)

    relation_on = (
        getattr(cfg.MODEL, "RELATION_ON", None)
        or getattr(cfg, "relation_on", False)
        or (hasattr(cfg, "model") and getattr(cfg.model, "relation_on", False))
    )
    attribute_on = (
        getattr(cfg.MODEL, "ATTRIBUTE_ON", None)
        or getattr(cfg, "attribute_on", False)
        or (hasattr(cfg, "model") and getattr(cfg.model, "attribute_on", False))
    )

    if relation_on:
        logger.info("Evaluating relations")
        iou_types = iou_types + ("relations",)
    if attribute_on:
        logger.info("Evaluating attributes")
        iou_types = iou_types + ("attributes",)

    raw_cfg = get_cfg_dict(cfg)
    datasets_cfg = raw_cfg.get("datasets", {})
    dataset_names = list(datasets_cfg.get("test", []) or [])
    if not dataset_names:
        ds_name = datasets_cfg.get("name", "")
        if ds_name:
            dataset_names = [f"{ds_name}_test"]
    logger.info(f"Evaluating on dataset(s): {dataset_names}")

    data_loaders_val = make_data_loader(
        cfg=cfg,
        mode="test",
        is_distributed=distributed,
        dataset_to_test="test",
    )

    dtype = getattr(cfg, "DTYPE", None) or getattr(cfg, "dtype", "float32")
    use_amp = dtype == "float16" or args.amp

    detections_per_img = 100
    if hasattr(cfg, "MODEL") and hasattr(cfg.MODEL, "ROI_HEADS"):
        detections_per_img = getattr(cfg.MODEL.ROI_HEADS, "DETECTIONS_PER_IMG", 100)
    elif hasattr(cfg, "model") and hasattr(cfg.model, "roi_heads"):
        detections_per_img = getattr(cfg.model.roi_heads, "detections_per_img", 100)

    for dataset_name, data_loader_val in zip(dataset_names, data_loaders_val):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Evaluating on: {dataset_name}")
        logger.info(f"{'=' * 80}\n")

        output_folder = os.path.join(
            eval_output_dir,
            f"{dataset_name}_{args.task}_det{detections_per_img}"
        )
        mkdir(output_folder)

        rpn_only = getattr(cfg.MODEL, "RPN_ONLY", False) if hasattr(cfg, "MODEL") else False
        device = (
            getattr(cfg.MODEL, "DEVICE", "cuda")
            if hasattr(cfg, "MODEL")
            else (getattr(cfg.model, "device", "cuda") if hasattr(cfg, "model") else "cuda")
        )

        expected_results = []
        expected_sigma = 4
        if hasattr(cfg, "TEST"):
            expected_results = getattr(cfg.TEST, "EXPECTED_RESULTS", [])
            expected_sigma = getattr(cfg.TEST, "EXPECTED_RESULTS_SIGMA_TOL", 4)
        elif hasattr(cfg, "test"):
            expected_results = getattr(cfg.test, "expected_results", [])
            expected_sigma = getattr(cfg.test, "expected_results_sigma_tol", 4)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            _ = inference(
                cfg,
                model,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=rpn_only,
                device=device,
                expected_results=expected_results,
                expected_results_sigma_tol=expected_sigma,
                output_folder=output_folder if args.save_predictions else None,
                logger=logger,
                informative=True,
            )

        synchronize()

        logger.info(f"\n{'=' * 80}")
        logger.info(f"EVALUATION COMPLETE: {dataset_name}")
        logger.info(f"Results saved to: {output_folder}")
        logger.info(f"{'=' * 80}\n")

    del model
    torch.cuda.empty_cache()
    logger.info("Evaluation finished!")


if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()