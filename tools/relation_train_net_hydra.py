#!/usr/bin/env python3
"""
Hydra-compatible training script for Scene Graph Generation.

TRAINING USAGE (Training + Testing):
    python tools/relation_train_net_hydra.py \
        --config-path ../configs/hydra/PSG \
        --config-name react_yolo12m \
        --task sgdet \
        --save-best

RESUME TRAINING FROM CHECKPOINT:
    python tools/relation_train_net_hydra.py \
        --config-path ../configs/hydra/PSG \
        --config-name react_yolo12m \
        --task sgdet \
        --save-best \
        --resume /path/to/checkpoint.pth

VALIDATE PRETRAINED BACKBONE BEFORE TRAINING:
    python tools/relation_train_net_hydra.py \
        --config-path ../configs/hydra/PSG \
        --config-name react_yolo12m \
        --task sgdet \
        --save-best \
        --validate-before-training

EVALUATION-ONLY USAGE (Test set evaluation only):
    python tools/relation_train_net_hydra.py \
        --eval-only \
        --checkpoint /path/to/checkpoint.pth \
        --config-path ../configs/hydra/PSG \
        --config-name react_yolo12m \
        --task sgdet

OVERRIDE PARAMETERS:
    python tools/relation_train_net_hydra.py \
        --config-path ../configs/hydra/PSG \
        --config-name react_yolo12m \
        solver.base_lr=0.001 \
        solver.max_epoch=30 \
        model.roi_relation_head.use_gt_box=false
"""

# Set up custom environment before nearly anything else is imported
from sgg_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sgg_benchmark.modeling.detector import build_detection_model
from omegaconf import open_dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sgg_benchmark.data import make_data_loader
from sgg_benchmark.engine.inference import inference
from sgg_benchmark.engine.trainer import get_mode, assert_mode, run_test
from sgg_benchmark.utils.checkpoint import DetectronCheckpointer
from sgg_benchmark.utils.comm import synchronize, get_rank
from sgg_benchmark.utils.logger import setup_logger
from sgg_benchmark.utils.miscellaneous import mkdir, save_config, set_seed
from sgg_benchmark.config.hydra_config import get_cfg

def parse_hydra_args():
    """Parse arguments for Hydra mode"""
    parser = argparse.ArgumentParser(description="SGG Training with Hydra")
    
    # Hydra config location
    parser.add_argument("--config-path", type=str, default="../configs/hydra",
                        help="Path to config directory (relative to this script)")
    parser.add_argument("--config-name", type=str, default="default",
                        help="Name of config file (without .yaml)")
    
    # Mode selection
    parser.add_argument("--eval-only", action="store_true",
                        help="Run evaluation only (no training)")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to checkpoint to load for evaluation (required for --eval-only)")
    
    # Training options
    parser.add_argument("--task", type=str, default="sgdet",
                        choices=["predcls", "sgcls", "sgdet"],
                        help="Task mode")
    parser.add_argument("--no-checkpoints", action="store_true",
                        help="Disable all checkpoint saving (useful for ablations)")
    parser.add_argument("--save-best", action="store_true",
                        help="Save best checkpoint based on validation metric")
    parser.add_argument("--use-wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--skip-test", action="store_true",
                        help="Skip final test evaluation")
    parser.add_argument("--name", type=str, default="sgg-benchmark",
                        help="Project name for wandb")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--validate-before-training", action="store_true",
                        help="Run validation on the pretrained backbone before starting training")
    
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs to use for training/testing")
    
    # Distributed training
    parser.add_argument("--local-rank", type=int, default=0,
                        help="Local rank for distributed training")
    
    args, unknown = parser.parse_known_args()
    return args, unknown


def eval_only(hydra_cfg: DictConfig, args):
    """Evaluation-only mode: load dataset, extract class info, then build model and evaluate"""
    
    # Setup distributed training
    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    args.distributed = num_gpus > 1
    
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    # Set seed
    set_seed(seed=hydra_cfg.seed)
    
    # Setup output directory
    output_dir = hydra_cfg.output_dir
    if output_dir:
        mkdir(output_dir)
    
    # Setup logger
    logger = setup_logger("sgg_benchmark", output_dir, get_rank(), verbose=hydra_cfg.verbose, steps=True)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info("Task mode: {}".format(args.task))
    logger.info("Running in EVALUATION-ONLY mode")
    
    # Assert task mode
    if args.task:
        assert_mode(hydra_cfg, args.task)
    
    # Validate checkpoint path
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        logger.error("Checkpoint path is required for eval-only mode and must exist")
        logger.error("Provided checkpoint: {}".format(args.checkpoint if args.checkpoint else "NONE"))
        raise ValueError("Invalid checkpoint path for evaluation mode")
    
    # Load dataset FIRST to get class information
    logger.info("Loading test dataset to extract class information...")
    data_loaders_val = make_data_loader(hydra_cfg, mode='test', is_distributed=args.distributed)
    dataset_names = hydra_cfg.datasets.test
    
    # Extract num classes from first dataset
    if data_loaders_val and len(data_loaders_val) > 0:
        first_dataset = data_loaders_val[0].dataset
        num_obj_classes = len(first_dataset.ind_to_classes) if hasattr(first_dataset, 'ind_to_classes') else 0
        num_rel_classes = len(first_dataset.ind_to_predicates) if hasattr(first_dataset, 'ind_to_predicates') else 0
        
        logger.info("Extracted from dataset: num_obj_classes={}, num_rel_classes={}".format(num_obj_classes, num_rel_classes))
        
        # Update config with extracted class counts
        with open_dict(hydra_cfg):
            hydra_cfg.model.roi_box_head.num_classes = num_obj_classes
            hydra_cfg.model.roi_relation_head.num_classes = num_rel_classes
    else:
        logger.warning("Could not extract class information from dataset, using config defaults")
    
    # Config is ready to use (dataset loading has populated class counts)
    cfg = hydra_cfg
    
    logger.info("Loading checkpoint: {}".format(args.checkpoint))
    
    # Build model
    model = build_detection_model(cfg)
    model = model.to(cfg.model.device)
    
    # Load checkpoint
    checkpointer = DetectronCheckpointer(cfg, model)
    checkpointer.load(args.checkpoint)
    logger.info("Checkpoint loaded successfully")
    
    # Run evaluation
    logger.info("Running evaluation on test set...")
    
    # Determine evaluation types based on model configuration
    # Include "relations" if the model has relation head enabled
    if cfg.model.relation_on:
        iou_types = ("bbox", "relations")
        logger.info("Evaluating bbox and relations")
    else:
        iou_types = ("bbox",)
        logger.info("Evaluating bbox only (relation head disabled)")

    # Run inference on each test dataset
    all_results = []
    for dataset_name, data_loader in zip(dataset_names, data_loaders_val):
        logger.info("Evaluating on dataset: {}".format(dataset_name))
        
        test_results = inference(
            cfg,
            model,
            data_loader,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.model.relation_on else True,
            device=cfg.model.device,
            expected_results=cfg.test.expected_results,
            expected_results_sigma_tol=cfg.test.expected_results_sigma_tol,
            output_folder=output_dir,
            logger=logger,
        )
        all_results.append((dataset_name, test_results))
    
    logger.info("Evaluation completed!")
    for dataset_name, results in all_results:
        logger.info("=" * 100)
        logger.info("Scene Graph Detection Results for {}".format(dataset_name))
        logger.info("=" * 100)
        
        if results == -1:
            logger.warning("No results available for {}".format(dataset_name))
        elif isinstance(results, dict):
            # Extract mode (sgdet, sgcls, predcls)
            modes = set()
            for key in results.keys():
                if '_' in key:
                    mode = key.rsplit('_', 1)[0]
                    if mode in ['sgdet', 'sgcls', 'predcls']:
                        modes.add(mode)
            
            for mode in sorted(modes):
                logger.info("\n" + "=" * 100)
                logger.info("Mode: {}".format(mode.upper()))
                logger.info("=" * 100)
                
                # Display Recall metrics
                if mode + '_recall' in results:
                    result_str = 'SGG eval: '
                    for k in sorted(results[mode + '_recall'].keys()):
                        v = results[mode + '_recall'][k]
                        result_str += '    R @ %d: %.4f; ' % (k, np.mean(v) if isinstance(v, list) else v)
                    result_str += ' for mode=%s, type=Recall (Main).' % mode
                    logger.info(result_str)
                
                # Display Mean Recall metrics
                if mode + '_mean_recall' in results:
                    result_str = 'SGG eval: '
                    for k in sorted(results[mode + '_mean_recall'].keys()):
                        v = results[mode + '_mean_recall'][k]
                        result_str += '   mR @ %d: %.4f; ' % (k, float(v) if isinstance(v, (int, float)) else np.mean(v))
                    result_str += ' for mode=%s, type=Mean Recall.' % mode
                    logger.info(result_str)
                
                # Display F1 Score metrics
                if mode + '_f1_score' in results:
                    result_str = 'SGG eval: '
                    for k in sorted(results[mode + '_f1_score'].keys()):
                        v = results[mode + '_f1_score'][k]
                        result_str += '    F1 @ %d: %.4f; ' % (k, v)
                    result_str += ' for mode=%s, type=F1.' % mode
                    logger.info(result_str)
                
                logger.info("")
        else:
            logger.info("Results: {}".format(results))
        
        logger.info("=" * 100)


def main_hydra(hydra_cfg: DictConfig, args):
    """Main training function using Hydra config: load data first, get classes, then build model"""
    from sgg_benchmark.solver import make_optimizer, make_lr_scheduler
    from sgg_benchmark.engine.trainer import run_training_loop
    
    # Setup distributed training
    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    args.distributed = num_gpus > 1
    
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    # Set seed
    set_seed(seed=hydra_cfg.seed)
    
    # Setup output directory
    output_dir = hydra_cfg.output_dir
    if output_dir:
        mkdir(output_dir)
    
    # Setup logger
    logger = setup_logger("sgg_benchmark", output_dir, get_rank(), verbose=hydra_cfg.verbose, steps=True)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info("Task mode: {}".format(args.task))
    
    # Assert task mode
    if args.task:
        assert_mode(hydra_cfg, args.task)
    
    # Load training dataset FIRST to get class information
    logger.info("Loading training dataset to extract class information...")
    train_data_loader = make_data_loader(hydra_cfg, mode='train', is_distributed=args.distributed)

    # DEBUG: Set a maximum number of iterations per epoch for faster debugging if requested
    # Can be set via env var SGG_DEBUG_ITER or in config as DEBUG_MAX_ITER
    debug_iter = int(os.environ.get("SGG_DEBUG_ITER", 0))
    if debug_iter > 0:
        logger.info(f"DEBUG: Limiting train iterations to {debug_iter} per epoch")
        with open_dict(hydra_cfg):
            hydra_cfg.DEBUG_MAX_ITER = debug_iter
    
    # Extract num classes from training dataset
    if train_data_loader is not None:
        train_dataset = train_data_loader.dataset
        num_obj_classes = len(train_dataset.ind_to_classes) if hasattr(train_dataset, 'ind_to_classes') else 0
        num_rel_classes = len(train_dataset.ind_to_predicates) if hasattr(train_dataset, 'ind_to_predicates') else 0
        
        logger.info("Extracted from dataset: num_obj_classes={}, num_rel_classes={}".format(num_obj_classes, num_rel_classes))
        
        # Update config with extracted class counts
        with open_dict(hydra_cfg):
            hydra_cfg.model.roi_box_head.num_classes = num_obj_classes
            hydra_cfg.model.roi_relation_head.num_classes = num_rel_classes
    else:
        logger.warning("Could not extract class information from dataset, using config defaults")
    
    # Config is ready to use (dataset loading has populated class counts)
    cfg = hydra_cfg
    
    # Save config
    output_config_path = os.path.join(cfg.output_dir, 'config.yml')
    logger.info("Saving config to: {}".format(output_config_path))
    save_config(cfg, output_config_path)
    
    # Also save Hydra config
    hydra_config_path = os.path.join(cfg.output_dir, 'hydra_config.yaml')
    OmegaConf.save(hydra_cfg, hydra_config_path)
    logger.info("Saving Hydra config to: {}".format(hydra_config_path))
    
    # Build model
    logger.info("Building model...")
    model = build_detection_model(cfg)
    device = torch.device(cfg.model.device)
    model.to(device)
    
    # Setup distributed training wrapper if needed
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    
    # Build optimizer and scheduler
    logger.info("Building optimizer and scheduler...")
    
    # Determine slow heads based on predictor type
    slow_heads = []
    if cfg.model.roi_relation_head.predictor == "IMPPredictor":
        slow_heads = [
            "roi_heads.relation.box_feature_extractor",
            "roi_heads.relation.union_feature_extractor.feature_extractor",
        ]
    elif cfg.model.roi_relation_head.predictor == 'SquatPredictor':
        slow_heads = ['roi.heads.relation.predictor.context_layer.mask_predictor']
    
    # Add backbone to slow heads if fine-tuning is enabled
    # We use a lower learning rate for the backbone to maintain stability
    if not cfg.model.backbone.freeze:
        slow_heads.append("backbone")
        logger.info("Differential learning rate: backbone will be trained with a lower LR")
    
    optimizer = make_optimizer(cfg, model, logger, slow_heads=slow_heads, slow_ratio=getattr(cfg.solver, "slow_ratio", 2.5), rl_factor=float(num_gpus))
    # Compute effective iters/epoch (respects max_iter early-stop cap if set)
    _raw_iters = len(train_data_loader)
    _max_iter = getattr(cfg.solver, 'max_iter', 0)
    iters_per_epoch = min(_raw_iters, _max_iter) if _max_iter > 0 else _raw_iters
    scheduler = make_lr_scheduler(cfg, optimizer, logger, iters_per_epoch=iters_per_epoch)
    
    # Initialize mixed-precision training
    use_amp = cfg.dtype == "float16"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    # Load validation data loaders
    logger.info("Loading validation dataset...")
    val_data_loaders = make_data_loader(cfg, mode='val', is_distributed=args.distributed)
    
    # Setup available metrics for validation
    available_metrics = {
        "mR": "_mean_recall",
        "R": "_recall",
        "F1": "_f1_score",
        "zR": "_zeroshot_recall",
        "ng-zR": "_ng_zeroshot_recall",
        "ng-R": "_recall_nogc",
        "ng-mR": "_ng_mean_recall",
        "topA": ["_accuracy_hit", "_accuracy_count"]
    }
    metric_to_track = available_metrics.get(cfg.metric_to_track, "_recall")
    
    # Setup checkpointer (needed for resume)
    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk, custom_scheduler=True, logger=logger
    )
    
    # Load pretrained backbone detector if specified
    if cfg.model.get('pretrained_detector_ckpt') and os.path.exists(cfg.model.pretrained_detector_ckpt):
        logger.info("Loading pretrained backbone detector from: {}".format(cfg.model.pretrained_detector_ckpt))
        checkpointer.load_backbone(cfg.model.pretrained_detector_ckpt)
    
    # Ensure relation predictor is in training mode for backprop
    if hasattr(model, 'roi_heads') and hasattr(model.roi_heads, 'relation'):
        model.roi_heads.train()
        model.backbone.eval()   # YOLO must stay in eval() so detection head outputs decoded predictions for NMS
        for name, param in model.roi_heads.relation.named_parameters():
            param.requires_grad = True
        logger.info("Relation predictor set to training mode")
    
    # Validate before training if requested (to verify pretrained backbone performance)
    if args.validate_before_training:
        logger.info("Running pre-training validation on pretrained backbone...")
        model.eval()
        pre_val_results = inference(
            cfg,
            model,
            val_data_loaders[0] if val_data_loaders else None,
            dataset_name="pretrained",
            iou_types=("bbox", "relations") if cfg.model.relation_on else ("bbox",),
            box_only=False if cfg.model.relation_on else True,
            device=cfg.model.device,
            expected_results=cfg.test.expected_results,
            expected_results_sigma_tol=cfg.test.expected_results_sigma_tol,
            output_folder=output_dir,
            logger=logger,
        )
        logger.info("Pre-training validation completed")
        
        # Set model back to training mode
        if hasattr(model, 'roi_heads') and hasattr(model.roi_heads, 'relation'):
            model.roi_heads.train()
    
    # Load checkpoint to resume training if provided
    if args.resume:
        if not os.path.exists(args.resume):
            logger.error("Resume checkpoint not found: {}".format(args.resume))
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        logger.info("Loading checkpoint to resume training: {}".format(args.resume))
        checkpointer.load(args.resume)
        logger.info("Checkpoint loaded successfully for resuming training")
    
    mode = get_mode(cfg)
    
    # Training arguments
    training_args = {
        "task": args.task,
        "save_best": args.save_best,
        "no_checkpoints": args.no_checkpoints,
        "use_wandb": args.use_wandb,
        "skip_test": args.skip_test,
        "local_rank": args.local_rank,
        "distributed": args.distributed,
        "project_name": args.name,
        "resume": args.resume,
        "validate_before_training": args.validate_before_training,
        "config_name": args.config_name,  # Pass config name for wandb run name
    }


    # print total params and trainable params
    from torch.nn.parameter import UninitializedParameter
    total_params = sum(p.numel() for p in model.parameters() if not isinstance(p, UninitializedParameter))
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad and not isinstance(p, UninitializedParameter))
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Run training loop using the new trainer module
    logger.info("Starting training loop...")
    model, best_checkpoint = run_training_loop(
        cfg=cfg,
        model=model,
        train_data_loader=train_data_loader,
        val_data_loaders=val_data_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        logger=logger,
        args=training_args,
        device=device,
        mode=mode,
        metric_to_track=metric_to_track,
        available_metrics=available_metrics,
    )
    
    # Test
    if not args.skip_test:
        logger.info("Running final evaluation...")
        checkpointer = DetectronCheckpointer(cfg, model)
        last_check = best_checkpoint + '.pth' if best_checkpoint else ""
        if last_check and os.path.exists(last_check):
            logger.info("Loading best checkpoint from {}...".format(last_check))
            _ = checkpointer.load(last_check)
        run_test(cfg, model, args.distributed, logger, output_dir=output_dir)
            
def main():
    """Entry point"""
    args, unknown = parse_hydra_args()
    
    # Validation for eval-only mode
    if args.eval_only and not args.checkpoint:
        print("ERROR: --checkpoint is required when using --eval-only mode")
        sys.exit(1)
    
    # Use Hydra config
    print("Using Hydra + OmegaConf config mode...")
    print(f"Config path: {args.config_path}")
    print(f"Config name: {args.config_name}")
    
    if args.eval_only:
        print(f"Mode: EVALUATION ONLY")
        print(f"Checkpoint: {args.checkpoint}")
    else:
        print(f"Mode: TRAINING")
    
    # Load Hydra config manually (not using @hydra.main decorator for more control)
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    # Get absolute config path
    if os.path.isabs(args.config_path):
        config_path = args.config_path
    else:
        config_path = os.path.join(os.path.dirname(__file__), args.config_path)
    config_path = os.path.abspath(config_path)
    
    print(f"Loading config from: {config_path}/{args.config_name}.yaml")
    
    # Initialize Hydra
    with initialize_config_dir(config_dir=config_path, version_base=None):
        # Compose config with overrides from command line
        overrides = []
        
        # Add task override
        # Use ++ prefix (Hydra "upsert": add or override) so the override works
        # regardless of whether the key is already present in the config file.
        if args.task:
            # Update GT box/label based on task
            if args.task == "predcls":
                overrides.extend([
                    "++model.roi_relation_head.use_gt_box=true",
                    "++model.roi_relation_head.use_gt_object_label=true"
                ])
            elif args.task == "sgcls":
                overrides.extend([
                    "++model.roi_relation_head.use_gt_box=true",
                    "++model.roi_relation_head.use_gt_object_label=false"
                ])
            elif args.task == "sgdet":
                overrides.extend([
                    "++model.roi_relation_head.use_gt_box=false",
                    "++model.roi_relation_head.use_gt_object_label=false"
                ])
        
        # Add any additional overrides from command line
        if unknown:
            overrides.extend(unknown)
        
        cfg = compose(config_name=args.config_name, overrides=overrides)
        
        # Merge with defaults from SGGBenchmarkConfig
        cfg = get_cfg(cfg)
        
        # Run training or evaluation
        if args.eval_only:
            eval_only(cfg, args)
        else:
            main_hydra(cfg, args)


if __name__ == "__main__":
    main()
