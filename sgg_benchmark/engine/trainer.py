# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import time
import datetime
import numpy as np
import json
import sys

import torch
import torch.distributed as dist

from sgg_benchmark.utils.comm import get_world_size, synchronize, get_rank
from sgg_benchmark.utils.checkpoint import DetectronCheckpointer
from sgg_benchmark.utils.logger import logger_step
from sgg_benchmark.engine.inference import inference, to_device
from sgg_benchmark.utils.viz_deformable_points import visualize_sampling_points, get_viz_batch

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def run_training_loop(cfg, model, train_data_loader, val_data_loaders, optimizer, scheduler, scaler, 
                      logger, args, device, mode, metric_to_track, available_metrics):
    """
    Main training loop function.
    
    Args:
        cfg: Configuration object
        model: Model to train
        train_data_loader: Training data loader
        val_data_loaders: Validation data loaders
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision
        logger: Logger instance
        args: Training arguments (dict with keys: distributed, local_rank, use_wandb, save_best, skip_test, project_name, validate_before_training, resume)
        device: Device to use
        mode: Task mode (sgdet, sgcls, predcls)
        metric_to_track: Metric to track from available_metrics
        available_metrics: Dictionary mapping metric names to their field names
    
    Returns:
        model: Trained model
        best_checkpoint: Path to best checkpoint
    """
    import wandb
    
    best_epoch = 0
    best_metric = 0.0
    best_checkpoint = None
    output_dir = cfg.output_dir
    
    max_epoch = cfg.solver.max_epoch
    use_amp = cfg.dtype == "float16"
    start_epoch = 0
    
    # Initialize wandb if enabled
    if args.get('use_wandb') and get_rank() == 0:
        # Get config name from args
        run_name = args.get('config_name', args.get('project_name', 'sgg-benchmark'))
        wandb.init(
            project="scene-graph-benchmark",
            entity="maelic",
            name=run_name,
            config=dict(cfg),
            dir=output_dir
        )
        logger.info("Initialized WandB with project=scene-graph-benchmark, entity=maelic, run_name={}".format(run_name))
    
    # Determine resume epoch from checkpoint metadata if available
    if args.get('resume') and os.path.exists(args['resume']):
        try:
            checkpoint_data = torch.load(args['resume'], map_location='cpu')
            if 'epoch' in checkpoint_data:
                start_epoch = checkpoint_data['epoch'] + 1
                logger.info("Resuming training from epoch {}".format(start_epoch))
            if 'best_metric' in checkpoint_data:
                best_metric = checkpoint_data.get('best_metric', 0.0)
            if 'best_epoch' in checkpoint_data:
                best_epoch = checkpoint_data.get('best_epoch', 0)
        except Exception as e:
            logger.warning("Could not extract epoch info from checkpoint: {}".format(e))
    
    logger.info("Start training for {} epochs (starting from epoch {})".format(max_epoch, start_epoch))
    start_training_time = time.time()

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk, custom_scheduler=True
    )
    
    for epoch in range(start_epoch, max_epoch):
        if args['distributed']:
            model.module.roi_heads.train()
            model.module.backbone.eval()
        else:
            model.roi_heads.train()
            model.backbone.eval()
        
        start_epoch_time = time.time()
        
        # Train one epoch
        _ = train_one_epoch(model, optimizer, train_data_loader, device, epoch, logger, cfg, scaler,
                           args['use_wandb'], use_amp, scheduler=scheduler)
        logger.info("Epoch {} training time: {:.2f} s".format(epoch, time.time() - start_epoch_time))
        
        # Clear CUDA cache between epochs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not args['save_best'] and not args.get('no_checkpoints', False):
            checkpointer.save("model_epoch_{}".format(epoch), epoch=epoch, best_metric=best_metric, best_epoch=best_epoch)
        
        # Validation
        val_result = None
        current_metric = None
        logger.info("Start validating")
        
        val_result = run_val(cfg, model, val_data_loaders, args['distributed'], logger, device=device)
        if mode + metric_to_track not in val_result.keys():
            logger.error("Metric to track not found in validation result, default to R")
            metric_to_track = "_recall"
        results = val_result[mode + metric_to_track]
        current_metric = float(np.mean(list(results.values())))
        logger.info("Average validation Result for %s: %.4f" % (cfg.metric_to_track, current_metric))
        
        if current_metric > best_metric:
            best_epoch = epoch
            best_metric = current_metric
            if args['save_best'] and not args.get('no_checkpoints', False):
                to_remove = best_checkpoint
                checkpointer.save("best_model_epoch_{}".format(epoch), epoch=epoch, best_metric=best_metric, best_epoch=best_epoch)
                best_checkpoint = os.path.join(output_dir, "best_model_epoch_{}".format(epoch))
                
                if to_remove is not None:
                    os.remove(to_remove + ".pth")
                    logger.info("New best model saved at iteration {}".format(epoch))
        
        logger.info("Now best epoch in {} is : {}, with value is {}".format(cfg.metric_to_track + "@k", best_epoch, best_metric))
        
        if args['use_wandb']:
            res_dict = {
                'avg_' + cfg.metric_to_track + "@k": current_metric,
            }
            
            # Add F1 score if available
            f1_key = mode + "_f1_score"
            if f1_key in val_result:
                res_dict[mode + '_f1'] = np.mean(list(val_result[f1_key].values())) if isinstance(val_result[f1_key], dict) else val_result[f1_key]
            
            # Add recall if available
            recall_key = mode + "_recall"
            if recall_key in val_result:
                res_dict[mode + '_recall'] = np.mean(list(val_result[recall_key].values())) if isinstance(val_result[recall_key], dict) else val_result[recall_key]
            
            # Add mean recall if available
            mean_recall_key = mode + "_mean_recall"
            if mean_recall_key in val_result:
                res_dict[mode + '_mean_recall'] = np.mean(list(val_result[mean_recall_key].values())) if isinstance(val_result[mean_recall_key], dict) else val_result[mean_recall_key]
            
            if cfg.test.get('informative', False):
                info_recall_key = mode + "_informative_recall"
                if info_recall_key in val_result:
                    res_dict[mode + '_informative_recall'] = val_result[info_recall_key]
            
            wandb.log(res_dict, step=epoch)
        
        # Scheduler step (epoch-level only; iter-based schedulers step inside train_one_epoch)
        if not getattr(scheduler, '_is_iter_based', False):
            if cfg.solver.schedule.type == "WarmupReduceLROnPlateau":
                scheduler.step(current_metric, epoch=epoch)
                if scheduler.stage_count >= cfg.solver.schedule.max_decay_step:
                    logger.info("Trigger MAX_DECAY_STEP at iteration {}.".format(epoch))
                    break
            else:
                scheduler.step()
    
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / epoch)".format(
            total_time_str, total_training_time / (max_epoch)
        )
    )
    
    name = "model_epoch_{}".format(best_epoch)
    if args['save_best']:
        name = "best_model_epoch_{}".format(best_epoch)
    last_filename = os.path.join(output_dir, "{}.pth".format(name))
    output_folder = os.path.join(output_dir, "last_checkpoint")
    with open(output_folder, "w") as f:
        f.write(last_filename)
    
    logger.info("Best Epoch is : %.4f" % best_epoch)
    
    return model, best_checkpoint


def train_one_epoch(model, optimizer, data_loader, device, epoch, logger, cfg, scaler, use_wandb=False, use_amp=True, scheduler=None):
    """Train for one epoch"""
    import tqdm
    import wandb
    # Detached logging mode: choose via env var, config, or auto-detect when stdout is not a TTY
    env_val = os.environ.get('DETACHED_LOGGING', None)
    if env_val is not None:
        detached_mode = env_val == '1'
    else:
        detached_mode = bool(cfg.solver.get('detached_logging', False)) or (not sys.stdout.isatty())

    log_interval = int(os.environ.get('TRAIN_LOG_INTERVAL', cfg.solver.get('log_interval', 200)))

    pbar = None if detached_mode else tqdm.tqdm(total=len(data_loader))
    if detached_mode:
        logger.info(f"Detached logging enabled (interval={log_interval})")

    epoch_start_time = time.time()

    # Reset peak memory stats at the start of epoch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    
    max_iter = cfg.solver.max_iter
    accum_steps = int(cfg.solver.get("accum_steps", 1))  # gradient accumulation steps
    
    for iteration, (images, targets, _) in enumerate(data_loader):
        if max_iter > 0 and iteration >= max_iter:
            logger.info(f"Reached max_iter={max_iter}, stopping epoch {epoch} early.")
            break
            
        if pbar is not None:
            pbar.update(1)
        
        if any(len(target) < 1 for target in targets):
            logger.error(f"Epoch={epoch} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}")
            continue
        end = time.time()
        
        images = images.to(device)
        targets = to_device(targets, device)
        
        # Note: If mixed precision is not used, this ends up doing nothing
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            loss_dict = model(images, targets)
            
            losses = sum(loss for loss in loss_dict.values())
            # Scale loss by 1/accum_steps so accumulated gradients equal a single
            # large-batch step — keeps gradient magnitudes independent of accum_steps.
            losses = losses / accum_steps
        
        # reduce losses over all GPUs for logging purposes (use unscaled value for display)
        loss_dict_reduced = reduce_loss_dict({k: v / accum_steps for k, v in loss_dict.items()})
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        
        # Detach loss for logging to prevent holding computation graph
        loss_value = losses_reduced.item()
        
        if use_wandb:
            wandb.log({"loss": loss_value}, step=epoch)
        
        # Scaling loss and backward
        scaler.scale(losses).backward()
        
        is_update_step = ((iteration + 1) % accum_steps == 0) or (iteration + 1 == len(data_loader))
        
        # Unscale the gradients of optimizer's assigned params in-place before clipping
        # Check if any gradients exist before unscaling and stepping to avoid GradScaler errors
        has_grads = any(p.grad is not None for group in optimizer.param_groups for p in group["params"])
        total_norm = 0.0
        n_zero_grad = 0

        if has_grads and is_update_step:
            scaler.unscale_(optimizer)
            # Compute gradient norms (stored; logged at log_interval to avoid spam)
            total_norm = 0.0
            n_zero_grad = 0
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if p.grad is None:
                    n_zero_grad += 1
                    continue
                pn = p.grad.data.norm(2).item()
                total_norm += pn ** 2
            total_norm = total_norm ** 0.5

            # fallback to native clipping, if no clip_grad_norm is used
            torch.nn.utils.clip_grad_norm_([p for _, p in model.named_parameters() if p.requires_grad and p.grad is not None], max_norm=cfg.solver.grad_norm_clip)

            scaler.step(optimizer)
            scaler.update()
            # Step per-iteration scheduler once per optimizer step (not per raw iteration)
            if scheduler is not None and getattr(scheduler, '_is_iter_based', False):
                scheduler.step()
        else:
            # If no gradients, we still need to update scaler if it was scaled
            # But normally we don't call step.
            if is_update_step:
                scaler.update()
        
        # Zero gradients only after the optimizer step, not between accumulation steps
        if is_update_step:
            optimizer.zero_grad(set_to_none=True)
                
        # get memory used from cuda
        if torch.cuda.is_available():
            current_mem = torch.cuda.memory_allocated(device) / 1024.0 / 1024.0
            
        # Build a compact per-component loss string (always computed; used in tqdm + interval log)
        # Short aliases: strip common suffixes so the bar stays narrow.
        _ALIASES = {"loss_rel": "rel", "l21_loss": "l21", "dist_loss2": "dist", "loss_dis": "dis",
                    "loss_relation": "rel", "loss_refine_obj": "obj", "loss_refine_rel": "rrel"}
        loss_parts = " | ".join(
            f"{_ALIASES.get(k, k)}={float(v):.3f}" for k, v in loss_dict_reduced.items()
        )
        comp_str = f"loss: [ {loss_parts} ]"

        # Update progress either via tqdm or periodic detached logging
        if pbar is not None:
            pbar.set_description(f"E{epoch} {comp_str}")
        elif detached_mode and (iteration % log_interval == 0):
            elapsed = time.time() - epoch_start_time
            iters_done = iteration + 1
            iters_total = len(data_loader)
            eta_s = elapsed / iters_done * (iters_total - iters_done)
            eta_str = str(datetime.timedelta(seconds=int(eta_s)))
            lr_now = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch} [{iters_done}/{iters_total}]"
                f" | loss={loss_value:.4f}"
                f" | {comp_str}"
                f" | lr={lr_now:.2e}"
                f" | elapsed={datetime.timedelta(seconds=int(elapsed))}"
                f" | ETA={eta_str}"
            )
        
        # Explicitly delete tensors to free memory immediately
        del loss_dict, losses, loss_dict_reduced, losses_reduced
        del images, targets
        
        # Periodic cache clearing during epoch (every 20 iterations to be safe)
        if iteration % 20 == 0 and iteration > 0:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    
    # Close progress bar, clear CUDA cache and collect garbage at the end of epoch
    if pbar is not None:
        pbar.close()
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Don't return loss tensor to avoid holding reference
    return None


def run_val(cfg, model, val_data_loaders, distributed, logger, device=None):
    """Run validation"""
    if distributed:
        model = model.module
    torch.cuda.empty_cache()
    iou_types = ("bbox",)
    if cfg.model.relation_on:
        iou_types = iou_types + ("relations",)
    if cfg.model.get('attribute_on', False):
        iou_types = iou_types + ("attributes",)
    
    dataset_result = inference(
        cfg,
        model,
        val_data_loaders[0],
        dataset_name="",
        iou_types=iou_types,
        box_only=cfg.model.get('rpn_only', False),
        device=device or cfg.model.device,
        expected_results=cfg.test.get('expected_results', []),
        expected_results_sigma_tol=cfg.test.get('expected_results_sigma_tol', 4),
        output_folder=None,
        logger=logger,
        informative=cfg.test.get('informative', False),
    )
    synchronize()
    
    if len(dataset_result) == 1:
        return dataset_result
    
    if distributed:
        for k1, v1 in dataset_result.items():
            for k2, v2 in v1.items():
                dataset_result[k1][k2] = torch.distributed.all_reduce(torch.tensor(np.mean(v2)).to(device).unsqueeze(0)).item() / torch.distributed.get_world_size()
    else:
        for k1, v1 in dataset_result.items():
            if type(v1) != dict or type(v1) != list:
                dataset_result[k1] = v1
                continue
            for k2, v2 in v1.items():
                if isinstance(v2, list):
                    # mean everything
                    v2 = [np.mean(v) for v in v2]
                dataset_result[k1][k2] = np.mean(v2)
    return dataset_result


# ---------------------------------------------------------------------------
# Task-mode helpers (Hydra / OmegaConf config style)
# ---------------------------------------------------------------------------

def get_mode(cfg):
    """Infer the task mode string from config flags.

    Returns one of ``"sgdet"``, ``"sgcls"``, or ``"predcls"``.
    """
    task = "sgdet"
    if cfg.model.roi_relation_head.use_gt_box:
        task = "sgcls"
        if cfg.model.roi_relation_head.use_gt_object_label:
            task = "predcls"
    return task


def assert_mode(cfg, task):
    """Validate that the config flags are consistent with *task*.

    Raises ``AssertionError`` if the flags do not match.
    """
    if task == "predcls":
        assert cfg.model.roi_relation_head.use_gt_box, \
            "predcls mode requires model.roi_relation_head.use_gt_box=true"
        assert cfg.model.roi_relation_head.use_gt_object_label, \
            "predcls mode requires model.roi_relation_head.use_gt_object_label=true"
    elif task == "sgcls":
        assert cfg.model.roi_relation_head.use_gt_box, \
            "sgcls mode requires model.roi_relation_head.use_gt_box=true"
        assert not cfg.model.roi_relation_head.use_gt_object_label, \
            "sgcls mode requires model.roi_relation_head.use_gt_object_label=false"
    elif task == "sgdet":
        assert not cfg.model.roi_relation_head.use_gt_box, \
            "sgdet mode requires model.roi_relation_head.use_gt_box=false"
        assert not cfg.model.roi_relation_head.use_gt_object_label, \
            "sgdet mode requires model.roi_relation_head.use_gt_object_label=false"


def run_test(cfg, model, distributed, logger, output_dir=None):
    """Run inference on all test splits defined in *cfg*.

    The model is expected to already have the correct weights loaded.
    Results are written to *output_dir* (defaults to ``cfg.output_dir``).
    """
    from sgg_benchmark.data import make_data_loader

    if distributed:
        model = model.module
    torch.cuda.empty_cache()

    iou_types = ("bbox", "relations") if cfg.model.relation_on else ("bbox",)
    out_dir = output_dir or cfg.output_dir

    test_data_loaders = make_data_loader(cfg, mode='test', is_distributed=distributed)
    dataset_names = cfg.datasets.test

    results = {}
    for dataset_name, test_loader in zip(dataset_names, test_data_loaders):
        dataset_result = inference(
            cfg,
            model,
            test_loader,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=not cfg.model.relation_on,
            device=cfg.model.device,
            expected_results=cfg.test.expected_results,
            expected_results_sigma_tol=cfg.test.expected_results_sigma_tol,
            output_folder=out_dir,
            logger=logger,
        )
        synchronize()
        results[dataset_name] = dataset_result

    return results