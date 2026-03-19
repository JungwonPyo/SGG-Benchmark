# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .lr_scheduler import (
    WarmupMultiStepLR, WarmupReduceLROnPlateau,
    WarmupCosineAnnealingLR, WarmupCosineAnnealingIterLR,
)
from .optimizer import MuSGD, Muon, MuonWithAdamW


def make_optimizer(cfg, model, logger, slow_heads=None, slow_ratio=5.0, rl_factor=1.0):
    params = []
    
    use_muon_global = getattr(cfg.solver, "optimizer", "SGD") == "MuonWithAdamW"
    
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.solver.base_lr
        weight_decay = cfg.solver.weight_decay

        # Deformable sampler geometry params need a higher LR: the gradient through bilinear
        # sampling is very weak (depends on feature-map spatial gradients), so delta_proj
        # and attn_proj/temp would barely move at the global LR.
        _delta_lr_mult = 10.0
        if any(tag in key for tag in ('deformable_union.delta_proj',
                                       'deformable_union.attn_proj',
                                       'deformable_union.attn_temp',
                                       'deformable_union.delta_geo_proj')):
            lr = lr * _delta_lr_mult
        # Muon works best on 2D weight matrices. We exclude 1D params (biases, norms, embeddings)
        use_muon_param = False
        if use_muon_global:
            # Check if it's a weight matrix that would benefit from orthogonalization
            if value.ndim >= 2 and "weight" in key and "norm" not in key.lower() and "embedding" not in key.lower():
                use_muon_param = True

        # level_projs are frozen (requires_grad=False) so they are skipped by
        # the `if not value.requires_grad: continue` guard above.
        if "bias" in key:
            # Use the already-set lr (which may have been raised for delta_proj etc.) times bias_lr_factor,
            # rather than resetting from base_lr.
            lr = lr * cfg.solver.bias_lr_factor
            weight_decay = cfg.solver.weight_decay_bias
        if slow_heads is not None:
            for item in slow_heads:
                if item in key:
                    logger.info("SLOW HEADS: {} is slow down by ratio of {}.".format(key, str(slow_ratio)))
                    lr = lr / slow_ratio
                    break
        
        param_group = {
            "params": [value], 
            "lr": lr * rl_factor, 
            "weight_decay": weight_decay,
            "use_muon": use_muon_param
        }
        params.append(param_group)

    if cfg.solver.optimizer == "SGD":
        optimizer = torch.optim.SGD(params, lr=cfg.solver.base_lr * rl_factor, momentum=cfg.solver.momentum)
    elif cfg.solver.optimizer == "ADAMW":
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.solver.base_lr * rl_factor,
            weight_decay=cfg.solver.weight_decay,
            betas=(0.9, 0.999),  # (beta1, beta2)
            eps=1e-8  # epsilon
        )
    elif cfg.solver.optimizer == "MuonWithAdamW":
        muon_scaling = getattr(cfg.solver, "muon_scaling", 0.5)
        adamw_scaling = getattr(cfg.solver, "adamw_scaling", 0.5)
        optimizer = MuonWithAdamW(
            params,
            lr=cfg.solver.base_lr * rl_factor,
            weight_decay=cfg.solver.weight_decay,
            betas=(0.9, 0.999),
            muon=muon_scaling,
            adamw=adamw_scaling
        )
    else:
        raise ValueError("Invalid Optimizer Type")
    return optimizer

def make_lr_scheduler(cfg, optimizer, logger=None, iters_per_epoch=None):
    if cfg.solver.schedule.type == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            cfg.solver.steps,
            cfg.solver.gamma,
            warmup_factor=cfg.solver.warmup_factor,
            warmup_epochs=cfg.solver.warmup_epochs,
            warmup_method=cfg.solver.warmup_method,
        )
    
    elif cfg.solver.schedule.type == "WarmupReduceLROnPlateau":
        return WarmupReduceLROnPlateau(
            optimizer,
            cfg.solver.schedule.factor,
            warmup_factor=cfg.solver.warmup_factor,
            warmup_epochs=cfg.solver.warmup_epochs,
            warmup_method=cfg.solver.warmup_method,
            patience=cfg.solver.schedule.patience,
            threshold=cfg.solver.schedule.threshold,
            cooldown=cfg.solver.schedule.cooldown,
            logger=logger,
        )
    
    elif cfg.solver.schedule.type == "WarmupCosineAnnealingLR":
        return WarmupCosineAnnealingLR(
            optimizer,
            T_max=cfg.solver.max_epoch,
            warmup_epochs=cfg.solver.warmup_epochs,
            warmup_factor=cfg.solver.warmup_factor,
            eta_min=getattr(cfg.solver.schedule, 'eta_min', 1e-7),
        )

    elif cfg.solver.schedule.type == "WarmupCosineAnnealingIterLR":
        if iters_per_epoch is None:
            raise ValueError(
                "iters_per_epoch must be provided to make_lr_scheduler() "
                "when using WarmupCosineAnnealingIterLR"
            )
        # The scheduler is stepped once per OPTIMIZER UPDATE, not per raw iteration.
        # Divide by accum_steps so warmup duration and cosine cycle track optimizer
        # steps, not raw iterations.  Without this, with accum_steps=4:
        #   warmup takes 4 real epochs instead of 1 (LR stays near base_lr*warmup_factor)
        #   cosine barely decays over the full training run.
        accum_steps = int(getattr(cfg.solver, 'accum_steps', 1))
        total_iters  = max(1, (cfg.solver.max_epoch * iters_per_epoch) // accum_steps)
        warmup_iters = max(1, int(cfg.solver.warmup_epochs * iters_per_epoch) // accum_steps)
        if logger:
            logger.info(
                f"[LR] WarmupCosineAnnealingIterLR: "
                f"total_iters={total_iters} ({cfg.solver.max_epoch} epochs × {iters_per_epoch} iters/epoch ÷ {accum_steps} accum), "
                f"warmup_iters={warmup_iters} ({cfg.solver.warmup_epochs} warmup epoch(s)), "
                f"eta_min={getattr(cfg.solver.schedule, 'eta_min', 1e-7):.1e}"
            )
        return WarmupCosineAnnealingIterLR(
            optimizer,
            total_iters=total_iters,
            warmup_iters=warmup_iters,
            warmup_factor=cfg.solver.warmup_factor,
            eta_min=getattr(cfg.solver.schedule, 'eta_min', 1e-7),
        )

    elif cfg.solver.schedule.type == "LambdaIterLR":
        if iters_per_epoch is None:
            raise ValueError(
                "iters_per_epoch must be provided to make_lr_scheduler() "
                "when using LambdaIterLR"
            )
        fn_name = getattr(cfg.solver.schedule, 'fn', 'onecycle')
        if fn_name not in _LAMBDA_FACTORIES:
            raise ValueError(
                f"Unknown LambdaIterLR fn '{fn_name}'. "
                f"Available: {list(_LAMBDA_FACTORIES.keys())}"
            )
        total_iters  = cfg.solver.max_epoch * iters_per_epoch
        warmup_iters = int(cfg.solver.warmup_epochs * iters_per_epoch)
        eta_min      = getattr(cfg.solver.schedule, 'eta_min', 1e-7)
        eta_min_factor = eta_min / cfg.solver.base_lr  # convert abs → multiplicative
        power        = float(getattr(cfg.solver.schedule, 'power', 2.0))
        factory      = _LAMBDA_FACTORIES[fn_name]
        # Build kwargs dynamically — polynomial uses `power`, others don't
        kwargs = dict(
            total_iters=total_iters,
            warmup_iters=warmup_iters,
            warmup_factor=cfg.solver.warmup_factor,
            eta_min_factor=eta_min_factor,
        )
        if fn_name == 'polynomial':
            kwargs['power'] = power
        lr_lambda = factory(**kwargs)
        if logger:
            logger.info(
                f"[LR] LambdaIterLR fn={fn_name}: "
                f"total_iters={total_iters}, warmup_iters={warmup_iters}, "
                f"eta_min_factor={eta_min_factor:.2e}"
                + (f", power={power}" if fn_name == 'polynomial' else "")
            )
        return LambdaIterLR(optimizer, lr_lambda)

    else:
        raise ValueError("Invalid Schedule Type")
