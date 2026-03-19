# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os

import torch

from sgg_benchmark.utils.model_serialization import load_state_dict
from sgg_benchmark.utils.model_zoo import cache_url
from sgg_benchmark.utils.miscellaneous import get_path

class Checkpointer(object):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
        custom_scheduler=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            try:
                from loguru import logger
            except ImportError:
                logger = logging.getLogger(__name__)
        self.logger = logger
        self.custom_scheduler = custom_scheduler

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None and not self.custom_scheduler:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None, with_optim=True, update_schedule=False, load_mapping={}, verbose=False):
        # if self.has_checkpoint():
        #     # override argument with existing checkpoint
        #     f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint, load_mapping, verbose)
        if with_optim:
            if "optimizer" in checkpoint and self.optimizer:
                self.logger.info("Loading optimizer from {}".format(f))
                self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
            if "scheduler" in checkpoint and self.scheduler:
                self.logger.info("Loading scheduler from {}".format(f))
                if update_schedule:
                    self.scheduler.last_epoch = checkpoint["iteration"]
                else:
                    self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"), weights_only=False)

    def _load_model(self, checkpoint, load_mapping, verbose=False):
        load_state_dict(self.model, checkpoint.pop("model"), load_mapping, verbose)


class DetectronCheckpointer(Checkpointer):
    """Enhanced checkpointer with support for backbone loading and checkpoint resolution.
    
    Features:
    - Loads model state with optional layer filtering (for backbone freezing)
    - Supports catalog lookup for model references
    - Handles URL caching for remote models
    - Tracks training metadata (epoch, best_metric)
    """
    def __init__(
        self,
        cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
        custom_scheduler=False,
    ):
        if save_dir is None:
            save_dir = ""
            
        if save_dir and not os.path.exists(save_dir):
            if os.path.exists(cfg.output_dir):
                save_dir = cfg.output_dir
            elif os.path.exists(os.path.join(get_path(), cfg.output_dir)):
                save_dir = os.path.join(get_path(), cfg.output_dir)

        super(DetectronCheckpointer, self).__init__(
            model, optimizer, scheduler, save_dir, save_to_disk, logger, custom_scheduler
        )
        # Store config as-is (OmegaConf DictConfig doesn't have clone method)
        self.cfg = cfg

    def save(self, name, epoch=None, best_metric=None, **kwargs):
        """Save checkpoint with optional training metadata.
        
        Args:
            name: Checkpoint name (without .pth extension)
            epoch: Current epoch number (optional)
            best_metric: Best validation metric value (optional)
            **kwargs: Additional metadata to save
        """
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None and not self.custom_scheduler:
            data["scheduler"] = self.scheduler.state_dict()
        
        # Add training metadata
        if epoch is not None:
            data["epoch"] = epoch
        if best_metric is not None:
            data["best_metric"] = best_metric
        
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load_backbone(self, checkpoint_path):
        """Load pretrained backbone and optionally freeze it.
        
        Args:
            checkpoint_path: Path to backbone checkpoint
            backbone_prefix: Prefix for backbone layers in model
            strict: Whether to strictly match state dict keys
            
        Returns:
            Loaded checkpoint data (for metadata extraction)
        """
        if not checkpoint_path:
            self.logger.info("No backbone checkpoint provided")
            return {}
        
        self.logger.info("Loading backbone from {}".format(checkpoint_path))
        try:
            self.model.backbone.load(checkpoint_path)
        except Exception as e:
            self.logger.error("Error loading backbone: {}".format(e))
            raise e
        
    def _load_file(self, f):
        """Load checkpoint file, handling URLs.
        
        Args:
            f: File path or URL
            
        Returns:
            Loaded checkpoint dictionary
        """
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f
        
        # load checkpoint
        loaded = super(DetectronCheckpointer, self)._load_file(f)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded


def clip_grad_norm(named_parameters, max_norm, logger, clip=False, verbose=False):
    """Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    max_norm = float(max_norm)

    total_norm = 0
    param_to_norm = {}
    param_to_shape = {}
    for n, p in named_parameters:
        if p.grad is not None:
            param_norm = p.grad.norm(2)
            total_norm += param_norm ** 2
            param_to_norm[n] = param_norm
            param_to_shape[n] = p.size()

    total_norm = total_norm ** (1. / 2)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1 and clip:
        for _, p in named_parameters:
            if p.grad is not None:
                p.grad.mul_(clip_coef)

    return total_norm