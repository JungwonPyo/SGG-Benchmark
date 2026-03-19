# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# Hydra + OmegaConf config system (recommended)
from .hydra_config import (
    get_cfg,
    Config,
    SGGBenchmarkConfig,
    load_config_from_file,
    save_config,
    update_config_from_list,
    convert_to_dict,
)

# Backward compatibility alias
cfg = None  # Will be created at runtime via get_cfg()

__all__ = [
    'cfg',  # For backward compatibility (use get_cfg() instead)
    'get_cfg',  # New Hydra
    'Config',  # Backward compatible wrapper
    'SGGBenchmarkConfig',
    'load_config_from_file',
    'save_config',
    'update_config_from_list',
    'convert_to_dict',
]

