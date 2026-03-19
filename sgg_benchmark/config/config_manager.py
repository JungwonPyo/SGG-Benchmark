"""
Simplified Config Manager for Hydra-based SGG.

This replaces the complex Config wrapper class with simple utilities
for loading, validating, and accessing Hydra configs.

No YACS conversion - pure Hydra/OmegaConf!
"""

from pathlib import Path
from typing import Optional, Union
from omegaconf import DictConfig, OmegaConf, open_dict
import logging

from .class_inference import resolve_num_classes

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Simple config manager for Hydra configs.
    
    Usage:
        # Load config
        cfg = ConfigManager.load("configs/hydra/VG/triplet_attention.yaml")
        
        # Access values (lowercase, nested)
        backbone = cfg.model.backbone.type
        num_classes = cfg.model.roi_box_head.num_classes
        
        # Modify config (using open_dict context)
        with open_dict(cfg):
            cfg.model.weight = "path/to/checkpoint.pth"
        
        # Validate
        ConfigManager.validate(cfg)
        
        # Save
        ConfigManager.save(cfg, "output/config.yaml")
    """
    
    @staticmethod
    def load(config_path: Union[str, Path], infer_num_classes: bool = True) -> DictConfig:
        """
        Load Hydra config from YAML file.
        
        Args:
            config_path: Path to YAML config file
            infer_num_classes: If True, automatically infer num_classes from dataset files
                              if config values are <= 0. Default: True
            
        Returns:
            DictConfig with loaded configuration
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        logger.info(f"Loading config from: {config_path}")
        cfg = OmegaConf.load(config_path)
        
        # Infer num_classes from dataset if enabled
        if infer_num_classes:
            try:
                logger.info("Attempting to infer num_classes from dataset files...")
                resolved = resolve_num_classes(cfg)
                with open_dict(cfg):
                    if 'model' in resolved:
                        if 'roi_box_head' in resolved['model']:
                            cfg.model.roi_box_head.num_classes = resolved['model']['roi_box_head']['num_classes']
                        if 'roi_relation_head' in resolved['model']:
                            cfg.model.roi_relation_head.num_classes = resolved['model']['roi_relation_head']['num_classes']
            except Exception as e:
                logger.warning(f"Failed to infer num_classes: {e}. Using config values.")
        
        # Set config as read-only by default (safety)
        OmegaConf.set_readonly(cfg, True)
        
        return cfg
    
    @staticmethod
    def merge(*configs: DictConfig) -> DictConfig:
        """
        Merge multiple configs (later configs override earlier ones).
        
        Args:
            *configs: Variable number of DictConfigs to merge
            
        Returns:
            Merged DictConfig
        """
        return OmegaConf.merge(*configs)
    
    @staticmethod
    def from_cli(args: list) -> DictConfig:
        """
        Create config from command-line arguments.
        
        Args:
            args: List of "key=value" strings
            
        Returns:
            DictConfig created from CLI args
            
        Example:
            cfg = ConfigManager.from_cli(["model.weight=path.pth", "solver.base_lr=0.001"])
        """
        return OmegaConf.from_cli(args)
    
    @staticmethod
    def save(cfg: DictConfig, save_path: Union[str, Path]) -> None:
        """
        Save config to YAML file.
        
        Args:
            cfg: Configuration to save
            save_path: Where to save the config
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            OmegaConf.save(cfg, f)
        
        logger.info(f"Config saved to: {save_path}")
    
    @staticmethod
    def validate(cfg: DictConfig) -> bool:
        """
        Validate configuration for common errors.
        
        Args:
            cfg: Configuration to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required top-level keys
        required_keys = ['model', 'solver', 'datasets']
        for key in required_keys:
            if not hasattr(cfg, key):
                raise ValueError(f"Missing required config section: {key}")
        
        # Validate task mode consistency
        if hasattr(cfg.model, 'roi_relation_head'):
            use_gt_box = cfg.model.roi_relation_head.get('use_gt_box', False)
            use_gt_label = cfg.model.roi_relation_head.get('use_gt_object_label', False)
            
            if use_gt_label and not use_gt_box:
                raise ValueError(
                    "Invalid task mode: use_gt_object_label=True requires use_gt_box=True"
                )
        
        # Validate dataset paths exist (if specified)
        if hasattr(cfg, 'datasets') and hasattr(cfg.datasets, 'catalog'):
            for dataset_name, dataset_info in cfg.datasets.catalog.items():
                if hasattr(dataset_info, 'data_dir'):
                    data_dir = Path(dataset_info.data_dir)
                    if not data_dir.exists():
                        logger.warning(f"Dataset directory not found: {data_dir}")
        
        # Validate num_classes consistency
        if cfg.model.relation_on:
            rel_classes = cfg.model.roi_relation_head.num_classes
            if rel_classes <= 0:
                raise ValueError(f"relation_on=True requires num_classes > 0, got {rel_classes}")
        
        logger.info("✓ Config validation passed")
        return True
    
    @staticmethod
    def set_task_mode(cfg: DictConfig, task: str) -> DictConfig:
        """
        Set configuration for specific task mode (sgdet/sgcls/predcls).
        
        Args:
            cfg: Configuration to modify
            task: Task mode - "sgdet", "sgcls", or "predcls"
            
        Returns:
            Modified configuration
        """
        task = task.lower()
        
        if task not in ['sgdet', 'sgcls', 'predcls']:
            raise ValueError(f"Invalid task: {task}. Must be sgdet, sgcls, or predcls")
        
        with open_dict(cfg):
            if task == "sgdet":
                # Scene Graph Detection: detect objects + relations
                cfg.model.roi_relation_head.use_gt_box = False
                cfg.model.roi_relation_head.use_gt_object_label = False
            elif task == "sgcls":
                # Scene Graph Classification: use GT boxes, predict labels + relations
                cfg.model.roi_relation_head.use_gt_box = True
                cfg.model.roi_relation_head.use_gt_object_label = False
            elif task == "predcls":
                # Predicate Classification: use GT boxes + labels, predict relations only
                cfg.model.roi_relation_head.use_gt_box = True
                cfg.model.roi_relation_head.use_gt_object_label = True
        
        logger.info(f"Set task mode: {task}")
        return cfg
    
    @staticmethod
    def pretty_print(cfg: DictConfig) -> str:
        """
        Get pretty-printed config as string.
        
        Args:
            cfg: Configuration to print
            
        Returns:
            YAML string representation
        """
        return OmegaConf.to_yaml(cfg, resolve=True)
    
    @staticmethod
    def to_container(cfg: DictConfig) -> dict:
        """
        Convert DictConfig to plain Python dict.
        
        Args:
            cfg: Configuration to convert
            
        Returns:
            Plain Python dictionary
        """
        return OmegaConf.to_container(cfg, resolve=True)
    
    @staticmethod
    def get_with_default(cfg: DictConfig, key: str, default: any) -> any:
        """
        Get config value with default fallback.
        
        Args:
            cfg: Configuration to query
            key: Dot-separated key path (e.g., "model.backbone.type")
            default: Default value if key not found
            
        Returns:
            Value from config or default
            
        Example:
            freeze = ConfigManager.get_with_default(cfg, "model.backbone.freeze", True)
        """
        try:
            return OmegaConf.select(cfg, key)
        except Exception:
            return default


def load_config_from_checkpoint_dir(checkpoint_dir: Union[str, Path]) -> DictConfig:
    """
    Load config from a checkpoint directory.
    
    Searches for config.yaml, hydra_config.yaml, or .hydra/config.yaml
    
    Args:
        checkpoint_dir: Directory containing checkpoint and config
        
    Returns:
        Loaded configuration
        
    Raises:
        FileNotFoundError: If no config found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Search for config in common locations
    config_candidates = [
        checkpoint_dir / "config.yaml",
        checkpoint_dir / "hydra_config.yaml",
        checkpoint_dir / ".hydra" / "config.yaml",
        checkpoint_dir / ".hydra" / "overrides.yaml",
    ]
    
    for config_path in config_candidates:
        if config_path.exists():
            logger.info(f"Found config at: {config_path}")
            return ConfigManager.load(config_path)
    
    raise FileNotFoundError(
        f"Could not find config in {checkpoint_dir}. "
        f"Looked for: {[c.name for c in config_candidates]}"
    )


def create_config_from_defaults() -> DictConfig:
    """
    Create a default configuration.
    
    Returns:
        DictConfig with default values
    """
    from sgg_benchmark.config.structured_configs import SGGConfig
    
    # Convert dataclass to DictConfig
    default_cfg = OmegaConf.structured(SGGConfig)
    return default_cfg


# Backward compatibility helpers
def get_cfg_defaults() -> DictConfig:
    """Get default config (backward compatibility with YACS style)."""
    return create_config_from_defaults()


def clone_cfg(cfg: DictConfig) -> DictConfig:
    """Clone a config (backward compatibility with YACS style)."""
    return OmegaConf.create(OmegaConf.to_container(cfg))


def freeze_cfg(cfg: DictConfig) -> None:
    """Make config read-only (backward compatibility with YACS style)."""
    OmegaConf.set_readonly(cfg, True)


def defrost_cfg(cfg: DictConfig) -> None:
    """Make config writable (backward compatibility with YACS style)."""
    OmegaConf.set_readonly(cfg, False)
