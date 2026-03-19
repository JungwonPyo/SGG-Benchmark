"""
Hydra + OmegaConf configuration system for SGG-Benchmark.
This replaces the YACS-based configuration with a more modern and flexible approach.

Usage:
    from sgg_benchmark.config import get_cfg
    
    @hydra.main(version_base=None, config_path="configs", config_name="default")
    def main(cfg: DictConfig):
        config = get_cfg(cfg)
        # Use config...
"""

import os
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig
from hydra.core.config_store import ConfigStore


@dataclass
class InputConfig:
    """Input preprocessing configuration"""
    img_size: Tuple[int, int] = (640, 640)
    pixel_mean: tuple = (102.9801, 115.9465, 122.7717)
    pixel_std: tuple = (1.0, 1.0, 1.0)
    to_bgr255: bool = True
    flip_prob_train: float = 0.5
    padding: bool = False
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    hue: float = 0.0
    vertical_flip_prob_train: float = 0.0


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    train: tuple = field(default_factory=tuple)
    val: tuple = field(default_factory=tuple)
    test: tuple = field(default_factory=tuple)
    name: str = ""
    type: str = ""
    data_dir: str = ""
    classes: tuple = field(default_factory=tuple)
    
    # Dataset catalog - flexible dict that can be defined in YAML
    # Users can add any dataset with any structure directly in config files
    catalog: dict = field(default_factory=dict)


@dataclass
class DataLoaderConfig:
    """DataLoader configuration"""
    num_workers: int = 4
    size_divisibility: int = 0
    aspect_ratio_grouping: bool = True


@dataclass
class BackboneConfig:
    """Backbone architecture configuration"""
    type: str = "R-50-C4"
    extra_config: str = ""
    freeze_conv_body_at: int = 2
    nms_thresh: float = 0.7
    freeze: bool = False
    # For YOLO: freeze layers with index < freeze_at, fine-tune the rest (neck + head).
    # -1 means disabled (either full freeze or full fine-tune, driven by `freeze`).
    # Typical value for yolo12m: 10 (freezes stem+backbone, trains neck+FPN layers 10-22).
    freeze_at: int = -1


@dataclass
class FPNConfig:
    """Feature Pyramid Network configuration"""
    use_gn: bool = False
    use_relu: bool = False


@dataclass
class GroupNormConfig:
    """Group Normalization configuration"""
    dim_per_gp: int = -1
    num_groups: int = 32
    epsilon: float = 1e-5


@dataclass
class YOLOConfig:
    """YOLO backbone configuration"""
    weights: str = ""
    size: str = "yolov8l"
    img_size: int = 640
    out_channels: tuple = (192, 384, 576)


@dataclass
class RPNConfig:
    """Region Proposal Network configuration"""
    use_fpn: bool = False
    rpn_mid_channel: int = 512
    anchor_sizes: tuple = (32, 64, 128, 256, 512)
    anchor_stride: tuple = (16,)
    aspect_ratios: tuple = (0.5, 1.0, 2.0)
    straddle_thresh: int = 0
    fg_iou_threshold: float = 0.7
    bg_iou_threshold: float = 0.3
    batch_size_per_image: int = 256
    positive_fraction: float = 0.5
    pre_nms_top_n_train: int = 12000
    pre_nms_top_n_test: int = 6000
    post_nms_top_n_train: int = 2000
    post_nms_top_n_test: int = 1000
    min_size: int = 0
    fpn_post_nms_top_n_train: int = 2000
    fpn_post_nms_top_n_test: int = 2000
    fpn_post_nms_per_batch: bool = True
    rpn_head: str = "SingleConvRPNHead"


@dataclass
class ROIHeadsConfig:
    """ROI Heads configuration"""
    fg_iou_threshold: float = 0.5
    bg_iou_threshold: float = 0.3
    bbox_reg_weights: tuple = (10.0, 10.0, 5.0, 5.0)
    batch_size_per_image: int = 256
    positive_fraction: float = 0.25
    score_thresh: float = 0.01
    nms: float = 0.3
    post_nms_per_cls_topn: int = 300
    nms_filter_duplicates: bool = False
    detections_per_img: int = 100


@dataclass
class ROIBoxHeadConfig:
    """ROI Box Head configuration"""
    feature_extractor: str = "ResNet50Conv5ROIFeatureExtractor"
    predictor: str = "FastRCNNPredictor"
    pooler_resolution: int = 14
    pooler_sampling_ratio: int = 0
    pooler_scales: tuple = (1.0 / 16,)
    mlp_head_dim: int = 2048
    use_gn: bool = False
    dilation: int = 1
    conv_head_dim: int = 256
    num_stacked_convs: int = 4
    num_classes: int = 0  # Will be auto-inferred from dataset if not overridden
    patch_size: int = 32
    # DAMPBoxFeatureExtractor: multi-scale integer gather (ablation winner A8).
    #   feat_idx_multiscale=True  — gather P3+P4+P5 centre tokens, fuse via LN+Linear (default).
    #   feat_idx_multiscale=False — fallback: single peak token on the assigned FPN level.
    feat_idx_multiscale: bool = True
    #   feat_idx_neighbors — Gaussian-weighted (2r+1)² neighbourhood radius applied at EACH
    #     FPN level in the multi-scale gather.  1 = 3×3 = 9 tokens per level → 27 gathers
    #     total (ablation winner: +1.1pp F1@100 vs r=0).  No extra parameters.
    feat_idx_neighbors: int = 1


@dataclass
class ROIAttributeHeadConfig:
    """ROI Attribute Head configuration"""
    feature_extractor: str = "FPN2MLPFeatureExtractor"
    predictor: str = "FPNPredictor"
    share_box_feature_extractor: bool = True
    use_binary_loss: bool = True
    attribute_loss_weight: float = 0.1
    num_attributes: int = 201
    max_attributes: int = 10
    attribute_bgfg_sample: bool = True
    attribute_bgfg_ratio: int = 3
    pos_weight: float = 5.0


@dataclass
class ROIMaskHeadConfig:
    """ROI Mask Head configuration"""
    feature_extractor: str = "ResNet50Conv5ROIFeatureExtractor"
    predictor: str = "MaskRCNNC4Predictor"
    pooler_resolution: int = 14
    pooler_sampling_ratio: int = 0
    pooler_scales: tuple = (1.0 / 16,)
    mlp_head_dim: int = 1024
    conv_layers: tuple = (256, 256, 256, 256)
    resolution: int = 14
    share_box_feature_extractor: bool = True
    postprocess_masks: bool = False
    postprocess_masks_threshold: float = 0.5
    dilation: int = 1
    use_gn: bool = False


@dataclass
class TransformerConfig:
    """Transformer configuration for relation head"""
    dropout_rate: float = 0.1
    obj_layer: int = 4
    rel_layer: int = 2
    num_head: int = 8
    inner_dim: int = 2048
    key_dim: int = 64
    val_dim: int = 64


@dataclass
class SquatModuleConfig:
    """SQUAT module configuration"""
    pre_norm: bool = False
    num_decoder: int = 3
    rho: float = 0.35
    beta: float = 0.7
    pretrain_mask: bool = False
    pretrain_mask_epoch: int = 1


@dataclass
class CausalConfig:
    """Causal analysis configuration"""
    effect_analysis: bool = False
    fusion_type: str = 'sum'
    context_layer: str = 'motifs'
    separate_spatial: bool = False
    effect_type: str = 'none'
    spatial_for_vision: bool = False


@dataclass
class RelationLossConfig:
    """Configuration for relationship classification loss"""
    loss_type: str = "CrossEntropyLoss"
    beta: float = 0.999
    gamma: float = 2.0          # Focal loss exponent (BalancedLogitAdjustedLoss)
    alpha: float = 0.25         # BG weight; FG weight = (1-alpha)*fg_boost
    fg_boost: float = 2.0       # Extra multiplier on FG loss
    fg_weight: float = 1.0
    label_smoothing_epsilon: float = 0.01
    logit_adjustment_tau: float = 0.3
    bg_discount: float = 2.0    # NOTE: schema default 2.0 is for SemanticCompatibilityLoss.
                                # For BalancedLogitAdjustedLoss set explicitly in yaml to (0,1].
    ccl_weight: float = 0.1
    decisive_margin: float = 2.0
    poly_epsilon: float = 0.0
    label_smoothing: float = 0.1           # label smoothing for auxiliary sampler CE
    sampler_aux_loss_weight: float = 0.1   # weight of direct fg CE on raw sampler output; 0.0 = disabled
    attn_entropy_weight: float = 0.01      # coefficient for −H(per_anchor_attention) diversity bonus; 0.0 = disabled
    offset_reg_weight: float = 0.005       # L2 reg on learned offsets; keeps points near grid; 0.0 = disabled
    containment_loss_weight: float = 0.02  # hinge: keeps sub/obj anchors inside entity boxes; 0.0 = disabled

@dataclass
class ROIRelationHeadConfig:
    """ROI Relation Head configuration"""
    predictor: str = "MotifPredictor"
    feature_extractor: str = "RelationFeatureExtractor"
    use_union_features: bool = True
    use_spatial_features: bool = True
    # When False, union RoI features are NOT computed at inference (saves ~47ms/img
    # on RTX 4000 Ada from profiling). The model is trained WITH union but tested
    # without — the freq-bias + sub/obj visual path is robust enough.
    use_union_features_inference: bool = True
    # Probability of randomly SKIPPING the union gate during a training forward
    # pass.  When use_union_features_inference=False this closes the train/test
    # distribution gap: the base composition path is forced to learn to work
    # without union correction, so at inference the drop is minimal.
    # 0.0 = always use union (run1/run5 behaviour)
    # 0.5 = skip union for ~half the batches (run6 target)
    union_dropout: float = 0.0
    # Maximum relation pairs to process at inference per batch.
    # Uses frequency-bias scores to keep only the most plausible (sub_cls, obj_cls)
    # pairs before the expensive compose_ffn / union_attn / proj_head ops.
    # 0 = no limit (all N*(N-1)/2 pairs); 300 is a good starting point.
    max_pairs_inference: int = 0
    textual_features_only: bool = False
    visual_features_only: bool = False
    logit_adjustment: bool = False
    logit_adjustment_tau: float = 0.3
    pooling_all_levels: bool = True
    batch_size_per_image: int = 64
    positive_fraction: float = 0.25
    use_gt_box: bool = True
    use_gt_object_label: bool = False
    embed_dim: int = 200
    context_dropout_rate: float = 0.2
    context_hidden_dim: int = 512
    context_pooling_dim: int = 4096
    context_obj_layer: int = 1
    context_rel_layer: int = 1
    mlp_head_dim: int = 2048
    loss: RelationLossConfig = field(default_factory=RelationLossConfig)
    num_classes: int = 0  # Will be auto-inferred from dataset if not overridden
    
    decoder_depth: int = 1
    transformer_depth: int = 1
    num_rel_layers: int = 2       # Stacked intra-relation self-attention layers
    use_scene_context: bool = True  # Pool P5→4x4 + AIFI global scene tokens in obj context
    use_geo_bias: bool = True      # Apply GeomRoPE geometry bias in sub/obj cross-attention
    use_cls_emb: bool = True       # Enrich sub/obj features with class embeddings
    use_geo_enc: bool = True       # Add spatial geometry encoding to query
    max_pairs_per_img: int = 512
    num_queries: int = 64
    use_cross_attention: bool = True
    attn_type: str = "standard"
    geometric_loss_weight: float = 0.0
    num_sample_points: int = 6
    num_sample_heads: int = 6  # multi-head deformable attention; must divide num_sample_points
    feature_strategy: str = "multi_scale"
    use_rmsnorm: bool = True
    use_swiglu: bool = True
    # Path to predicate prototype embeddings (e.g. mpnet_rel.pt).
    # Empty string = fall back to standard nn.Linear classifier.
    clip_rel_path: str = ""

    # REACT regularizer loss weights (optional). Keys: l21_loss, dist_loss2, loss_dis
    react_loss_weights: dict = field(default_factory=lambda: {"l21_loss": 0.01, "dist_loss2": 0.1, "loss_dis": 1.0})

    # Nested configs
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    squat_module: SquatModuleConfig = field(default_factory=SquatModuleConfig)
    causal: CausalConfig = field(default_factory=CausalConfig)
    
    label_smoothing_loss: bool = False
    use_frequency_bias: bool = False
    require_box_overlap: bool = True
    num_sample_per_gt_rel: int = 4
    add_gtbox_to_proposal_in_train: bool = False
    classifier: str = "linear"
    predict_use_vision: bool = False
    
    # Causal/Semantic options not directly in loss
    use_bg_discounting: bool = False
    bg_discounting_threshold: float = 0.1


@dataclass
class ResNetsConfig:
    """ResNet configuration"""
    num_groups: int = 1
    width_per_group: int = 64
    stride_in_1x1: bool = True
    trans_func: str = "BottleneckWithFixedBatchNorm"
    stem_func: str = "StemWithFixedBatchNorm"
    res5_dilation: int = 1
    backbone_out_channels: int = 1024
    res2_out_channels: int = 256
    stem_out_channels: int = 64


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    flip_aug: bool = False
    rpn_only: bool = False
    mask_on: bool = False
    attribute_on: bool = False
    relation_on: bool = True
    device: str = "cuda"
    meta_architecture: str = "GeneralizedYOLO"
    cls_agnostic_bbox_reg: bool = False
    weight: str = ""
    pretrained_detector_ckpt: str = ""
    text_embedding: str = "glove.6B"
    box_head: bool = False
    
    # Nested configs
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    fpn: FPNConfig = field(default_factory=FPNConfig)
    group_norm: GroupNormConfig = field(default_factory=GroupNormConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    rpn: RPNConfig = field(default_factory=RPNConfig)
    roi_heads: ROIHeadsConfig = field(default_factory=ROIHeadsConfig)
    roi_box_head: ROIBoxHeadConfig = field(default_factory=ROIBoxHeadConfig)
    roi_attribute_head: ROIAttributeHeadConfig = field(default_factory=ROIAttributeHeadConfig)
    roi_mask_head: ROIMaskHeadConfig = field(default_factory=ROIMaskHeadConfig)
    roi_relation_head: ROIRelationHeadConfig = field(default_factory=ROIRelationHeadConfig)
    resnets: ResNetsConfig = field(default_factory=ResNetsConfig)


@dataclass
class ScheduleConfig:
    """Learning rate schedule configuration"""
    type: str = "WarmupMultiStepLR"
    patience: int = 2
    threshold: float = 1e-4
    cooldown: int = 1
    factor: float = 0.5
    max_decay_step: int = 7
    eta_min: float = 1e-7  # floor LR for WarmupCosineAnnealingLR
    plateau_epochs: int = 5  # flat zone between warmup and cosine decay (WarmupPlateauCosineAnnealingLR)


@dataclass
class SolverConfig:
    """Solver/Optimizer configuration"""
    max_iter: int = 0 # Set to 0 to use max_epoch instead, for debugging purposes
    max_epoch: int = 100
    base_lr: float = 0.001
    bias_lr_factor: int = 2
    momentum: float = 0.9
    weight_decay: float = 0.0001
    weight_decay_bias: float = 0.0
    clip_norm: float = 5.0
    gamma: float = 0.5
    steps: tuple = (30000,)
    warmup_factor: float = 0.1
    warmup_epochs: int = 500
    warmup_method: str = "linear"
    checkpoint_period: int = 2500
    grad_norm_clip: float = 5.0
    print_grad_freq: int = 5000
    to_val: bool = True
    pre_val: bool = True
    val_period: int = 2500
    update_schedule_during_load: bool = False
    ims_per_batch: int = 8
    optimizer: str = "SGD"
    slow_ratio: float = 5.0
    deform_offset_slow_ratio: float = 1.0  # offset_proj + attn_weight_proj LR divisor (1.0 = disabled)
    muon_scaling: float = 100.0
    adamw_scaling: float = 1.0
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    # Gradient accumulation (effective batch = ims_per_batch * accum_steps)
    accum_steps: int = 1


@dataclass
class BBoxAugConfig:
    """Bounding box augmentation configuration"""
    enabled: bool = False
    h_flip: bool = False
    scales: tuple = field(default_factory=tuple)
    max_size: int = 4000
    scale_h_flip: bool = False


@dataclass
class RelationTestConfig:
    """Relation testing configuration"""
    multiple_preds: bool = False
    iou_threshold: float = 0.5
    require_overlap: bool = True
    later_nms_prediction_thres: float = 0.3
    sync_gather: bool = False


@dataclass
class TestConfig:
    """Test configuration"""
    expected_results: list = field(default_factory=list)
    expected_results_sigma_tol: int = 4
    ims_per_batch: int = 8
    detections_per_img: int = 100
    informative: bool = False
    bbox_aug: BBoxAugConfig = field(default_factory=BBoxAugConfig)
    save_proposals: bool = False
    relation: RelationTestConfig = field(default_factory=RelationTestConfig)
    allow_load_from_cache: bool = True
    top_k: int = 100
    custum_eval: bool = False
    custum_path: str = ""


@dataclass
class GCLSettingConfig:
    """GCL (Group Collaborative Learning) configuration"""
    group_split_mode: str = 'divide4'
    knowledge_transfer_mode: str = 'KL_logit_TopDown'
    no_relation_restrain: bool = False
    zero_label_padding_mode: bool = False
    knowledge_loss_coefficient: float = 1.0


@dataclass
class GlobalSettingConfig:
    """Global/GCL configuration"""
    basic_encoder: str = 'Cross-Attention'
    gcl_setting: GCLSettingConfig = field(default_factory=GCLSettingConfig)


@dataclass
class SGGBenchmarkConfig:
    """Main configuration for SGG-Benchmark"""
    seed: int = 42
    metric_to_track: str = "mR"
    dtype: str = "float32"
    output_dir: str = "."
    glove_dir: str = "."
    verbose: str = "INFO"
    paths_catalog: str = ""
    paths_data: str = ""
    
    # Nested configs
    input: InputConfig = field(default_factory=InputConfig)
    datasets: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    test: TestConfig = field(default_factory=TestConfig)
    global_setting: GlobalSettingConfig = field(default_factory=GlobalSettingConfig)


# Register configurations with Hydra
cs = ConfigStore.instance()
cs.store(name="sgg_benchmark_config", node=SGGBenchmarkConfig)


def get_cfg(hydra_cfg: Optional[DictConfig] = None) -> DictConfig:
    """
    Get configuration from Hydra config or create default.
    This function implements the "defaults-first" logic:
    1. Load default parameters from the SGGBenchmarkConfig dataclass.
    2. Update with parameters provided in the hydra_cfg (from YAML/overrides).
    3. Keep all default values for keys not specified in hydra_cfg.
    
    Args:
        hydra_cfg: Optional Hydra configuration from compose() or @hydra.main
        
    Returns:
        OmegaConf DictConfig with full configuration (defaults + overrides)
    """
    # Load defaults first
    base_config = OmegaConf.structured(SGGBenchmarkConfig)
    
    if hydra_cfg is None:
        return base_config
    
    # Pre-process hydra_cfg to handle deprecated string-based loss config
    if "model" in hydra_cfg and "roi_relation_head" in hydra_cfg.model:
        head = hydra_cfg.model.roi_relation_head
        if "loss" in head and isinstance(head.loss, str):
            from omegaconf import open_dict
            with open_dict(hydra_cfg):
                loss_name = head.loss
                hydra_cfg.model.roi_relation_head.loss = {"loss_type": loss_name}

    # Merge with hydra_cfg to apply overrides
    # Note: If hydra_cfg contains keys NOT in SGGBenchmarkConfig, this will raise an error.
    # This is intended behavior to ensure config consistency.
    merged = OmegaConf.merge(base_config, hydra_cfg)
    
    return merged


def convert_to_dict(cfg: DictConfig) -> Dict[str, Any]:
    """Convert OmegaConf DictConfig to regular Python dict with interpolation resolved"""
    return OmegaConf.to_container(cfg, resolve=True)


def load_config_from_file(config_path: str) -> DictConfig:
    """
    Load configuration from YAML file and merge with defaults.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        DictConfig with loaded configuration and defaults
    """
    # Load defaults first
    base_cfg = OmegaConf.structured(SGGBenchmarkConfig)
    
    # Load YAML file
    yaml_cfg = OmegaConf.load(config_path)
    
    # Merge YAML into defaults
    merged = OmegaConf.merge(base_cfg, yaml_cfg)
    return merged


def save_config(cfg: DictConfig, output_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        cfg: Configuration to save
        output_path: Path to save YAML file
    """
    OmegaConf.save(cfg, output_path)


def update_config_from_list(cfg: DictConfig, opts: list) -> DictConfig:
    """
    Update configuration from command line options list.
    Compatible with the old YACS style: ['KEY1', 'VALUE1', 'KEY2', 'VALUE2']
    
    Args:
        cfg: Base configuration
        opts: List of key-value pairs
        
    Returns:
        Updated configuration
    """
    if opts is None or len(opts) == 0:
        return cfg
    
    assert len(opts) % 2 == 0, "Override options must be key-value pairs"
    
    updates = {}
    for i in range(0, len(opts), 2):
        key = opts[i]
        value = opts[i + 1]
        
        # Convert key from YACS format (MODEL.RELATION_ON) to nested dict
        keys = key.lower().split('.')
        current = updates
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Try to infer type from value
        try:
            # Try to parse as literal
            import ast
            current[keys[-1]] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Keep as string
            current[keys[-1]] = value
    
    # Merge updates into config
    update_cfg = OmegaConf.create(updates)
    merged = OmegaConf.merge(cfg, update_cfg)
    
    return merged


# Backward compatibility: Create a YACS-like interface
class Config:
    """
    Backward compatible config class that mimics YACS behavior.
    Wraps OmegaConf DictConfig with YACS-like API.
    """
    def __init__(self, cfg: Optional[DictConfig] = None):
        if cfg is None:
            cfg = get_cfg()
        self._cfg = cfg
    
    def clone(self):
        """Clone the config"""
        return Config(OmegaConf.create(OmegaConf.to_container(self._cfg)))
    
    def merge_from_file(self, cfg_filename: str):
        """Merge config from file"""
        file_cfg = OmegaConf.load(cfg_filename)
        self._cfg = OmegaConf.merge(self._cfg, file_cfg)
    
    def merge_from_list(self, cfg_list: list):
        """Merge config from list of key-value pairs"""
        self._cfg = update_config_from_list(self._cfg, cfg_list)
    
    def freeze(self):
        """Make config read-only"""
        OmegaConf.set_readonly(self._cfg, True)
    
    def defrost(self):
        """Make config writable"""
        OmegaConf.set_readonly(self._cfg, False)
    
    def __getattr__(self, name):
        """Access config attributes with case-insensitive access"""
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        
        # Try exact name first
        try:
            return getattr(self._cfg, name)
        except (AttributeError, KeyError):
            pass
        
        # Try lowercase version (YACS uses uppercase, Hydra uses lowercase)
        name_lower = name.lower()
        try:
            attr = getattr(self._cfg, name_lower)
            # If it's a dict/DictConfig, wrap it recursively for nested access
            if isinstance(attr, (dict, DictConfig)):
                return Config(attr)
            return attr
        except (AttributeError, KeyError):
            pass
        
        # Try uppercase version
        name_upper = name.upper()
        try:
            attr = getattr(self._cfg, name_upper)
            if isinstance(attr, (dict, DictConfig)):
                return Config(attr)
            return attr
        except (AttributeError, KeyError):
            pass
        
        # If nothing works, raise AttributeError
        raise AttributeError(f"Config has no attribute '{name}' (tried: {name}, {name_lower}, {name_upper})")
    
    def __setattr__(self, name, value):
        """Set config attributes"""
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            setattr(self._cfg, name, value)

    def get(self, name, default=None):
        """Access config attributes with case-insensitive access and default value"""
        try:
            return self.__getattr__(name)
        except AttributeError:
            return default
    
    def dump(self, **kwargs):
        """Dump config as YAML string"""
        return OmegaConf.to_yaml(self._cfg, **kwargs)
    
    @property
    def cfg(self):
        """Get underlying OmegaConf config"""
        return self._cfg
