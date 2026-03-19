"""
Structured Hydra Configs using dataclasses.

This replaces the YACS CfgNode system with native Hydra structured configs.
Benefits:
- Type hints and validation
- Better IDE autocomplete
- Runtime type checking
- No YACS dependency
- Cleaner config access (lowercase instead of UPPERCASE)

Usage:
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    
    # Register configs with Hydra
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=SGGConfig)
    
    # Load and use
    cfg = compose(config_name="my_experiment")
    model = build_detection_model(cfg)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Any

# ============================================================================
# Input & Preprocessing
# ============================================================================

@dataclass
class InputConfig:
    """Image input and preprocessing configuration."""
    img_size: Tuple[int, int] = field(default_factory=lambda: (640, 640))
    pixel_mean: List[float] = field(default_factory=lambda: [102.9801, 115.9465, 122.7717])
    pixel_std: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    to_bgr255: bool = True
    flip_prob_train: float = 0.5
    vertical_flip_prob_train: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    hue: float = 0.0
    padding: bool = True


# ============================================================================
# Backbone Configurations
# ============================================================================

@dataclass
class BackboneConfig:
    """Backbone network configuration."""
    type: str = "yolo"  # "yolo", "resnet", "fpn"
    conv_body: str = "yolo12m"  # Model architecture name
    freeze: bool = True
    freeze_at: int = 2  # For ResNet: freeze layers up to this stage
    out_channels: List[int] = field(default_factory=lambda: [256, 512, 512])
    extra_config: str = ""  # Additional backbone-specific config
    nms_thresh: float = 0.001


@dataclass
class YOLOConfig:
    """YOLO-specific configuration."""
    size: str = "yolo12m"  # yolo8m, yolo11m, yolo12m, etc.
    img_size: int = 640
    weights: str = ""  # Path to pretrained weights
    out_channels: List[int] = field(default_factory=lambda: [256, 512, 512])


@dataclass
class RPNConfig:
    """Region Proposal Network configuration (for Faster R-CNN)."""
    use_fpn: bool = False
    anchor_sizes: Tuple[int, ...] = (32, 64, 128, 256, 512)
    anchor_stride: Tuple[int, ...] = (16,)
    aspect_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0)
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
    nms_thresh: float = 0.7
    fpn_post_nms_top_n_train: int = 2000
    fpn_post_nms_top_n_test: int = 2000
    fpn_post_nms_per_batch: bool = True
    rpn_head: str = "SingleConvRPNHead"


# ============================================================================
# ROI Heads Configurations
# ============================================================================

@dataclass
class ROIHeadsConfig:
    """Common ROI heads configuration."""
    fg_iou_threshold: float = 0.5
    bg_iou_threshold: float = 0.5
    batch_size_per_image: int = 512
    positive_fraction: float = 0.25
    bbox_reg_weights: Tuple[float, ...] = (10.0, 10.0, 5.0, 5.0)
    score_thresh: float = 0.05
    nms: float = 0.5
    detections_per_img: int = 100
    post_nms_per_cls_topn: int = 300
    nms_filter_duplicates: bool = True


@dataclass
class ROIBoxHeadConfig:
    """ROI Box Head (object detection) configuration."""
    feature_extractor: str = "YOLOV8FeatureExtractor2"
    predictor: str = "FastRCNNPredictor"
    pooler_resolution: int = 7
    pooler_sampling_ratio: int = 2
    pooler_scales: Tuple[float, ...] = (0.125, 0.0625, 0.03125)
    num_classes: int = 151  # 151 for VG, 134 for PSG
    mlp_head_dim: int = 1024
    use_gn: bool = False
    dilation: int = 1
    conv_head_dim: int = 256
    num_stacked_convs: int = 4


@dataclass
class TransformerConfig:
    """Transformer architecture configuration."""
    dropout_rate: float = 0.1
    obj_layer: int = 4
    rel_layer: int = 2
    num_head: int = 8
    key_dim: int = 64
    val_dim: int = 64
    inner_dim: int = 2048


@dataclass
class CausalConfig:
    """Causal analysis configuration (for REACT and similar methods)."""
    effect_analysis: bool = False
    fusion_type: str = "gate"  # "sum" or "gate"
    context_layer: str = "motifs"  # "motifs", "vctree", "vtranse"
    separate_spatial: bool = False
    effect_type: str = "none"  # "TDE", "NIE", "TE", "none"
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
    decisive_margin: float = 0.0
    poly_epsilon: float = 0.0
    label_smoothing: float = 0.1           # label smoothing for auxiliary sampler CE
    sampler_aux_loss_weight: float = 0.1   # weight of direct fg CE on raw sampler output; 0.0 = disabled
    attn_entropy_weight: float = 0.01      # coefficient for −H(per_anchor_attention) diversity bonus; 0.0 = disabled
    offset_reg_weight: float = 0.005       # L2 reg on learned offsets; keeps points near grid; 0.0 = disabled
    containment_loss_weight: float = 0.02  # hinge: keeps sub/obj anchors inside entity boxes; 0.0 = disabled

@dataclass
class ROIRelationHeadConfig:
    """ROI Relation Head (scene graph generation) configuration."""
    # Core settings
    feature_extractor: str = "RelationFeatureExtractor"
    predictor: str = "TransformerPredictor"
    num_classes: int = 51  # Number of predicate classes
    
    # Features
    use_union_features: bool = True
    use_spatial_features: bool = True
    pooling_all_levels: bool = True
    
    # Feature dimensions
    context_pooling_dim: int = 2048
    context_hidden_dim: int = 512
    context_dropout_rate: float = 0.2
    context_obj_layer: int = 1
    context_rel_layer: int = 1
    embed_dim: int = 200
    mlp_head_dim: int = 1024
    
    # Training settings
    batch_size_per_image: int = 256
    positive_fraction: float = 0.25
    use_gt_box: bool = False
    use_gt_object_label: bool = False
    
    # Loss
    loss: RelationLossConfig = field(default_factory=RelationLossConfig)
    label_smoothing: float = 0.0
    geometric_loss_weight: float = 0.0
    num_sample_points: int = 6
    
    # Sampling
    require_box_overlap: bool = True
    num_sample_per_gt_rel: int = 4
    add_gtbox_to_proposal_in_train: bool = True
    
    # Transformer settings (if using TransformerPredictor)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    
    # Causal settings (if using REACT)
    causal: CausalConfig = field(default_factory=CausalConfig)
    
    # Advanced
    use_frequency_bias: bool = False
    predict_use_vision: bool = True
    textual_features_only: bool = False
    visual_features_only: bool = False
    classifier: str = "linear"

    # DETR-like decoder settings
    decoder_depth: int = 1
    transformer_depth: int = 2
    num_rel_layers: int = 2       # Stacked intra-relation self-attention layers
    use_scene_context: bool = True  # Pool P5→4x4 + AIFI global scene tokens in obj context
    use_cls_emb: bool = True       # Enrich sub/obj features with class embeddings
    use_geo_enc: bool = True       # Add spatial geometry encoding to query
    max_pairs_per_img: int = 512
    num_queries: int = 64
    use_cross_attention: bool = True
    attn_type: str = "standard"
    geometric_loss_weight: float = 0.0
    num_sample_points: int = 6
    feature_strategy: str = "multi_scale"
    use_rmsnorm: bool = True
    use_swiglu: bool = True
    # Path to predicate prototype embeddings (e.g. mpnet_rel.pt).
    # Empty string = fall back to standard nn.Linear classifier.
    clip_rel_path: str = ""

    # Logit adjustment
    logit_adjustment: bool = False
    logit_adjustment_tau: float = 0.3
    bg_discount: float = 2.0
    ccl_weight: float = 0.1
    decisive_margin: float = 2.0

@dataclass
class ROIAttributeHeadConfig:
    """ROI Attribute Head configuration."""
    feature_extractor: str = "FPN2MLPFeatureExtractor"
    predictor: str = "FPNPredictor"
    use_binary_loss: bool = True
    attribute_loss_weight: float = 0.1
    num_attributes: int = 201
    max_attributes: int = 10
    share_box_feature_extractor: bool = True
    attribute_bgfg_sample: bool = True
    attribute_bgfg_ratio: int = 3
    pos_weight: float = 5.0


# ============================================================================
# Model Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Top-level model configuration."""
    meta_architecture: str = "GeneralizedYOLO"
    weight: str = ""  # Path to checkpoint
    pretrained_detector_ckpt: str = ""
    device: str = "cuda"
    
    # Model components
    rpn_only: bool = False
    relation_on: bool = True
    mask_on: bool = False
    attribute_on: bool = False
    box_head: bool = False
    
    # Model features
    flip_aug: bool = False
    cls_agnostic_bbox_reg: bool = False
    text_embedding: str = "glove.6B"
    
    # Sub-configs
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    rpn: RPNConfig = field(default_factory=RPNConfig)
    roi_heads: ROIHeadsConfig = field(default_factory=ROIHeadsConfig)
    roi_box_head: ROIBoxHeadConfig = field(default_factory=ROIBoxHeadConfig)
    roi_relation_head: ROIRelationHeadConfig = field(default_factory=ROIRelationHeadConfig)
    roi_attribute_head: ROIAttributeHeadConfig = field(default_factory=ROIAttributeHeadConfig)


# ============================================================================
# Dataset Configuration
# ============================================================================

@dataclass
class DatasetConfig:
    """Dataset configuration."""
    train: List[str] = field(default_factory=lambda: ["custom_dataset_train"])
    val: List[str] = field(default_factory=lambda: ["custom_dataset_val"])
    test: List[str] = field(default_factory=lambda: ["custom_dataset_test"])
    name: str = "custom_dataset"
    type: str = ""
    data_dir: str = ""


@dataclass
class DataLoaderConfig:
    """DataLoader configuration."""
    num_workers: int = 4
    size_divisibility: int = 32
    aspect_ratio_grouping: bool = True


# ============================================================================
# Solver/Optimizer Configuration
# ============================================================================

@dataclass
class ScheduleConfig:
    """Learning rate schedule configuration."""
    type: str = "WarmupMultiStepLR"  # "WarmupMultiStepLR", "WarmupReduceLROnPlateau", "WarmupCosineAnnealingLR", "WarmupPlateauCosineAnnealingLR"
    patience: int = 2
    threshold: float = 0.001
    cooldown: int = 0
    factor: float = 0.1
    max_decay_step: int = 3
    eta_min: float = 1e-7  # floor LR for WarmupCosineAnnealingLR
    plateau_epochs: int = 5  # flat zone between warmup and cosine decay (WarmupPlateauCosineAnnealingLR)


@dataclass
class SolverConfig:
    """Optimizer and training configuration."""
    # Optimizer
    optimizer: str = "SGD"  # "SGD", "Adam", "AdamW"
    base_lr: float = 0.01
    bias_lr_factor: float = 1.0
    momentum: float = 0.9
    weight_decay: float = 0.0001
    weight_decay_bias: float = 0.0
    slow_ratio: float = 5.0
    deform_offset_slow_ratio: float = 1.0  # offset_proj + attn_weight_proj LR divisor (1.0 = disabled)
    muon_scaling: float = 0.5
    adamw_scaling: float = 0.5
    
    # Learning rate schedule
    gamma: float = 0.1
    steps: Tuple[int, ...] = (10000, 16000)
    warmup_factor: float = 0.1
    warmup_epochs: int = 500
    warmup_method: str = "linear"
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    
    # Training iterations
    max_iter: int = 0 # Set to 0 to use max_epoch instead, for debugging purposes
    max_epoch: int = 20
    ims_per_batch: int = 8
    
    # Gradient clipping
    clip_norm: float = 5.0
    grad_norm_clip: float = 5.0

    # Gradient accumulation (effective batch = ims_per_batch * accum_steps)
    accum_steps: int = 1
    
    # Checkpointing
    checkpoint_period: int = 2000
    to_val: bool = True
    pre_val: bool = False
    val_period: int = 2000
    
    # Logging
    print_grad_freq: int = 4000
    
    # Advanced
    update_schedule_during_load: bool = False


# ============================================================================
# Test/Evaluation Configuration
# ============================================================================

@dataclass
class BBoxAugConfig:
    """Bounding box test-time augmentation."""
    enabled: bool = False
    h_flip: bool = False
    scales: List[int] = field(default_factory=list)
    max_size: int = 4000
    scale_h_flip: bool = False


@dataclass
class RelationTestConfig:
    """Relation-specific test configuration."""
    multiple_preds: bool = False
    iou_threshold: float = 0.5
    require_overlap: bool = True
    later_nms_prediction_thres: float = 0.5
    sync_gather: bool = True


@dataclass
class TestConfig:
    """Test/evaluation configuration."""
    ims_per_batch: int = 1
    detections_per_img: int = 100
    
    # Evaluation settings
    expected_results: List[Any] = field(default_factory=list)
    expected_results_sigma_tol: int = 4
    
    # Augmentation
    bbox_aug: BBoxAugConfig = field(default_factory=BBoxAugConfig)
    
    # Relation-specific
    relation: RelationTestConfig = field(default_factory=RelationTestConfig)
    
    allow_load_from_cache: bool = False
    informative: bool = False
    save_proposals: bool = False
    top_k: int = 100


# ============================================================================
# Main Configuration
# ============================================================================

@dataclass
class SGGConfig:
    """
    Main Scene Graph Generation Configuration.
    
    This is the root config that contains all sub-configs.
    Register this with Hydra's ConfigStore for structured configs.
    
    Example:
        from hydra.core.config_store import ConfigStore
        cs = ConfigStore.instance()
        cs.store(name="base_sgg_config", node=SGGConfig)
    """
    # General settings
    seed: int = 42
    output_dir: str = "./checkpoints"
    glove_dir: str = "./glove"
    verbose: str = "INFO"
    dtype: str = "float32"  # "float32" or "float16"
    metric_to_track: str = "mR"  # Metric for model selection
    
    # Sub-configurations
    input: InputConfig = field(default_factory=InputConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    datasets: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    test: TestConfig = field(default_factory=TestConfig)


# ============================================================================
# Helper Functions
# ============================================================================

def get_default_config() -> SGGConfig:
    """Get default configuration as a structured dataclass."""
    return SGGConfig()


def validate_config(cfg: SGGConfig) -> bool:
    """
    Validate configuration for common errors.
    
    Args:
        cfg: Configuration to validate
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    # Validate task consistency
    if cfg.model.roi_relation_head.use_gt_box and not cfg.model.roi_relation_head.use_gt_object_label:
        # SGCls mode
        pass
    elif cfg.model.roi_relation_head.use_gt_box and cfg.model.roi_relation_head.use_gt_object_label:
        # PredCls mode
        pass
    elif not cfg.model.roi_relation_head.use_gt_box and not cfg.model.roi_relation_head.use_gt_object_label:
        # SGDet mode
        pass
    else:
        raise ValueError("Invalid task mode: use_gt_box and use_gt_object_label combination is invalid")
    
    # Validate dataset — with the new simplified schema, name is enough
    if not cfg.datasets.name and not cfg.datasets.train and not cfg.datasets.test:
        raise ValueError("Must specify at least datasets.name (e.g. 'PSG', 'VG150', 'IndoorVG')")
    
    # Validate model components
    if cfg.model.relation_on and cfg.model.roi_relation_head.num_classes <= 0:
        raise ValueError("relation_on=True requires num_classes > 0")
    
    return True


# ============================================================================
# Registration with Hydra (call this in __init__.py)
# ============================================================================

def register_configs():
    """Register all structured configs with Hydra's ConfigStore."""
    from hydra.core.config_store import ConfigStore
    
    cs = ConfigStore.instance()
    
    # Register main config
    cs.store(name="base_sgg_config", node=SGGConfig)
    
    # Register sub-configs for composition
    cs.store(group="model", name="base_model", node=ModelConfig)
    cs.store(group="model/backbone", name="yolo", node=BackboneConfig)
    cs.store(group="solver", name="base_solver", node=SolverConfig)
    cs.store(group="test", name="base_test", node=TestConfig)
    
    print("✓ Structured configs registered with Hydra")
