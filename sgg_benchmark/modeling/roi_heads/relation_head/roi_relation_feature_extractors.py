# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from sgg_benchmark.modeling import registry
from sgg_benchmark.modeling.make_layers import make_fc
from sgg_benchmark.structures.box_ops import box_union, filter_instances, box_resize
from sgg_benchmark.modeling.roi_heads.box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from sgg_benchmark.modeling.roi_heads.attribute_head.roi_attribute_feature_extractors import make_roi_attribute_feature_extractor


class _FP32LayerNorm(nn.Module):
    """LayerNorm that always computes in float32, regardless of fp16 autocast.

    PyTorch's built-in nn.LayerNorm respects autocast and will run in fp16 when
    called inside torch.autocast(dtype=float16). With elementwise_affine=False
    (no learnable scale/bias) the internal variance calculation uses eps=1e-5.
    In fp16, 1e-5 rounds to exactly 0.0, so any near-zero-variance vector
    gets divided by 0 → NaN. Running in float32 avoids this entirely.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.float(), self.normalized_shape, weight=None, bias=None, eps=self.eps
        ).to(x.dtype)


@registry.ROI_RELATION_FEATURE_EXTRACTORS.register("RelationFeatureExtractor")
class RelationFeatureExtractor(nn.Module):
    """
    Heads for Motifs for relation triplet classification
    """
    def __init__(self, cfg, in_channels):
        super(RelationFeatureExtractor, self).__init__()
        if not hasattr(cfg, 'clone'):
            from sgg_benchmark.config import Config
            cfg = Config(cfg)
        self.cfg = cfg.clone()

        # Extract a single channel value for box head from multi-scale input if needed
        if hasattr(in_channels, '__getitem__') and not isinstance(in_channels, torch.Tensor):
            in_channels = int(in_channels[0])

        # should corresponding to obj_feature_map function in neural-motifs
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pool_all_levels = cfg.MODEL.ROI_RELATION_HEAD.POOLING_ALL_LEVELS

        self.use_spatial = cfg.MODEL.ROI_RELATION_HEAD.USE_SPATIAL_FEATURES
        self.use_union = cfg.MODEL.ROI_RELATION_HEAD.USE_UNION_FEATURES
        # When use_union_features_inference=False, skip union RoI at inference to
        # save ~47ms/image (profiled). The model trains with union but the residual
        # path (sub*obj + sub - obj + freq_bias) is sufficient at test time.
        union_at_inference = cfg.MODEL.ROI_RELATION_HEAD.get(
            'USE_UNION_FEATURES_INFERENCE', True)
        self.skip_union_at_inference = (not union_at_inference) and self.use_union

        assert self.use_spatial or self.use_union, 'No features selected for the relation head'
        
        if cfg.MODEL.ATTRIBUTE_ON:
            self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, half_out=True, cat_all_levels=pool_all_levels)
            self.att_feature_extractor = make_roi_attribute_feature_extractor(cfg, in_channels, half_out=True, cat_all_levels=pool_all_levels)
            self.out_channels = self.feature_extractor.out_channels * 2
        else:
            self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, cat_all_levels=pool_all_levels)
            self.out_channels = self.feature_extractor.out_channels

        # separate spatial
        self.separate_spatial = self.cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        if self.separate_spatial:
            input_size = self.feature_extractor.resize_channels
            out_dim = self.feature_extractor.out_channels
            self.spatial_fc = nn.Sequential(*[make_fc(self.cfg, input_size, out_dim//2), nn.ReLU(inplace=True),
                                              make_fc(self.cfg, out_dim//2, out_dim), nn.ReLU(inplace=True),
                                            ])

        # union rectangle size
        self.rect_size = resolution * 4 -1
        self.rect_conv = nn.Sequential(*[
            nn.Conv2d(2, in_channels //2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels//2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels, momentum=0.01),
            ])


    def forward(self, x, proposals, rel_pair_idxs=None):
        # Fast path: skip all union computation at inference when flagged.
        # This avoids the expensive RoIAlign call over N_rel union boxes.
        if self.skip_union_at_inference and not self.training:
            return None

        device = x[0].device
        union_proposals = []
        rect_inputs = []
        for proposal, rel_pair_idx in zip(proposals, rel_pair_idxs):
            head_proposal = filter_instances(proposal, rel_pair_idx[:, 0])
            tail_proposal = filter_instances(proposal, rel_pair_idx[:, 1])
            if self.use_union:
                union_boxes = box_union(head_proposal["boxes"], tail_proposal["boxes"])
                union_proposal = proposal.copy()
                union_proposal["boxes"] = union_boxes
                # YOLO proposals carry `lb_boxes` (letterbox-space coords) that the
                # PoolerYOLO uses for RoIAlign.  After proposal.copy() the `lb_boxes`
                # field still holds the OBJECT-level lb_boxes (N_obj rows), but we
                # now need N_rel union lb_boxes.  Compute them from the already-
                # filtered head/tail lb_boxes so the YOLO pooler extracts the correct
                # number of union features.
                if "lb_boxes" in head_proposal:
                    union_proposal["lb_boxes"] = box_union(
                        head_proposal["lb_boxes"], tail_proposal["lb_boxes"])
                if "labels" in union_proposal:
                    del union_proposal["labels"]
                union_proposals.append(union_proposal)

            if self.use_spatial:
                # use range to construct rectangle, sized (rect_size, rect_size)
                num_rel = rel_pair_idx.size(0)
                dummy_x_range = torch.arange(self.rect_size, device=device).view(1, 1, -1).expand(num_rel, self.rect_size, self.rect_size)
                dummy_y_range = torch.arange(self.rect_size, device=device).view(1, -1, 1).expand(num_rel, self.rect_size, self.rect_size)
                # resize bbox to the scale rect_size
                head_proposal_resized = box_resize(head_proposal, head_proposal["image_size"], (self.rect_size, self.rect_size))
                tail_proposal_resized = box_resize(tail_proposal, tail_proposal["image_size"], (self.rect_size, self.rect_size))
                head_rect = ((dummy_x_range >= head_proposal_resized["boxes"][:,0].floor().view(-1,1,1).long()).bool() & \
                            (dummy_x_range <= head_proposal_resized["boxes"][:,2].ceil().view(-1,1,1).long()).bool() & \
                            (dummy_y_range >= head_proposal_resized["boxes"][:,1].floor().view(-1,1,1).long()).bool() & \
                            (dummy_y_range <= head_proposal_resized["boxes"][:,3].ceil().view(-1,1,1).long()).bool()).float()

                tail_rect = ((dummy_x_range >= tail_proposal_resized["boxes"][:,0].floor().view(-1,1,1).long()).bool() & \
                            (dummy_x_range <= tail_proposal_resized["boxes"][:,2].ceil().view(-1,1,1).long()).bool() & \
                            (dummy_y_range >= tail_proposal_resized["boxes"][:,1].floor().view(-1,1,1).long()).bool() & \
                            (dummy_y_range <= tail_proposal_resized["boxes"][:,3].ceil().view(-1,1,1).long()).bool()).float()

                rect_input = torch.stack((head_rect, tail_rect), dim=1) # (num_rel, 4, rect_size, rect_size)
                rect_inputs.append(rect_input)

        # merge two parts
        if self.separate_spatial and self.use_union and self.use_spatial:
            assert (self.use_union and self.use_spatial) # only support this mode if self.separate_spatial is True
            # rectangle feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
            rect_inputs = torch.cat(rect_inputs, dim=0)
            rect_features = self.rect_conv(rect_inputs)
            union_vis_features = self.feature_extractor.pooler(x, union_proposals)
            region_features = self.feature_extractor.forward_without_pool(union_vis_features)
            spatial_features = self.spatial_fc(rect_features.view(rect_features.size(0), -1))
            union_features = (region_features, spatial_features)
        else:
            union_vis_features = torch.tensor([]).to(device)
            rect_features = torch.tensor([]).to(device) 
            if self.use_union:
                union_vis_features = self.feature_extractor.pooler(x, union_proposals)
                union_features = union_vis_features 
            if self.use_spatial:
                rect_inputs = torch.cat(rect_inputs, dim=0)
                rect_features = self.rect_conv(rect_inputs)
                union_features = rect_features 
            if self.use_union and self.use_spatial:
                union_features = union_vis_features + rect_features
            union_features = self.feature_extractor.forward_without_pool(union_features) # (total_num_rel, out_channels)

        if self.cfg.MODEL.ATTRIBUTE_ON:
            union_att_features = self.att_feature_extractor.pooler(x, union_proposals)
            union_features_att = union_att_features + rect_features
            union_features_att = self.att_feature_extractor.forward_without_pool(union_features_att)
            union_features = torch.cat((union_features, union_features_att), dim=-1)
            
        return union_features

@registry.ROI_RELATION_FEATURE_EXTRACTORS.register("FeatIdxRelationExtractor")
class FeatIdxRelationExtractor(nn.Module):
    """Zero-RoI-Align union feature extractor for REACTPlusPlusPredictor.

    Instead of running RoI Align on N_rel union boxes (~47ms for 1225 pairs),
    this module gathers the P3 backbone token at the centre of each union box
    via a single integer index lookup — effectively free at any pair count.

    Mechanism (mirrors reactV2.py union gather, lines 204-232):
        1. Compute union box centre in letterbox pixel space.
        2. Map centre to P3 grid: col = int(cx / lb_sz * W3), row = int(cy / lb_sz * H3).
        3. Gather p3[batch, :, row, col].T  →  [N_rel, C_p3].
        4. Linear(C_p3, out_channels) to match the downstream predictor's in_channels.

    Cost vs RoI Align:
        RoI Align:  O(N_rel × 7 × 7 × bilinear) ≈ 47 ms at 1225 pairs
        FeatIdx:    O(N_rel × 1) integer index + O(N_rel × C_p3 × out_ch) matmul ≈ 0.1 ms

    No skip-at-inference logic needed: the operation is free so union features
    can be used at both train and test time without any latency penalty.
    """

    def __init__(self, cfg, in_channels):
        super().__init__()
        if not hasattr(cfg, 'clone'):
            from sgg_benchmark.config import Config
            cfg = Config(cfg)
        self.cfg = cfg.clone()

        # in_channels is a list for multi-scale FPN: [192, 384, 576] for YOLOv8m.
        # We use only P3 (finest scale, index 0) for the union gather.
        if hasattr(in_channels, '__getitem__') and not isinstance(in_channels, torch.Tensor):
            p3_channels = int(in_channels[0])
        else:
            p3_channels = int(in_channels)

        # out_channels must match the roi_box_head output so the downstream
        # predictor's union_proj = Linear(in_channels, D) sees the right dim.
        self.out_channels = cfg.model.roi_box_head.mlp_head_dim  # e.g. 256

        # Single linear projection: raw P3 channel → feature space.
        # Much cheaper than the two-layer MLP + RoI Align in RelationFeatureExtractor.
        self.proj = nn.Linear(p3_channels, self.out_channels, bias=False)
        # Light normalisation before projection for training stability.
        # Uses _FP32LayerNorm: plain nn.LayerNorm runs in fp16 under autocast and
        # can produce NaN when P3 feature vectors have near-zero variance.
        self.norm = _FP32LayerNorm(p3_channels)

        self.use_union   = cfg.model.roi_relation_head.use_union_features
        self.use_spatial = cfg.model.roi_relation_head.use_spatial_features
        assert self.use_union or self.use_spatial, \
            "FeatIdxRelationExtractor requires use_union_features=true"

    @torch.autocast(device_type='cuda', enabled=False)
    def forward(self, x, proposals, rel_pair_idxs=None):
        """
        Args:
            x (list[Tensor]): FPN feature maps [P3, P4, P5] with shape (B, C, H, W).
            proposals (list[dict]): per-image proposal dicts containing 'lb_boxes'.
            rel_pair_idxs (list[Tensor]): per-image (N_rel, 2) subject/object indices.
        Returns:
            Tensor: (N_rel_total, out_channels)
        """
        device = x[0].device
        p3 = x[0]                         # (B, C_p3, H3, W3)
        H3, W3 = p3.shape[2], p3.shape[3]  # e.g. 80×80 for 640-pixel input

        union_feat_list = []
        for batch_idx, (props, pairs) in enumerate(zip(proposals, rel_pair_idxs)):
            if pairs.shape[0] == 0:
                continue

            # GT proposals added during training have no lb_boxes / lb_input_size.
            # Using their original-image coords as letterbox coords would map to the
            # wrong P3 grid cell.  Return zeros for these pairs — they will be handled
            # by the relation head's label-smoothed CE loss without union context.
            if "lb_boxes" not in props:
                n_rel = pairs.shape[0]
                union_feat_list.append(
                    torch.zeros(n_rel, self.out_channels, device=device))
                continue

            s_local, o_local = pairs[:, 0], pairs[:, 1]

            # Use lb_boxes (letterbox pixel coords) so that the union-box centre maps
            # correctly onto the P3 feature grid, which is in letterbox space.
            boxes  = props.get("lb_boxes", props["boxes"])  # (N_obj, 4)
            lb_sz  = float(props.get("lb_input_size", 640))

            # Union box centre: smallest containing box of sub and obj boxes
            ucx = (torch.min(boxes[s_local, 0], boxes[o_local, 0]) +
                   torch.max(boxes[s_local, 2], boxes[o_local, 2])) * 0.5
            ucy = (torch.min(boxes[s_local, 1], boxes[o_local, 1]) +
                   torch.max(boxes[s_local, 3], boxes[o_local, 3])) * 0.5

            # Map centre to P3 grid cell (stride = lb_sz / H3, e.g. 640/80 = 8)
            col = (ucx / lb_sz * W3).long().clamp(0, W3 - 1)  # (N_rel,)
            row = (ucy / lb_sz * H3).long().clamp(0, H3 - 1)  # (N_rel,)

            # Gather: p3[batch_idx] is (C_p3, H3, W3)
            # p3[batch_idx, :, row, col] → (C_p3, N_rel).T → (N_rel, C_p3)
            feat = p3[batch_idx, :, row, col].t()   # (N_rel, C_p3)
            union_feat_list.append(feat)

        if not union_feat_list:
            return torch.zeros(0, self.out_channels, device=device)

        # Cast to float32: fp16 Linear matmul (self.proj) can overflow when weight
        # norms grow after training.  Running the projection in float32 is safe because
        # _FP32LayerNorm already normalises values to ~N(0,1) before self.proj.
        union_feats = torch.cat(union_feat_list, dim=0).float()   # (N_rel_total, C_p3) fp32
        projected = self.proj(self.norm(union_feats))              # (N_rel_total, out_channels)
        return projected


@registry.ROI_RELATION_FEATURE_EXTRACTORS.register("P5SceneContextExtractor")
class P5SceneContextExtractor(nn.Module):
    """AIFI-based global scene context extractor using the deepest YOLO feature (P5).
    Inspired by the AIFI implementation in https://github.com/ultralytics/ultralytics/blob/a4fb22a7b6bcac00bd2a4918f4b93d66d88356bf/ultralytics/nn/modules/transformer.py#L171

    Instead of extracting a single P3 pixel at the union-box centre (FeatIdxRelationExtractor),
    this module builds a compact set of *image-level* scene tokens that relation
    representations can query via cross-attention.

    Pipeline per forward pass:
        1. Adaptive-avg-pool P5  [B, C_p5, H5, W5] → [B, C_p5, k, k]  (k=4 → 16 tokens).
        2. Flatten + project     [B, k², C_p5]      → [B, k², out_channels].
        3. AIFI self-attention   one TransformerEncoderLayer so each token sees the
                                 full scene before relation pairs consult it.
        4. Return flat           [B * k², out_channels].

    The predictor detects "scene mode" by comparing shape[0] vs N_rel_total and
    activates `scene_cross_attn(r → scene_tokens)` instead of the per-pair union branch.

    Latency profile (approx., 640-px input, YOLOv8m, B=1):
        adaptive_avg_pool → 0.01 ms
        proj (576→256)    → 0.02 ms
        AIFI (16 tokens)  → 0.05 ms      (16×16 attention matrix — essentially free)
        cross-attn in predictor (N_rel × 16)  slightly cheaper than N_rel × 38 (old proto attn)
    """

    POOL_K = 4  # 4×4 = 16 spatial scene tokens per image

    def __init__(self, cfg, in_channels):
        super().__init__()
        if not hasattr(cfg, 'clone'):
            from sgg_benchmark.config import Config
            cfg = Config(cfg)
        self.cfg = cfg.clone()

        self.pool_k = self.POOL_K
        # out_channels must match roi_box_head.mlp_head_dim (= predictor's in_channels)
        self.out_channels = cfg.model.roi_box_head.mlp_head_dim   # e.g. 256

        # LazyLinear infers the P5 channel count on the first forward pass, making
        # this extractor robust to any YOLO variant regardless of neck channel widths.
        self.proj_lin = nn.LazyLinear(self.out_channels, bias=False)

        # AIFI: one lightweight TransformerEncoderLayer on k*k=16 tokens.
        # norm_first=True (Pre-LN) for training stability.
        # dim_feedforward=2× for modest extra capacity without much cost.
        self.aifi = nn.TransformerEncoderLayer(
            d_model=self.out_channels,
            nhead=4,
            dim_feedforward=self.out_channels * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )

    @torch.autocast(device_type='cuda', enabled=False)
    def forward(self, x, proposals, rel_pair_idxs=None):
        """
        Args:
            x (list[Tensor]): FPN feature maps from YOLO neck, each (B, C, H, W).
                              Uses x[-1] (deepest / smallest spatial / most semantic).
            proposals, rel_pair_idxs: unused — context is image-level, not per-pair.
        Returns:
            Tensor: (B * pool_k * pool_k, out_channels)
                Image i occupies rows [i*k² : (i+1)*k²].
        """
        # Cast to float32 immediately: the backbone emits fp16 tensors, but this
        # entire method runs with autocast disabled so every op stays in float32.
        # Keeps proj_lin (float32 weight by default) and aifi (float32) consistent.
        p5 = x[-1].float()                                     # (B, C_p5, H5, W5)
        k  = self.pool_k

        # 1. Spatially compress P5 to k×k tokens
        pooled = F.adaptive_avg_pool2d(p5, (k, k))            # (B, C_p5, k, k)
        tokens = pooled.flatten(2).permute(0, 2, 1)           # (B, k², C_p5)  float32

        # 2. Pre-norm (no learnable scale/bias) + project to out_channels.
        # NOTE: F.layer_norm(tokens, [C_p5]) would fail ONNX export because the legacy
        # TorchScript exporter requires normalized_shape to be a compile-time constant,
        # but C_p5 = p5.shape[1] is a dynamic Gather node in the traced graph.
        # Manually expand the formula instead — identical math, fully ONNX-compatible:
        #   tokens = (tokens - mean) / sqrt(var + eps)
        # Already float32, so no intermediate cast needed.
        _mean = tokens.mean(dim=-1, keepdim=True)
        _var  = tokens.var(dim=-1, keepdim=True, unbiased=False)
        tokens = (tokens - _mean) / (_var + 1e-5).sqrt()      # (B, k², C_p5)  float32
        tokens = self.proj_lin(tokens)                         # (B, k², out_ch) float32

        # 3. AIFI: intra-scale self-attention (free at 16 tokens).
        # Already float32 — no cast needed.
        tokens = self.aifi(tokens)                             # (B, k², out_ch) float32

        # 4. Return flat layout [B*k², out_ch] — use -1 for ONNX-compatible reshape
        #    (avoids dynamic B*k*k multiplication in graph).
        return tokens.reshape(-1, self.out_channels)


def make_roi_relation_feature_extractor(cfg, in_channels):
    func = registry.ROI_RELATION_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_RELATION_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
