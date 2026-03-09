# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import DictConfig

from sgg_benchmark.modeling import registry
from sgg_benchmark.modeling.backbone import resnet
from sgg_benchmark.modeling.poolers import Pooler, PoolerYOLO
from sgg_benchmark.modeling.make_layers import group_norm
from sgg_benchmark.modeling.make_layers import make_fc
from math import gcd

import matplotlib.pyplot as plt
import math

from sgg_benchmark.layers.misc import Conv

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("PatchFeatureExtractor")
class PatchFeatureExtractor(nn.Module):
    # Inspired by https://github.com/lorjul/panoptic-scene-graph-generation
    def __init__(self, cfg: DictConfig, in_channels, half_out=False, cat_all_levels=False):
        super(PatchFeatureExtractor, self).__init__()

        c1, c2, c3 = cfg.model.yolo.out_channels
        c_out = c1
        
        # Normalize scales
        self.norm_p3 = Conv(c1, c_out, 1, 1)
        self.norm_p4 = Conv(c2, c_out, 1, 1)
        self.norm_p5 = Conv(c3, c_out, 1, 1)
        
        # Fuse with simple 1x1 convolutions
        self.fuse_p54 = Conv(c_out * 2, c_out, 1, 1)
        self.fuse_all = Conv(c_out * 3, c_out, 1, 1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.embed_dim = cfg.model.roi_box_head.mlp_head_dim
        self.patch_size = cfg.model.roi_box_head.patch_size

        # Idea 3: FPN-based Multi-Scale Patches with level-aware kernels
        # To maintain the same token grid resolution across levels:
        # P3 (stride 8) kernel size = patch_size
        # P4 (stride 16) kernel size = patch_size // 2
        # P5 (stride 32) kernel size = patch_size // 4
        self.patch_embed_p3 = nn.Conv2d(c_out, self.embed_dim, self.patch_size, self.patch_size)
        self.patch_embed_p4 = nn.Conv2d(c_out, self.embed_dim, max(1, self.patch_size // 2), max(1, self.patch_size // 2))
        self.patch_embed_p5 = nn.Conv2d(c_out, self.embed_dim, max(1, self.patch_size // 4), max(1, self.patch_size // 4))

    def forward(self, x):

        p3, p4, p5 = x
        
        p3_n = self.norm_p3(p3)
        p4_n = self.norm_p4(p4)
        p5_n = self.norm_p5(p5)
        
        # Original logic for fused map (used for bounding box features etc.)
        p5_up = self.upsample(self.upsample(p5_n))
        p4_up = self.upsample(p4_n)
        p54_fused = self.fuse_p54(torch.cat([p5_up, p4_up], dim=1))
        combined = self.fuse_all(torch.cat([p3_n, p54_fused, p5_up], dim=1))
    
        # Idea 3: FPN-based Multi-Scale Patches
        # Extract patches from each level separately using the level-aware layers
        
        # We ensure coverage of the full feature map by padding to a multiple of patch_size if needed
        def pad_to_multiple(tensor, multiple):
            h, w = tensor.shape[2:]
            ph = (multiple - h % multiple) % multiple
            pw = (multiple - w % multiple) % multiple
            if ph > 0 or pw > 0:
                # Pad bottom and right
                return F.pad(tensor, (0, pw, 0, ph)), (h + ph, w + pw)
            return tensor, (h, w)

        p3_pad, _ = pad_to_multiple(p3_n, self.patch_size)
        patches3 = self.patch_embed_p3(p3_pad).flatten(2).transpose(1, 2)
        
        p4_pad, _ = pad_to_multiple(p4_n, max(1, self.patch_size // 2))
        patches4 = self.patch_embed_p4(p4_pad).flatten(2).transpose(1, 2)
        
        p5_pad, _ = pad_to_multiple(p5_n, max(1, self.patch_size // 4))
        patches5 = self.patch_embed_p5(p5_pad).flatten(2).transpose(1, 2)
        
        # Concatenate all patches [B, N_total, D]
        multi_scale_patches = torch.cat([patches3, patches4, patches5], dim=1)

        return combined, multi_scale_patches
        
@registry.ROI_BOX_FEATURE_EXTRACTORS.register("YOLOV8FeatureExtractor3")
class YOLOV8FeatureExtractor3(nn.Module):
    def __init__(self, cfg: DictConfig, in_channels, half_out=False, cat_all_levels=False):
        super(YOLOV8FeatureExtractor3, self).__init__()

    def forward(self, x, proposals):
        idxs = [proposal["feat_idx"] for proposal in proposals]

        s = gcd(*[feat.shape[1] for feat in x]) # smallest vector length (64 for YOLOv8)

        obj_feats = torch.cat([x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, s, x.shape[1] // s).mean(dim=-1) for x in x], dim=1)

        feats = torch.cat([feats[idx] for feats, idx in zip(obj_feats, idxs)])

        return feats
    

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("YOLOV8FeatureExtractor2")
class YOLOV8FeatureExtractor2(nn.Module):
    def __init__(self, cfg: DictConfig, in_channels, half_out=False, cat_all_levels=False):
        super(YOLOV8FeatureExtractor2, self).__init__()

        resolution = cfg.model.roi_box_head.pooler_resolution
        sampling_ratio = cfg.model.roi_box_head.pooler_sampling_ratio

        input_size = in_channels * resolution ** 2
        representation_size = cfg.model.roi_box_head.mlp_head_dim
        use_gn = cfg.model.roi_box_head.use_gn
        self.fc6 = make_fc(cfg, input_size, representation_size, use_gn)

        if half_out:
            out_dim = int(representation_size / 2)
        else:
            out_dim = representation_size
        
        self.fc7 = make_fc(cfg, representation_size, out_dim, use_gn)
        self.fc8 = make_fc(cfg, cfg.model.yolo.out_channels[0], out_dim, use_gn)
        self.fc9 = make_fc(cfg, out_dim, out_dim, use_gn)
        self.resize_channels = input_size
        self.out_channels = out_dim

    def forward(self, x, proposals):
        idxs = [proposal["feat_idx"] for proposal in proposals]

        s = gcd(*[feat.shape[1] for feat in x]) # smallest vector length (64 for YOLOv8)

        obj_feats = torch.cat([x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, s, x.shape[1] // s).mean(dim=-1) for x in x], dim=1)

        feats = torch.cat([feats[idx] for feats, idx in zip(obj_feats, idxs)])

        feats = F.relu(self.fc8(feats))
        feats = F.relu(self.fc9(feats))
        return feats
    
    def feat_visualization(self, x, idxs, stage, save_dir, n=32):
        _, channels, height, width = x.shape  # batch, channels, height, width
        f = save_dir / f"stage{stage}_{idxs}_features.png"  # filename

        blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels

        n = min(n, channels)  # number of plots
        _, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 rows x n/8 cols
        ax = ax.ravel()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        for i in range(n):
            ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
            ax[i].axis("off")

        plt.savefig(f, dpi=300, bbox_inches="tight")
        plt.close()
    
    def forward_without_pool(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x
    
@registry.ROI_BOX_FEATURE_EXTRACTORS.register("YOLOV8FeatureExtractor")
class YOLOV8FeatureExtractor(nn.Module):
    def __init__(self, cfg: DictConfig, in_channels, half_out=False, cat_all_levels=False):
        super(YOLOV8FeatureExtractor, self).__init__()

        resolution = cfg.model.roi_box_head.pooler_resolution
        sampling_ratio = cfg.model.roi_box_head.pooler_sampling_ratio
        pooler = Pooler(
            output_size=(resolution, resolution),
            sampling_ratio=sampling_ratio,
            cat_all_levels=cat_all_levels,
            in_channels=in_channels,
            scales=cfg.model.roi_box_head.pooler_scales,
        )

        self.pooler = pooler

        input_size = in_channels * resolution ** 2
        representation_size = cfg.model.roi_box_head.mlp_head_dim
        use_gn = cfg.model.roi_box_head.use_gn
        self.pooler = pooler
        self.fc6 = make_fc(cfg, input_size, representation_size, use_gn)

        if half_out:
            out_dim = int(representation_size / 2)
        else:
            out_dim = representation_size
        
        self.fc7 = make_fc(cfg, representation_size, out_dim, use_gn)
        self.resize_channels = input_size
        self.out_channels = out_dim

        self.down_sample = nn.ModuleList()
        channels_scale = cfg.model.yolo.out_channels
        for i in range(len(channels_scale)):
            if channels_scale[i] != in_channels:
                self.down_sample.append(nn.Conv2d(channels_scale[i], in_channels, kernel_size=1, stride=1, padding=0))
            else:
                self.down_sample.append(None)

    def forward(self, x, proposals):
        for i, down_sample in enumerate(self.down_sample):
            if down_sample is not None:
                x[i] = down_sample(x[i])
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x
    
    def forward_without_pool(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x

class _FP32LayerNorm(nn.Module):
    """LayerNorm that always computes in float32, regardless of autocast.

    Under fp16 autocast, nn.LayerNorm's internal variance can underflow to 0
    when the input has near-zero activations (e.g. freshly-initialised Linear
    with frozen YOLO backbone features).  PyTorch's CUDA LN kernel does upcast
    internally for the *accumulation*, but when called from Python via
    torch.nn.functional.layer_norm the C++ path may not upcast the *input*
    before computing the squared residuals, causing var ≈ 0 → x/0 = Inf → NaN.
    This wrapper explicitly casts input to float32 before calling F.layer_norm.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.float(), self.normalized_shape, weight=None, bias=None, eps=self.eps
        ).to(x.dtype)


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("DAMPBoxFeatureExtractor")
class DAMPBoxFeatureExtractor(nn.Module):
    """Detection-Anchored Multi-Scale Pooling (DAMP).

    A drop-in replacement for RoI Align that exploits a signal already produced
    by the YOLO detector: ``feat_idx``, the flat index of the FPN grid cell where
    the detection confidence was highest.  Rather than re-sampling the image with
    bilinear interpolation over a 7×7 grid (as RoI Align does), DAMP treats the
    confidence-peak cell as a precision anchor and aggregates features around it
    across all FPN pyramid levels in a single O(N) pass.

    ── Motivation ────────────────────────────────────────────────────────────────
    RoI Align pools from a fixed-size grid *defined by the predicted bounding box*.
    This introduces two sources of noise for scene-graph tasks:
      1. Box imprecision — especially for small or partially occluded objects, the
         regressed box may not be well-centred on the object's discriminative region.
      2. Fixed resolution — a 7×7 grid captures the same spatial extent regardless
         of object scale, diluting features for tiny objects and wasting compute for
         large ones.

    DAMP instead anchors to the YOLO *confidence peak* — the single grid cell where
    the detector is most certain about the object's centre — and propagates that
    anchor fractionally across all FPN levels.  P3 (stride 8) gives fine texture
    and geometry; P4 (stride 16) gives part-level context; P5 (stride 32) gives
    semantic, scene-level context.  A Gaussian-weighted neighbourhood (r ≥ 1) adds
    local spatial tolerance without any additional parameters.

    ── Algorithm ─────────────────────────────────────────────────────────────────
    Given a flat index feat_idx ∈ [0, H3·W3 + H4·W4 + H5·W5) for each object:

      1. Decode the detection anchor to fractional image coordinates:            O(N)
             level   ← bucketize(feat_idx, cumulative level sizes)
             (r, c)  ← divmod(feat_idx − level_offset, level_width)
             cx_frac ← (c + 0.5) / W_level  ∈ (0, 1)
             cy_frac ← (r + 0.5) / H_level  ∈ (0, 1)

      2. For each FPN level i ∈ {P3, P4, P5}:                                   O(N·K·C_i)
           Snap anchor to the nearest integer grid cell, then gather
           a (2r+1)² Gaussian-weighted neighbourhood of RAW C_i-dim vectors:
             g_i ← Σ_{dr,dc} w(dr,dc) · fpn_i[b, r_i+dr, c_i+dc]  ∈ ℝᶜⁱ

      3. Project each gathered vector to D dimensions:                            O(N·L·C_i·D)
             f_i ← LN(g_i) · W_i  ∈ ℝᴰ

      4. Fuse:  o ← Linear(LN([f_P3 ‖ f_P4 ‖ f_P5]))  →  ℝᴰ

    Steps 2–3 process only N·K·L vectors (e.g. 80×1×3 = 240) instead of
    projecting all 8400 spatial tokens and discarding 35× of the work.

    ── Config ────────────────────────────────────────────────────────────────────
      feat_idx_multiscale : bool  gather P3+P4+P5 (True, default) or assigned level only
      feat_idx_neighbors  : int   neighbourhood radius r (0 = centre only, default)

    Output: [N_obj_total, out_channels] — drop-in replacement for YOLOV8FeatureExtractor.
    """

    def __init__(self, cfg: DictConfig, in_channels, half_out=False, cat_all_levels=False):
        super().__init__()

        out_ch = cfg.model.roi_box_head.mlp_head_dim   # e.g. 256
        if half_out:
            out_ch = out_ch // 2
        self.out_channels = out_ch

        # FPN channel widths — read from YOLO backbone config to handle any variant.
        # Do NOT rely on in_channels arg: make_roi_box_feature_extractor resolves it
        # to in_channels[0] = P3 channels (192), losing P4/P5 info.
        fpn_ch = list(cfg.model.yolo.out_channels)   # e.g. [192, 384, 576]

        # Per-level projection applied AFTER gathering — only on the N·K gathered
        # raw vectors, not on the full H×W feature map.  This avoids projecting
        # ~8400 tokens to then discard 35× of the work (for typical N≈80 objects).
        # Each projector: LN(C_i, no affine) → Linear(C_i → out_ch, no bias).
        # NOTE: LayerNorm is wrapped in _FP32LayerNorm to force float32 computation
        # under fp16 autocast — PyTorch's built-in LN uses eps=1e-5 but the *input*
        # can still be fp16 causing inf when variance ≈ 0 (near-zero feature vectors).
        self.level_projs = nn.ModuleList([
            nn.Sequential(
                _FP32LayerNorm(c),
                nn.Linear(c, out_ch, bias=False),
            )
            for c in fpn_ch
        ])

        # Multi-scale gather (default, ablation winner A8):
        # Decode the object centre from feat_idx → fractional (cx, cy) ∈ (0,1),
        # then snap to the nearest integer grid cell on P3, P4, AND P5.
        # The three projected tokens are concatenated and fused via LayerNorm + Linear.
        # Set feat_idx_multiscale=False to fall back to single-level peak-token gather.
        self._multiscale = bool(getattr(cfg.model.roi_box_head, 'feat_idx_multiscale', True))
        if self._multiscale:
            n_levels = len(fpn_ch)  # 3 for YOLOv8 P3/P4/P5
            self.ms_norm = _FP32LayerNorm(out_ch * n_levels)
            self.ms_proj = nn.Linear(out_ch * n_levels, out_ch, bias=False)

        # Optional Gaussian-weighted neighbourhood applied at EACH FPN level during the
        # multi-scale gather.  r=0 → single centre token (default); r=1 → 3×3 = 9 tokens.
        # Zero extra parameters — all tokens are already projected by level_projs.
        self._neighbors = int(getattr(cfg.model.roi_box_head, 'feat_idx_neighbors', 0))
        if self._neighbors > 0:
            r = self._neighbors
            self._neighbor_deltas = [
                (dr, dc) for dr in range(-r, r + 1) for dc in range(-r, r + 1)
            ]
            import math as _math
            weights = torch.tensor(
                [_math.exp(-(dr ** 2 + dc ** 2)) for dr, dc in self._neighbor_deltas],
                dtype=torch.float32)
            self.register_buffer('_neighbor_weights', weights / weights.sum())

    @torch.autocast(device_type='cuda', enabled=False)
    def forward(self, x, proposals):
        """
        Args:
            x (list[Tensor]): FPN maps [P3, P4, P5], each (B, C_i, H_i, W_i).
            proposals (list[dict]): per-image dicts containing 'feat_idx' [N_obj].
        Returns:
            Tensor: [N_obj_total, out_channels]
        """
        device = x[0].device
        B = x[0].shape[0]

        level_hw = [(feat.shape[2], feat.shape[3]) for feat in x]

        # Gather by feat_idx — one integer per detected object.
        # ONNX-COMPATIBILITY NOTE (B=1 path): see comment below.
        if B == 1:
            idxs = proposals[0]["feat_idx"]          # [N_obj] — dynamic tensor from NMS
            if idxs.shape[0] == 0:
                return x[0].new_zeros(0, self.out_channels)
            batch_ids = torch.zeros_like(idxs)       # [N_obj] zeros — preserves ONNX dynamic shape
        else:
            obj_counts = [p["feat_idx"].shape[0] for p in proposals]
            total = sum(obj_counts)
            if total == 0:
                return x[0].new_zeros(0, self.out_channels)
            batch_ids = torch.arange(B, device=device).repeat_interleave(
                torch.tensor(obj_counts, dtype=torch.long, device=device)
            )
            idxs = torch.cat([p["feat_idx"] for p in proposals])    # [N_obj]

        if self._multiscale:
            # ── DAMP: gather-then-project ──────────────────────────────────────
            # Decode fractional anchor coordinates from feat_idx (assigned level).
            level_sizes_t = torch.tensor(
                [h * w for h, w in level_hw], device=device, dtype=torch.long)
            offsets = torch.cat(
                [level_sizes_t.new_zeros(1), level_sizes_t.cumsum(0)[:-1]])
            H_t = torch.tensor([h for h, w in level_hw], device=device, dtype=torch.long)
            W_t = torch.tensor([w for h, w in level_hw], device=device, dtype=torch.long)
            obj_level = torch.bucketize(idxs, offsets, right=True) - 1  # [N_obj]
            local_idx = idxs - offsets[obj_level]
            obj_W = W_t[obj_level]
            obj_H = H_t[obj_level]
            cx_frac = ((local_idx % obj_W).float() + 0.5) / obj_W.float()   # [N_obj] ∈ (0,1)
            cy_frac = ((local_idx // obj_W).float() + 0.5) / obj_H.float()  # [N_obj] ∈ (0,1)

            ms_feats = []
            for i, (H_i, W_i) in enumerate(level_hw):
                r_i = (cy_frac * H_i).long().clamp(0, H_i - 1)   # [N_obj]
                c_i = (cx_frac * W_i).long().clamp(0, W_i - 1)   # [N_obj]
                # Gather raw C_i-dim vectors (cheap — only N·K gathers).
                feat_i = x[i]  # (B, C_i, H_i, W_i)
                if self._neighbors > 0:
                    w = self._neighbor_weights.to(dtype=feat_i.dtype)  # [K]
                    acc = feat_i.new_zeros(batch_ids.shape[0], feat_i.shape[1])
                    for (dr, dc), wi in zip(self._neighbor_deltas, w):
                        rn = (r_i + dr).clamp(0, H_i - 1)
                        cn = (c_i + dc).clamp(0, W_i - 1)
                        acc = acc + feat_i[batch_ids, :, rn, cn] * wi  # [N, C_i]
                    raw = acc
                else:
                    raw = feat_i[batch_ids, :, r_i, c_i]              # [N, C_i]
                # Cast to float32: fp16 Linear matmul can overflow the fp16 max (65504)
                # when weight norms grow after training.  Inf input to _FP32LayerNorm
                # gives Inf - mean([Inf,...]) = NaN which cannot be recovered.
                raw = raw.float()
                proj = self.level_projs[i](raw)
                ms_feats.append(proj)              # [N, out_ch] (float32)

            cat_feats = torch.cat(ms_feats, dim=-1)
            normed = self.ms_norm(cat_feats)
            gathered = self.ms_proj(normed)
        else:
            # ── Fallback: single peak token on the assigned FPN level ──────────
            # Decode level + local (r, c) from feat_idx.
            level_sizes_t = torch.tensor(
                [h * w for h, w in level_hw], device=device, dtype=torch.long)
            offsets = torch.cat(
                [level_sizes_t.new_zeros(1), level_sizes_t.cumsum(0)[:-1]])
            H_t = torch.tensor([h for h, w in level_hw], device=device, dtype=torch.long)
            W_t = torch.tensor([w for h, w in level_hw], device=device, dtype=torch.long)
            obj_level = torch.bucketize(idxs, offsets, right=True) - 1
            local_idx = idxs - offsets[obj_level]
            obj_W = W_t[obj_level]
            r_i = (local_idx // obj_W).clamp(0)
            c_i = (local_idx %  obj_W)
            # Vectorised gather + project: group objects by their assigned level so
            # we can do a single batched gather + matmul per level (no Python loop).
            N = idxs.shape[0]
            out = x[0].new_zeros(N, self.out_channels)
            for lv_idx, feat_lv in enumerate(x):
                mask = (obj_level == lv_idx)
                if not mask.any():
                    continue
                bi_lv = batch_ids[mask]
                r_lv  = r_i[mask]
                c_lv  = c_i[mask]
                raw_fp16 = feat_lv[bi_lv, :, r_lv, c_lv]
                raw   = raw_fp16.float()   # [N_lv, C_lv] fp32: prevents fp16 overflow
                proj_out = self.level_projs[lv_idx](raw)
                out[mask] = proj_out
            gathered = out

        return gathered


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIFeatureExtractor")
class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, config, in_channels):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=config.MODEL.RESNETS.RES5_DILATION
        )

        self.pooler = pooler
        self.head = head
        self.out_channels = head.out_channels

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor")
class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg: DictConfig, in_channels, half_out=False, cat_all_levels=False):
        super(FPN2MLPFeatureExtractor, self).__init__()

        resolution = cfg.model.roi_box_head.pooler_resolution
        scales = cfg.model.roi_box_head.pooler_scales
        sampling_ratio = cfg.model.roi_box_head.pooler_sampling_ratio
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            in_channels=in_channels,
            cat_all_levels=cat_all_levels,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.model.roi_box_head.mlp_head_dim
        use_gn = cfg.model.roi_box_head.use_gn
        self.pooler = pooler
        self.fc6 = make_fc(cfg, input_size, representation_size, use_gn)

        if half_out:
            out_dim = int(representation_size / 2)
        else:
            out_dim = representation_size
        
        self.fc7 = make_fc(cfg, representation_size, out_dim, use_gn)
        self.resize_channels = input_size
        self.out_channels = out_dim

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x

    def forward_without_pool(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPNXconv1fcFeatureExtractor")
class FPNXconv1fcFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg: DictConfig, in_channels):
        super(FPNXconv1fcFeatureExtractor, self).__init__()

        resolution = cfg.model.roi_box_head.pooler_resolution
        scales = cfg.model.roi_box_head.pooler_scales
        sampling_ratio = cfg.model.roi_box_head.pooler_sampling_ratio
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        use_gn = cfg.model.roi_box_head.use_gn
        conv_head_dim = cfg.model.roi_box_head.conv_head_dim
        num_stacked_convs = cfg.model.roi_box_head.num_stacked_convs
        dilation = cfg.model.roi_box_head.dilation

        xconvs = []
        for ix in range(num_stacked_convs):
            xconvs.append(
                nn.Conv2d(
                    in_channels,
                    conv_head_dim,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    bias=False if use_gn else True
                )
            )
            in_channels = conv_head_dim
            if use_gn:
                xconvs.append(group_norm(cfg, in_channels))
            xconvs.append(nn.ReLU(inplace=True))

        self.add_module("xconvs", nn.Sequential(*xconvs))
        for modules in [self.xconvs,]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if not use_gn:
                        torch.nn.init.constant_(l.bias, 0)

        input_size = conv_head_dim * resolution ** 2
        representation_size = cfg.model.roi_box_head.mlp_head_dim
        self.fc6 = make_fc(cfg, input_size, representation_size, use_gn=False)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.xconvs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        return x


def make_roi_box_feature_extractor(cfg: DictConfig, in_channels, half_out=False, cat_all_levels=False):
    """
    Build ROI box feature extractor from config.
    
    Args:
        cfg: Hydra configuration
        in_channels: Number of input channels
        half_out: Whether to use half output dimension
        cat_all_levels: Whether to concatenate all FPN levels
        
    Returns:
        Feature extractor instance
    """
    # Resolve in_channels if it's a list (e.g. for YOLO multi-scale features)
    if hasattr(in_channels, '__getitem__') and not isinstance(in_channels, (torch.Tensor, str)):
        in_channels = int(in_channels[0])

    func = registry.ROI_BOX_FEATURE_EXTRACTORS[
        cfg.model.roi_box_head.feature_extractor
    ]
    return func(cfg, in_channels, half_out, cat_all_levels)
