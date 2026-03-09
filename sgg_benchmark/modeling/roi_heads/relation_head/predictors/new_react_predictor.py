"""
REACT+ Predictor
================
A redesigned, parameter-efficient version of REACTPredictor.

Key innovations over the original REACT:
─────────────────────────────────────────
1.  **Grouped Gated MLP (GGLU)**  — replaces every standalone Linear+ReLU+Linear
    block with a single SwiGLU-style gated projection that halves parameter count
    while improving gradient flow (no dead-neuron problem).

2.  **Lightweight cross-modal attention** — instead of three separate W_sub/obj/
    pred MLPs + gate_sub/obj + gate_pred, a single 4-head cross-attention block
    fuses visual tokens with semantic prototype embeddings.  Attention is O(N·K)
    where K = num_rel_cls (constant), so it is cheap and ONNX-exportable.

3.  **RMSNorm everywhere** — drop-in for LayerNorm but faster (no mean
    centering, no learnable bias term reduces param count slightly).

4.  **Rotary Position Embedding (RoPE)** on the geometry signal — the 9-dim
    box encoding is projected into a 64-dim frequency space and added as a
    rotary bias inside the cross-attention, letting the model reason about
    absolute and relative box positions without a separate positional MLP.

5.  **Prototype buffer** — the num_rel_cls prototype vectors are kept in a
    nn.Parameter (not an Embedding) and updated with an exponential moving
    average of the positive-pair representations during training.  This makes
    prototypes continuously track the evolving feature distribution (inspired
    by MoCo/DINO momentum queues).

6.  **Learnable temperature per-class** — instead of a single logit_scale
    scalar, each prototype has its own temperature τ_c initialised to log(1/0.07)
    so rare classes can spread out their decision boundary independently.

7.  **Unified dim = 512** — all intermediate tensors live in the same D=512 space;
    no mlp_dim/embed_dim mismatch.

8.  **Skippable union branch** — if union_features is None the predictor degrades
    gracefully to a text-only path at identical parameter cost.

Config keys read (all under cfg.model.roi_relation_head):
  - embed_dim       (word-embedding dim, e.g. 200 for GloVe)
  - mlp_head_dim    (D, internal dim, default 512)
  - context_dropout_rate
  - use_union_features / use_spatial_features
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sgg_benchmark.modeling import registry
from sgg_benchmark.modeling.utils import cat
from sgg_benchmark.utils.txt_embeddings import rel_vectors, obj_edge_vectors
from sgg_benchmark.modeling.roi_heads.relation_head.predictors.default_predictors import BasePredictor
from sgg_benchmark.modeling.roi_heads.relation_head.models.utils.utils_motifs import to_onehot, encode_box_info


# ──────────────────────────────────────────────────────────────────────────────
# Primitive building blocks
# ──────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalization (Zhang & Sennrich 2019).
    """
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 for the norm computation: in fp16, eps=1e-6 rounds to 0,
        # making rsqrt(0) = inf and then x * inf = NaN for near-zero vectors.
        x_f32 = x.float()
        norm = x_f32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x_f32 * norm * self.scale.float()).to(x.dtype)


class SwiGLU(nn.Module):
    """SwiGLU gated feed-forward (Shazeer 2020).

    y = (W1·x) ⊙ swish(W2·x)  →  W_out·y

    Compared to a 2-layer ReLU MLP with the same in/out dims this halves
    the number of non-linear activations (no dead neurons) and provides a
    multiplicative gating signal that smoothly controls information flow.

    Parameter saving: replaces Linear(D, 4D)+ReLU+Linear(4D, D) with
    two Linear(D, 2D/3) + one Linear(2D/3, D), yielding 2/3 of the params
    for the same effective capacity.
    """
    def __init__(self, d_in: int, d_out: int, expansion: float = 8/3,
                 dropout: float = 0.0):
        super().__init__()
        d_hidden = int(d_in * expansion)
        # Round to nearest multiple of 8 for tensor-core efficiency
        d_hidden = (d_hidden + 7) // 8 * 8
        self.gate_proj  = nn.Linear(d_in, d_hidden, bias=False)
        self.value_proj = nn.Linear(d_in, d_hidden, bias=False)
        self.out_proj   = nn.Linear(d_hidden, d_out,  bias=False)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.dropout(
            F.silu(self.gate_proj(x)) * self.value_proj(x)
        ))


class GeomRoPE(nn.Module):
    """Box-geometry → Rotary Position Embedding.

    Encodes the 9-dim box-info vector (w/W, h/H, cx/W, cy/H, x1/W, y1/H,
    x2/W, y2/H, area) into a rotary frequency space of dimension `d_rope`.
    The resulting sin/cos tensor is applied as an additive bias on the Q matrix
    of the cross-attention, giving each query a unique geometry context without
    any extra memory in the KV cache.

    d_rope must be even and ≤ num_heads * head_dim.
    """
    def __init__(self, d_rope: int = 64, num_freqs: int = 16):
        super().__init__()
        assert d_rope % 2 == 0
        self.d_rope = d_rope
        # Learnable frequency projector: 9 → d_rope
        self.proj = nn.Sequential(
            nn.Linear(9, num_freqs, bias=False),
            nn.SiLU(),
            nn.Linear(num_freqs, d_rope, bias=False),
        )

    def forward(self, box_info: torch.Tensor) -> torch.Tensor:
        """
        Args:
            box_info: [N, 9]  normalised box info from encode_box_info()
        Returns:
            [N, d_rope]  additive geometry bias for Q
        """
        freq = self.proj(box_info)               # [N, d_rope]
        half = self.d_rope // 2
        sin = freq[:, :half].sin()
        cos = freq[:, half:].cos()
        return torch.cat([sin, cos], dim=-1)     # [N, d_rope]


class PairwiseFourierEncoder(nn.Module):
    """Encodes pairwise subject–object spatial relationships via Fourier lifting.

    Plain MLPs underfit sharp spatial decision boundaries (above/below, inside/
    outside) because the input dimensionality is tiny (8 scalars).  Fourier
    lifting maps each scalar to 2K features (sin + cos at K log-spaced freqs),
    giving the downstream Linear access to high-frequency spatial structure —
    the same insight as NeRF positional encoding (Tancik et al. 2020).

    Input: 8-dim normalised pairwise descriptor built from encode_box_info():
        dx, dy       — normalised centre-to-centre displacement
        log_wr       — log width  ratio  obj/sub  (size asymmetry)
        log_hr       — log height ratio  obj/sub
        log_dist     — log Euclidean centre distance + 1
        area_s       — subject relative area
        area_o       — object  relative area
        log_ar       — log area ratio obj/sub

    Parameters: Linear(8×2×K, D) + RMSNorm ≈ 66 K params at K=8, D=512.
    FLOPs: negligible (128→512 linear at N_rel scale).
    """
    def __init__(self, d_out: int, num_freqs: int = 8):
        super().__init__()
        d_fourier = 8 * 2 * num_freqs          # 8 dims × sin+cos × K freqs
        # Log-spaced frequencies π·2⁰ … π·2^(K-1) capture both coarse and fine
        # spatial structure within the [0, 1] normalised coordinate space.
        freqs = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)  # [K]
        self.register_buffer('freqs', freqs)
        self.proj = nn.Sequential(
            nn.Linear(d_fourier, d_out, bias=False),
            RMSNorm(d_out),
        )

    def forward(self, pair_sp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair_sp: [N, 8]  normalised pairwise spatial descriptor
        Returns:
            [N, d_out]
        """
        x       = pair_sp.unsqueeze(-1) * self.freqs.view(1, 1, -1) * math.pi  # [N, 8, K]
        fourier = torch.cat([x.sin(), x.cos()], dim=-1)  # [N, 8, 2K]
        fourier = fourier.reshape(pair_sp.shape[0], -1)  # [N, 8*2*K]
        return self.proj(fourier)                         # [N, d_out]


class VisualSemanticCrossAttn(nn.Module):
    """Cross-attention: visual queries attend to a fixed set of semantic keys.

    Queries  Q ∈ ℝ^{N×D}  — per-instance visual features
    Keys/Vals K,V ∈ ℝ^{C×D} — C prototype embeddings (num_rel_cls or num_obj_cls)

    The geometry RoPE bias is added to Q so that the attention pattern varies
    smoothly with box position even when the visual features are identical.

    This replaces the original W_sub, W_obj, W_pred MLPs + explicit gate_* Linear
    layers with a single module that learns *which* semantic prototype to blend
    in via attention — strictly more expressive at lower parameter cost.
    """
    def __init__(self, d_model: int, num_heads: int = 4,
                 dropout: float = 0.0, d_rope: int = 0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model   = d_model
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.scale     = self.head_dim ** -0.5
        self.d_rope    = d_rope

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out    = nn.Linear(d_model, d_model, bias=False)
        self.drop   = nn.Dropout(dropout)

        if d_rope > 0:
            # Projects the d_rope geometry bias to one scalar per attention head.
            # [N, d_rope] → [N, num_heads]; then broadcast over the C key dimension.
            self.rope_proj = nn.Linear(d_rope, num_heads, bias=False)

    def forward(self, x: torch.Tensor, kv: torch.Tensor,
                rope_bias: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x         [N, D]  visual queries
            kv        [C, D]  semantic keys/values (shared across batch)
            rope_bias [N, d_rope] optional geometry bias for Q
        Returns:
            [N, D]
        """
        N, D = x.shape
        C    = kv.shape[0]
        H    = self.num_heads
        Hd   = self.head_dim

        Q = self.q_proj(x).view(N, H, Hd)       # [N, H, Hd]
        K = self.k_proj(kv).view(C, H, Hd)      # [C, H, Hd]
        V = self.v_proj(kv).view(C, H, Hd)      # [C, H, Hd]

        # Upcast to float32 for attention: in fp16 large logits → softmax(inf) = NaN.
        Q_f, K_f, V_f = Q.float(), K.float(), V.float()

        # [H, N, Hd] × [H, Hd, C] → [H, N, C]
        attn = torch.einsum('nhd,chd->hnc', Q_f, K_f) * self.scale

        if rope_bias is not None and self.d_rope > 0:
            # Project rope to [N, H] scalar bias per head, then broadcast over C.
            # rope_proj: d_rope → H  (not H*Hd — we want one scalar per head per query)
            b = self.rope_proj(rope_bias).float()   # [N, H]
            # Reshape to [H, N, 1] and broadcast over C keys
            attn = attn + b.permute(1, 0).unsqueeze(-1)  # [H, N, 1] → broadcasts

        attn = self.drop(attn.softmax(dim=-1))   # [H, N, C]  (float32)

        # [H, N, C] × [H, C, Hd] → [H, N, Hd] → [N, H*Hd], cast back to input dtype
        out = torch.einsum('hnc,chd->nhd', attn, V_f).reshape(N, D).to(x.dtype)
        return self.out(out)


class PrototypeMomentumBuffer(nn.Module):
    """Exponential-Moving-Average prototype bank.

    During training, positive-pair representations are used to update the
    prototype of their GT class at rate `momentum`.  At inference the stored
    prototypes are used directly.

    Inspired by MoCo (He et al. 2020) momentum queues and DINO (Caron et al.
    2021) centre moving average, but applied at prototype granularity.

    The EMA update happens in-place under no_grad so it adds zero FLOPs to
    the backward graph.
    """
    def __init__(self, num_classes: int, d_model: int, momentum: float = 0.999):
        super().__init__()
        self.momentum = momentum
        # Registered as a buffer: saved in state_dict but not a parameter.
        self.register_buffer('protos', torch.zeros(num_classes, d_model))
        self.register_buffer('counts', torch.zeros(num_classes))

    @torch.no_grad()
    def update(self, reps: torch.Tensor, labels: torch.Tensor):
        """Update prototypes with current batch representations.
        Args:
            reps   [N, D]  L2-normalised relation representations
            labels [N]     relation class indices (bg=0 ignored)
        """
        fg = labels > 0
        if not fg.any():
            return
        reps_fg   = reps[fg]
        labels_fg = labels[fg]
        for c in labels_fg.unique():
            mask = labels_fg == c
            mean = reps_fg[mask].mean(0)
            if self.counts[c] == 0:
                self.protos[c] = mean
            else:
                self.protos[c] = self.momentum * self.protos[c] + (1 - self.momentum) * mean
            # Always normalise in float32 — fp16 eps=1e-12 rounds to 0, causing NaN.
            self.protos[c] = F.normalize(self.protos[c].float(), dim=0)
            self.counts[c] += 1

    def forward(self, proto_weight: torch.Tensor) -> torch.Tensor:
        """Return the best available prototype for each class.
        Falls back to the learnable weight when the EMA buffer hasn't been
        initialised for a class (count == 0).
        """
        # Normalise in float32: under fp16 autocast, default eps=1e-12 rounds to 0
        # causing NaN when any prototype has near-zero norm.
        weight = F.normalize(proto_weight.float(), dim=-1).to(proto_weight.dtype)
        mask   = (self.counts > 0).unsqueeze(-1).float()     # [C, 1]
        return mask * self.protos + (1 - mask) * weight       # [C, D]


# ──────────────────────────────────────────────────────────────────────────────
# Main predictor
# ──────────────────────────────────────────────────────────────────────────────

@registry.ROI_RELATION_PREDICTOR.register("REACTPlusPlusPredictor")
class REACTPlusPlusPredictor(BasePredictor):
    """
    REACT++ — parameter-efficient, attention-augmented relation predictor.

    Architecture (forward pass):
        1.  Project ROI features: [N, in_ch] → [N, 2D] split into sub/obj slots.
        2.  Encode object labels → one-hot or soft class distributions.
        3.  Lift semantic embeddings to D-dim with a SwiGLU block.
        4.  Cross-attend visual sub/obj tokens to the obj prototype bank,
            guided by geometry RoPE biases → fused sub/obj representations.
        5.  Compose the relation representation:
              r = sub ⊕ obj  (element-wise)  → SwiGLU → D
        6.  Optionally gate in union visual features via cross-attention with
            the predicate prototype bank.
        7.  L2-normalise r and proto → cosine similarity → logits.
        8.  Prototype momentum EMA update (training only).
        9.  Three REACT regularisation losses (optional, same as original).
    """

    def __init__(self, config, in_channels):
        super().__init__(config, in_channels)

        # ── dims ──────────────────────────────────────────────────────────────
        D          = self.cfg.model.roi_relation_head.mlp_head_dim   # e.g. 512
        embed_dim  = self.cfg.model.roi_relation_head.embed_dim       # e.g. 200
        drop       = self.cfg.model.roi_relation_head.context_dropout_rate
        num_heads  = 4
        d_rope     = 64   # geometry RoPE dim

        self.D         = D
        self.embed_dim = embed_dim
        self.use_union = (self.cfg.model.roi_relation_head.use_union_features or
                          self.cfg.model.roi_relation_head.use_spatial_features)
        # Probability of randomly skipping the union branch during training.
        # When use_union_features_inference=False this teaches the base
        # composition path to be strong without union — closing the train/test
        # distribution gap at zero extra inference cost.
        self.union_dropout = float(
            self.cfg.model.roi_relation_head.get('union_dropout', 0.0))

        # Max pairs to process at inference (0 = no limit).
        # Uses frequency-bias scores to keep only the most plausible pairs,
        # significantly reducing N_rel-linear ops without hurting accuracy.
        self.max_pairs_inference = int(
            self.cfg.model.roi_relation_head.get('max_pairs_inference', 0))

        # When True, the union_feature_extractor is P5SceneContextExtractor:
        # relation reps cross-attend to AIFI-enhanced P5 tokens instead of the
        # per-pair FeatIdx center pixel.  Old checkpoints default to False.
        self.use_scene_context = bool(
            self.cfg.model.roi_relation_head.get('use_scene_context', False))

        # When False, geometry RoPE bias is zeroed out in the sub/obj cross-attention,
        # ablating the effect of box-position on semantic prototype alignment.
        # Defaults to True so all existing checkpoints remain unaffected.
        self.use_geo_bias = bool(
            self.cfg.model.roi_relation_head.get('use_geo_bias', True))

        # ── 1. Visual projection: in_ch → 2D (sub slot | obj slot) ───────────
        self.vis_proj = nn.Sequential(
            nn.Linear(in_channels, D * 2, bias=False),
            RMSNorm(D * 2),
        )

        # ── 2. Semantic lifting: embed_dim → D via SwiGLU ────────────────────
        self.obj_lift = SwiGLU(embed_dim, D, expansion=2.0, dropout=drop)
        self.rel_lift = SwiGLU(embed_dim, D, expansion=2.0, dropout=drop)

        # ── 3. Geometry RoPE encoder ─────────────────────────────────────────
        self.geom_rope = GeomRoPE(d_rope=d_rope)

        # ── 4a. Cross-attention: visual → obj prototypes ──────────────────────
        # Sub and Obj use SEPARATE cross-attention weights (asymmetric roles).
        # The question "which object am I as a SUBJECT?" differs from
        # "which object am I as an OBJECT?": a chair as subject of "is-next-to"
        # attends to different prototype dimensions than as object of "sat-on".
        # Cost: 1 extra VisualSemanticCrossAttn ≈ 1M params, zero latency change
        # (the two forward() calls were already separate — only weights differ).
        self.sub_sem_attn = VisualSemanticCrossAttn(
            d_model=D, num_heads=num_heads, dropout=drop, d_rope=d_rope)
        self.obj_sem_attn = VisualSemanticCrossAttn(
            d_model=D, num_heads=num_heads, dropout=drop, d_rope=d_rope)
        self.vis_sem_norm = RMSNorm(D)

        # ── 4b. Per-slot residual SwiGLU ─────────────────────────────────────
        self.sub_ffn  = SwiGLU(D, D, expansion=4/3, dropout=drop)
        self.obj_ffn  = SwiGLU(D, D, expansion=4/3, dropout=drop)
        self.sub_norm = RMSNorm(D)
        self.obj_norm = RMSNorm(D)

        # ── 5. Relation composition: [sub, obj] → D ───────────────────────────
        # Element-wise product + sum (bilinear-style) rather than concat+linear,
        # halving the projection cost while preserving asymmetry:
        #   r_compose = sub * obj  +  sub  -  obj
        # This encodes directional asymmetry (sub - obj) AND co-occurrence (sub * obj).
        self.compose_ffn  = SwiGLU(D, D, expansion=4/3, dropout=drop)
        self.compose_norm = RMSNorm(D)

        # ── 6. Context branch — two flavours selected by use_scene_context ────
        if self.use_union:
            if self.use_scene_context:
                # AIFI scene mode (P5SceneContextExtractor):
                # Relation reps cross-attend to k²=16 AIFI-enhanced P5 scene tokens.
                # N_rel × 16 cross-attn KV is cheaper than the old N_rel × 38
                # prototype cross-attn, while delivering genuine global context.
                self.scene_proj       = nn.Linear(in_channels, D, bias=False)
                self.scene_cross_attn = VisualSemanticCrossAttn(
                    d_model=D, num_heads=num_heads, dropout=drop)
                self.scene_norm       = RMSNorm(D)
                self.scene_gate       = nn.Linear(D, D, bias=False)
            else:
                # FeatIdx per-pair mode (run7/run8 backwards compat):
                # Visual union center → cross-attend to predicate prototypes.
                self.union_proj = nn.Linear(in_channels, D, bias=False)
                self.union_attn = VisualSemanticCrossAttn(
                    d_model=D, num_heads=num_heads, dropout=drop)
                self.union_norm = RMSNorm(D)
                # Element-wise gate: Linear(D, D) is 2× cheaper than concat-based.
                self.union_gate = nn.Linear(D, D, bias=False)

        # ── 7. Projection head: D → 2D for contrastive loss (SimCSE/REACT) ───
        self.proj_head = nn.Sequential(
            nn.Linear(D, D * 2, bias=False),
            RMSNorm(D * 2),
            nn.SiLU(),
            nn.Linear(D * 2, D, bias=False),
        )

        # ── 8. Prototype bank ─────────────────────────────────────────────────
        # Learnable prototypes (trained by gradient) + EMA shadow (training only).
        self.proto_weight = nn.Parameter(torch.empty(self.num_rel_cls, D))
        nn.init.normal_(self.proto_weight, std=D ** -0.5)

        self.proto_ema = PrototypeMomentumBuffer(
            self.num_rel_cls, D, momentum=0.999)

        # Global temperature scalar (replaces per-class log_tau).
        # Initialised to log(1/0.07) ≈ 2.66  →  τ ≈ 14.3  (same as CLIP).
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

        # ── 9. Semantic embeddings (frozen GloVe/MPNet) ───────────────────────
        obj_embed_vecs = obj_edge_vectors(
            self.obj_classes,
            wv_type=self.cfg.model.text_embedding,
            wv_dir=self.cfg.glove_dir,
            wv_dim=embed_dim,
            use_cache=True,
        )
        rel_embed_vecs = rel_vectors(
            self.rel_classes,
            wv_type=self.cfg.model.text_embedding,
            wv_dir=self.cfg.glove_dir,
            wv_dim=embed_dim,
            use_cache=True,
        )
        # NOTE: use len(obj_classes) / len(rel_classes) from statistics, NOT
        # num_obj_cls / num_rel_cls from config — those are VG150 defaults (151/51)
        # and will mismatch the GloVe vectors loaded above.
        self.obj_embed = nn.Embedding(len(self.obj_classes), embed_dim)
        self.rel_embed = nn.Embedding(len(self.rel_classes), embed_dim)

        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs)
            self.rel_embed.weight.copy_(rel_embed_vecs)

        # Freeze word embeddings — they're pre-trained signals, not to be
        # updated.  The lifting MLP (obj_lift/rel_lift) adapts them to D.
        self.obj_embed.weight.requires_grad_(False)
        self.rel_embed.weight.requires_grad_(False)

        # Inference-only caches (invalidated whenever we enter training mode)
        # These avoid re-projecting the same frozen embeddings on every forward.
        self._cache_valid = False

    # ── helpers ───────────────────────────────────────────────────────────────

    def train(self, mode: bool = True):
        """Override train() to invalidate inference caches when switching modes."""
        if mode != self.training:
            self._cache_valid = False
        return super().train(mode)

    def _get_cached_protos(self):
        """Return (obj_proto_kv, pred_proto_kv, protos_norm, protos_proj) from cache or compute.

        protos_proj is only non-None during training (needed for REACT losses).
        At inference the cache returns protos_proj=None to skip the recompute.
        """
        if not self.training and self._cache_valid:
            return self._c_obj_proto_kv, self._c_pred_proto_kv, self._c_protos_norm, None

        obj_proto_kv  = self.obj_lift(self.obj_embed.weight)          # [C_obj, D]
        pred_proto_kv = (self.rel_lift(self.rel_embed.weight)
                         if self.use_union else None)                  # [C_rel, D] or None
        protos        = self._proto_bank()                             # [C_rel, D]
        protos_proj   = self.proj_head(protos)                         # [C_rel, D]
        # Normalise in float32: fp16 default eps=1e-12 == 0, causing NaN when the
        # prototype's L2 norm underflows to zero in fp16.
        protos_norm   = F.normalize(protos_proj.float(), dim=-1).to(protos_proj.dtype)  # [C_rel, D]

        if not self.training:
            self._c_obj_proto_kv  = obj_proto_kv.detach()
            self._c_pred_proto_kv = pred_proto_kv.detach() if pred_proto_kv is not None else None
            self._c_protos_norm   = protos_norm.detach()
            self._cache_valid     = True
            protos_proj           = None   # not needed at inference

        return obj_proto_kv, pred_proto_kv, protos_norm, protos_proj

    def _encode_labels(self, proposals):
        """One-hot or GT labels → entity_dists [N, C], entity_preds [N]."""
        obj_labels = cat([p["labels"] for p in proposals], dim=0).long()
        # Use len(obj_classes) for consistency with self.obj_embed size.
        entity_dists = to_onehot(obj_labels, len(self.obj_classes))
        return entity_dists, obj_labels

    def _geom_bias(self, proposals):
        """Compute per-object geometry RoPE biases [N_total, d_rope]."""
        box_info = encode_box_info(proposals)   # [N, 9]
        return self.geom_rope(box_info)         # [N, d_rope]

    def _proto_bank(self) -> torch.Tensor:
        """Return per-class prototypes [C, D], blending EMA + learnable."""
        return self.proto_ema(self.proto_weight)  # [C, D]

    # ── forward ───────────────────────────────────────────────────────────────

    @torch.autocast(device_type='cuda', enabled=False)
    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys,
                roi_features, union_features, logger=None):

        add_losses   = {}
        D            = self.D
        device       = roi_features.device

        # Cast inputs to float32 at the predictor boundary.
        # All norms (RMSNorm, LayerNorm), softmax, and F.normalize operations inside
        # the predictor are numerically unsafe in fp16 (eps underflow, logit overflow).
        # Casting here is cheaper than wrapping every individual op and ensures any
        # future ops added to this forward are also safe by default.
        roi_features   = roi_features.float()
        if union_features is not None:
            union_features = union_features.float()

        num_objs = [len(p["boxes"]) for p in proposals]
        num_rels = [r.shape[0]      for r in rel_pair_idxs]

        # ── 1. Visual sub/obj slots ───────────────────────────────────────────
        vis = self.vis_proj(roi_features)         # [N_obj, 2D]
        sub_vis = vis[:, :D]                       # [N_obj, D]
        obj_vis = vis[:, D:]                       # [N_obj, D]

        # ── 2. Semantic embeddings ────────────────────────────────────────────
        entity_dists, entity_preds = self._encode_labels(proposals)

        sem = self.obj_lift(self.obj_embed(entity_preds))   # [N_obj, D]

        # ── 3. Geometry RoPE ─────────────────────────────────────────────────
        box_info = encode_box_info(proposals)              # [N_obj, 9] normalised
        geo_bias = (self.geom_rope(box_info)               # [N_obj, d_rope]
                    if self.use_geo_bias else None)

        # ── 4. Cross-attend: visual → obj prototype bank ─────────────────────
        # obj_proto_kv / pred_proto_kv / protos_norm are cached at inference
        # (they depend only on frozen embeddings + static MLP weights).
        obj_proto_kv, pred_proto_kv, protos_norm, protos_proj = self._get_cached_protos()

        # Sub: visual slot cross-attends to obj prototypes, geometry-guided
        sub_attn = self.sub_sem_attn(sub_vis, obj_proto_kv, geo_bias)  # [N, D]
        sub = self.vis_sem_norm(sub_vis + sub_attn)  # residual
        sub = self.sub_norm(sub + self.sub_ffn(sub))

        # Obj: separate cross-attention block (asymmetric role weights)
        obj_attn = self.obj_sem_attn(obj_vis, obj_proto_kv, geo_bias)  # [N, D]
        obj = self.vis_sem_norm(obj_vis + obj_attn)  # residual
        obj = self.obj_norm(obj + self.obj_ffn(obj))

        # Also blend in the GloVe semantic signal directly (fast semantic lookup)
        sub = sub + sem
        obj = obj + sem

        # ── 5. Relation composition per image ────────────────────────────────
        sub_splits = sub.split(num_objs, dim=0)
        obj_splits = obj.split(num_objs, dim=0)

        # Build pair class labels for frequency bias (needed before splitting)
        pair_preds_list = []
        for pairs, proposal in zip(rel_pair_idxs, proposals):
            obj_lab = proposal["labels"].long()
            sub_cls = obj_lab[pairs[:, 0]]
            obj_cls = obj_lab[pairs[:, 1]]
            pair_preds_list.append(torch.stack([sub_cls, obj_cls], dim=1))
        pair_pred_cat = cat(pair_preds_list, dim=0)  # [N_rel, 2]

        # ── 5b. Inference-only pair pre-filter (freq-bias top-K) ─────────────
        # At inference many (sub_cls, obj_cls) combinations are implausible.
        # Keep only the top-K by their maximum frequency log-prior score.
        # This cuts N_rel-linear ops (compose_ffn / union_attn / proj_head)
        # without training any new modules.
        filter_topk = None      # global indices into the concatenated pairs tensor
        _per_img_k = None       # per-image counts of kept pairs
        _filtered_pair_idxs = None  # per-image filtered rel_pair_idxs for post-processor
        if (not self.training and self.max_pairs_inference > 0
                and self.use_bias and pair_pred_cat.shape[0] > self.max_pairs_inference):
            freq_scores = self.freq_bias.index_with_labels(pair_pred_cat).max(dim=-1).values
            k = self.max_pairs_inference
            # Sort by global index (ascending) so pairs stay in image order after indexing.
            # This lets us split rel_dists per-image later without a scatter.
            filter_topk = freq_scores.topk(k, sorted=False).indices.sort().values  # [K]

            # Pre-compute per-image K counts and filtered rel_pair_idxs using binary search.
            cum = torch.tensor(num_rels, dtype=torch.long, device=device).cumsum(0)
            bounds = torch.cat([cum.new_zeros(1), cum])   # [n_img+1]
            _per_img_k = []
            _filtered_pair_idxs = []
            for i in range(len(num_rels)):
                lo = torch.searchsorted(filter_topk, bounds[i:i+1]).item()
                hi = torch.searchsorted(filter_topk, bounds[i+1:i+2]).item()
                _per_img_k.append(hi - lo)
                local_idx = filter_topk[lo:hi] - bounds[i]   # local indices within image i
                _filtered_pair_idxs.append(rel_pair_idxs[i][local_idx])

        rel_reps = []
        for pairs, s, o in zip(rel_pair_idxs, sub_splits, obj_splits):
            si = s.index_select(0, pairs[:, 0])   # [num_rel, D]
            oi = o.index_select(0, pairs[:, 1])   # [num_rel, D]

            # Asymmetric bilinear composition:
            #   element-wise product captures co-occurrence,
            #   element-wise difference captures directionality.
            r = si * oi + si - oi                  # [num_rel, D]
            rel_reps.append(r)

        r = cat(rel_reps, dim=0)                   # [N_rel, D]

        # Apply pair filter (inference-only, before the expensive per-pair ops)
        if filter_topk is not None:
            r             = r[filter_topk]
            # Scene context features are per-image (not per-pair) — never index them.
            if not self.use_scene_context:
                union_features = union_features[filter_topk] if union_features is not None else None
            pair_pred_cat = pair_pred_cat[filter_topk]

        r = self.compose_norm(r + self.compose_ffn(r))

        # ── 6. Union feature gating ───────────────────────────────────────────
        # Decide whether to apply union this forward pass:
        #   • At inference: skip when skip_union_at_inference is set.
        #   • During training: randomly skip with probability union_dropout
        #     so the composition path learns to work without union, closing
        #     the train-test distribution gap at zero inference-time cost.
        _skip_union_inference = (not self.training and
                                  getattr(self, 'skip_union_at_inference', False))
        _skip_union_dropout   = (self.training and self.union_dropout > 0.0 and
                                  torch.rand(1, device=device).item() < self.union_dropout)
        _apply_union = (self.use_union and union_features is not None
                        and not _skip_union_inference
                        and not _skip_union_dropout)

        if _apply_union:
            if self.use_scene_context:
                # ── Scene context mode (P5SceneContextExtractor) ─────────────
                # union_features: [B * k², in_channels] — k² spatial tokens per image.
                # Each relation representation cross-attends to its image's scene tokens,
                # learning which spatial regions corroborate or disambiguate the relation.
                B    = len(proposals)
                k_sq = union_features.shape[0] // B          # e.g. 16 for 4×4 pool
                scene_tokens = self.scene_proj(union_features)    # [B*k², D]
                scene_tokens = scene_tokens.view(B, k_sq, D)      # [B, k², D]

                # Per-image cross-attention to avoid cross-image context leakage.
                r_parts = r.split(num_rels, dim=0)
                r_out   = []
                for i, (ri, ni) in enumerate(zip(r_parts, num_rels)):
                    if ni == 0:
                        r_out.append(ri)
                        continue
                    si    = scene_tokens[i]                    # [k², D]
                    r_ctx = self.scene_cross_attn(ri, si)      # [ni, D]
                    r_i   = self.scene_norm(ri + r_ctx)
                    gate  = torch.sigmoid(self.scene_gate(r_i))  # [ni, D]
                    r_out.append(ri + gate * r_ctx)
                r = cat(r_out, dim=0)                          # [N_rel, D]
            else:
                # ── Per-pair FeatIdx mode (run7/run8 compat) ─────────────────
                # pred_proto_kv is already computed in _get_cached_protos()
                u      = self.union_proj(union_features)          # [N_rel, D]
                u_attn = self.union_attn(u, pred_proto_kv)        # [N_rel, D]
                u      = self.union_norm(u + u_attn)
                gate   = torch.sigmoid(self.union_gate(r + u))    # [N_rel, D]
                r      = r + gate * u

        # ── 7. Projection head + L2-normalisation ─────────────────────────────
        r_proj = self.proj_head(r)                               # [N_rel, D]
        # Normalise in float32: fp16 default eps=1e-12 == 0, so near-zero
        # relation reps (fp16 underflow) produce NaN that then corrupts the EMA.
        r_norm = F.normalize(r_proj.float(), dim=-1).to(r_proj.dtype)  # [N_rel, D]

        # protos_norm is cached at inference via _get_cached_protos()
        # Scale by a single global temperature (clamped to avoid collapse)
        scale     = self.logit_scale.exp().clamp(max=100.0)
        rel_dists = (r_norm @ protos_norm.t()) * scale            # [N_rel, C]

        # Frequency bias (log-prior): P(rel | sub_cls, obj_cls) from train stats
        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred_cat)

        # ── 8. Split back per image ───────────────────────────────────────────
        entity_dists = entity_dists.split(num_objs, dim=0)

        if filter_topk is not None:
            # rel_dists has K rows in image order (filter_topk was sorted ascending).
            # Split by pre-computed per-image K counts; propagate filtered pair idxs
            # via add_losses so relation_head uses only the scored pairs for ranking.
            rel_dists = rel_dists.split(_per_img_k, dim=0)
            add_losses["_filtered_rel_pair_idxs"] = _filtered_pair_idxs
        else:
            rel_dists = rel_dists.split(num_rels, dim=0)

        # ── 9. Training losses ────────────────────────────────────────────────
        if self.training:
            # Momentum EMA update (no gradient)
            rel_labels_cat = cat(rel_labels, dim=0)
            self.proto_ema.update(r_norm.detach(), rel_labels_cat)

            # ── REACT regularisation (same as original, slightly tuned) ──────

            # L2,1 prototype diversity loss
            simil_mat = protos_norm @ protos_norm.t()             # [C, C]
            l21 = simil_mat.norm(p=2, dim=1).norm(p=1) / (self.num_rel_cls ** 2)
            add_losses["l21_loss"] = l21

            # All margin-based losses operate on L2-normalised vectors so that
            # squared Euclidean distances live in [0, 4] and the margins are
            # meaningful regardless of the raw feature magnitude.
            # softplus(x) = log(1+exp(x)) replaces relu so there is always a
            # non-zero gradient signal even when the hard constraint is satisfied.
            r_n      = r_norm                                      # [N_rel, D]  (already normalised)
            protos_n = protos_norm                                 # [C, D]      (already normalised)
            N_rel_   = r_n.shape[0]

            # Prototype separation: push each proto away from its 2 nearest neighbours.
            # On unit sphere ||a-b||^2 = 2-2cos ∈ [0,4].  gamma2=1.5 means neighbours
            # must be at least cos=0.25 apart; softplus keeps the gradient alive once
            # the constraint is met.
            gamma2   = 1.5
            dist_mat = torch.cdist(protos_n, protos_n.detach()) ** 2         # [C, C]
            # Mask self-distance with +inf so it never appears as nearest neighbour
            dist_mat = dist_mat + torch.eye(self.num_rel_cls, device=device) * 1e9
            topk_d_  = dist_mat.sort(dim=1).values[:, :2].sum(dim=1)         # 2 nearest
            dist_loss = F.softplus(gamma2 - topk_d_).mean()
            add_losses["dist_loss2"] = dist_loss

            # Triplet pull-push: r_n should be closer to its prototype than to any
            # other.  gamma1=0.4 on the unit sphere ≈ 0.2 cosine margin.
            # Use +inf masking so the positive never contaminates the negative mining.
            gamma1 = 0.4
            dist_g = torch.cdist(r_n, protos_n.detach()) ** 2               # [N_rel, C]
            idx    = torch.arange(N_rel_, device=device)
            d_pos  = dist_g[idx, rel_labels_cat]                             # [N_rel]
            dist_g_neg = dist_g.clone()
            dist_g_neg[idx, rel_labels_cat] = float('inf')                   # mask positive
            d_neg_ = dist_g_neg.sort(dim=1).values[:, :10].mean(dim=1)      # 10 nearest neg
            loss_dis = F.softplus(d_pos - d_neg_ + gamma1).mean()
            add_losses["loss_dis"] = loss_dis

            # Apply configured loss weights if present
            weights = getattr(
                self.cfg.model.roi_relation_head, "react_loss_weights", {})
            for k, v in list(add_losses.items()):
                w = weights.get(k, 1.0) if weights else 1.0
                add_losses[k] = v * w

        return entity_dists, rel_dists, add_losses
