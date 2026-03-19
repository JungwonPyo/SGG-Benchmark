# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from typing import Optional, Literal

class EdgeDensityLoss(nn.Module):
    """
    Based on 
    [1] B. Knyazev, H. de Vries, C. Cangea, G.W. Taylor, A. Courville, E. Belilovsky.
    Graph Density-Aware Losses for Novel Compositions in Scene Graph Generation. BMVC 2020.
    https://arxiv.org/abs/2005.08230
    """

    def __init__(self, loss_weight=1.0):
        super(EdgeDensityLoss, self).__init__()
        self.loss_weight = loss_weight
        # Note: loss_weight is not currently used in forward(), kept for compatibility

    def forward(self, input, target):
        # Compute base cross entropy loss with reduction='none' to get per-sample losses
        loss = F.cross_entropy(input, target, reduction='none')

        # Get foreground and background indices (detach to prevent gradient tracking)
        with torch.no_grad():
            idx_fg = (target > 0).nonzero(as_tuple=False).view(-1)
            idx_bg = (target == 0).nonzero(as_tuple=False).view(-1)

            M_FG = idx_fg.numel()
            M_BG = idx_bg.numel()
            M = input.size(0)

            # Create edge weights on the same device as input (no gradients needed)
            edge_weights = torch.ones(M, dtype=input.dtype, device=input.device)
            
            if M_FG > 0:
                edge_weights[idx_fg] = 1.0 / M_FG
                if M_BG > 0:
                    edge_weights[idx_bg] = 1.0 / M_FG

        # Apply weights and sum (edge_weights has no gradients)
        weighted_loss = loss * edge_weights
        return weighted_loss.sum()

class BCEWithLogitsIgnoreNegative(nn.BCEWithLogitsLoss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        reduction: Literal["none", "mean", "sum"] = "mean",
        pos_weight: Optional[Tensor] = None,
        collect_stats: Optional[int] = None,
    ) -> None:
        """
        :param collect_stats: Set this parameter to the number of target classes to count the
        number of positives/negatives until .reset_stats() is called.
        """
        assert reduction in ("none", "mean", "sum")
        super().__init__(weight=weight, reduction="none", pos_weight=pos_weight)
        self.my_reduction = reduction
        self.pos_count_stats = None
        self.neg_count_stats = None
        if collect_stats is not None:
            self.pos_count_stats = torch.zeros(collect_stats, dtype=torch.long)
            self.neg_count_stats = torch.zeros(collect_stats, dtype=torch.long)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Handle 1D targets (indices) by converting to one-hot
        if target.dim() == 1:
            num_classes = input.size(-1)
            # -1 or other negative values indicate "ignore sample"
            # We'll handle this by creating a 2D mask later
            mask_1d = (target >= 0).unsqueeze(1).expand_as(input)
            
            # Convert to one-hot, clamping negatives to 0 for one_hot then masking
            target_clamped = torch.clamp(target, min=0).long()
            target = F.one_hot(target_clamped, num_classes).to(input.dtype)
            mask = mask_1d
        else:
            # For 2D targets, negative values indicate "ignore this specific (sample, class)"
            mask = target >= 0
            target = target.float()

        if self.pos_count_stats is not None:
            # Only count if not ignored
            self.pos_count_stats += ((target == 1) & mask).sum(0)
            self.neg_count_stats += ((target == 0) & mask).sum(0)

        # Compute raw BCE loss
        # Note: we use super().forward with the float target
        # We must handle the fact that super().forward might fail if target has negatives,
        # so we pass target * mask which converts -1 to 0 (but they are masked anyway)
        loss = F.binary_cross_entropy_with_logits(input, target, weight=self.weight, pos_weight=self.pos_weight, reduction='none')
        loss = loss * mask

        if self.my_reduction == "none":
            return loss
        if self.my_reduction == "sum":
            return loss.sum()
        if self.my_reduction == "mean":
            # Normalize by the number of non-ignored entries
            return loss.sum() / torch.clamp(mask.sum(), min=1.0)
        raise RuntimeError()

    def reset_stats(self):
        if self.pos_count_stats is not None:
            self.pos_count_stats[:] = 0
            self.neg_count_stats[:] = 0

class BalancedLogitAdjustedLoss(nn.Module):
    """
    Combines Logit Adjustment (LA) with Focal Loss characteristics,
    Foreground/Background balancing, and Empirical Soft Supervision.

    Why this works:
    1. LA: Fixes the long-tail bias mathematically.
    2. Focal: Stops easy background examples from overwhelming gradients.
    3. Empirical Soft Supervision: For BG-labeled pairs whose (S,O) category
       combination appears in the training fg_matrix, replaces the hard BG label
       with a soft mixture of the BG label and the empirical P(FG_rel | S, O)
       distribution. This is structured label smoothing — instead of punishing
       the model for predicting a plausible relation, we teach it *which* relations
       are historically likely for this object-pair category.

       Unlike the old passive discounting (which only fired when the model was
       already confused and primarily helped head classes), this fires for *all*
       BG-labeled pairs with evidence in fg_matrix, and adds a positive supervisory
       signal before the model has a chance to over-commit to BG.

       Mixing weight: λ_eff = bg_discount × confidence(pair_total)
       confidence uses a Bayesian-style prior: pair_total / (pair_total + 10)
       → low-evidence pairs (rarely annotated) stay close to the hard BG label.
    """
    def __init__(self, pred_freq, gamma=2.0, alpha=0.4, tau=0.0, fg_boost=1.2,
                 fg_matrix=None, bg_discount=0.5):
        super(BalancedLogitAdjustedLoss, self).__init__()
        
        # --- 1. Frequency Statistics ---
        if not isinstance(pred_freq, torch.Tensor):
            pred_freq = torch.tensor(pred_freq, dtype=torch.float32)
        
        pred_freq = torch.nan_to_num(pred_freq, nan=1e-6)
        pred_freq = torch.clamp(pred_freq, min=1e-12)
        
        # P(c) class prior (used by LA when tau > 0)
        self.register_buffer('priors', pred_freq / (pred_freq.sum() + 1e-12))

        # --- Per-class CB gradient rebalancing (inverse-sqrt frequency) ---
        # LA shifts the decision boundary logit-wise; CB weights shift gradient magnitude.
        # Together they are complementary: LA says "the model needs higher confidence for
        # rare classes", CB says "rare-class samples get proportionally larger gradients".
        # Formula: w_c = 1/sqrt(freq_c), fg classes normalized to unit mean, capped at 10×.
        # bg class (index 0) is excluded from fg normalization and left at 1.0 so that
        # alpha-weighting (set in forward) continues to control the fg/bg split.
        pcw = 1.0 / (pred_freq.clamp(min=1.0) ** 0.5)
        pcw[0] = 1.0  # bg: CB weighting is irrelevant (bg handled by alpha)
        fg_mean = pcw[1:].mean().clamp(min=1e-6)
        pcw[1:] = (pcw[1:] / fg_mean).clamp(max=3.0)   # fg unit-mean, hard cap at 3×
        # Tighter than old 10× cap: max_per_sample = CE×focal×fg_boost×pcw_cap;
        # 3× keeps worst-case bounded while still giving rare classes 3× more gradient.
        self.register_buffer('per_class_weight', pcw)

        # fg_matrix shape: [num_obj_classes, num_obj_classes, num_rel_classes]
        if fg_matrix is not None:
            if not isinstance(fg_matrix, torch.Tensor):
                fg_matrix = torch.as_tensor(fg_matrix)
            # Log-transform counts to dampen extreme values (e.g., "on" vs "parked on")
            # We normalize this to [0, 1] range per subject-object pair later if needed, 
            # but usually raw presence is enough.
            self.register_buffer("fg_matrix", fg_matrix.float())
        else:
            self.fg_matrix = None

        # Hyperparameters
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau          # LA strength (0 = disabled)
        self.fg_boost = fg_boost
        self.bg_discount = bg_discount
        # Per-sample CE ceiling: log(C) is the max CE for a C-class uniform distribution.
        # Capping at 1.5× gives generous headroom for genuinely hard samples while bounding
        # outliers (mislabeled pairs, numerical edge cases, very hard early-training steps).
        self.max_ce_clip = 1.5 * math.log(max(2, int(len(pred_freq))))

    def get_empirical_soft_target(self, s_idx, o_idx, num_classes, device):
        """
        Build a soft target distribution from the empirical P(FG_rel | S, O) stored
        in fg_matrix.  The hard BG one-hot is mixed with the dataset prior using a
        confidence-weighted λ_eff so that low-evidence pairs stay close to the hard label.

        Returns:
            soft_target  [N, num_classes]  soft label for each BG pair
            has_evidence [N] bool          True where the pair appeared in fg_matrix
        """
        counts = self.fg_matrix[s_idx, o_idx]           # [N, num_rel]
        pair_total = counts.sum(dim=-1)                  # [N]
        has_evidence = pair_total > 0

        # FG-only empirical distribution: zero out the BG slot and renormalise
        # so that the injected mass is purely about *which FG predicate* is likely.
        p_fg = counts.clone()
        p_fg[:, 0] = 0.0
        fg_sum = p_fg.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        p_fg = p_fg / fg_sum                             # [N, num_rel], sums to 1

        # Confidence: Bayesian-style trust calibrated on annotation volume.
        # prior=10 → 50% trust at 10 annotations, ~90% trust at 90 annotations.
        prior = 10.0
        confidence = pair_total / (pair_total + prior)   # [N], in [0, 1)

        # Effective mixing weight: scales with both bg_discount and evidence strength
        eff_lambda = self.bg_discount * confidence       # [N]

        # Soft target: interpolate between hard BG and empirical FG distribution
        soft_target = torch.zeros(s_idx.shape[0], num_classes, device=device, dtype=torch.float32)
        soft_target[:, 0] = 1.0 - eff_lambda            # BG mass
        soft_target += eff_lambda.unsqueeze(1) * p_fg   # FG mass

        return soft_target, has_evidence

    def forward(self, logits, target, sbj_labels=None, obj_labels=None):
        device = logits.device
        # Always compute the loss in float32 regardless of input dtype.
        # If logits arrive as float16 (e.g. from the non-CLIP linear head under AMP),
        # casting to float32 here prevents:
        #   - priors.to(dtype=float16): 1e-12 underflows to 0 → log(0) = -inf
        #   - -inf logit adjustments → arbitrarily large negative loss
        logits = logits.float()
        priors = self.priors.to(device=device, dtype=torch.float32)
        
        # --- 1. Logit Adjustment (active when tau > 0) ---
        log_priors = torch.log(priors + 1e-12).clamp(min=-20.0, max=20.0)
        adjusted_logits = logits + self.tau * log_priors.unsqueeze(0)
        adjusted_logits = torch.clamp(adjusted_logits, min=-50.0, max=50.0)

        # --- 2. Base Cross Entropy ---
        ce_loss = F.cross_entropy(adjusted_logits, target, reduction='none')

        # --- 3. Plausibility-Aware Soft Supervision ---
        # For BG-labeled pairs whose (S,O) category combination has fg_matrix evidence,
        # replace the hard CE with a soft-label KL term that mixes the hard BG label
        # with the empirical P(FG_rel | S, O) distribution.
        # This turns passive discounting into active positive supervision:
        # the model is taught *which* relations are historically likely for this pair type,
        # regardless of what it currently predicts.
        if (self.fg_matrix is not None
                and self.bg_discount > 0.0
                and sbj_labels is not None
                and obj_labels is not None):
            bg_mask = (target == 0)
            if bg_mask.any():
                bg_idx = bg_mask.nonzero(as_tuple=True)[0]
                s_idx = sbj_labels[bg_idx].clamp(max=self.fg_matrix.shape[0] - 1)
                o_idx = obj_labels[bg_idx].clamp(max=self.fg_matrix.shape[1] - 1)
                num_classes = adjusted_logits.shape[1]

                soft_target, has_evidence = self.get_empirical_soft_target(
                    s_idx, o_idx, num_classes, device)

                if has_evidence.any():
                    ev_local  = has_evidence.nonzero(as_tuple=True)[0]
                    ev_global = bg_idx[ev_local]
                    log_probs = F.log_softmax(adjusted_logits[ev_global].float(), dim=-1)
                    # KL(soft_target || model) = -sum(soft_target * log_softmax)
                    ce_loss[ev_global] = -(soft_target[ev_local] * log_probs).sum(dim=-1)

        # --- 4. Focal Weighting ---
        # (1 - p_t)^gamma
        with torch.no_grad():
            pt = torch.exp(-ce_loss.clamp(max=20.0))
            focal_weight = (1 - pt) ** self.gamma

        # --- 5. Class Balancing ---
        weight = torch.ones_like(target, dtype=logits.dtype, device=device)
        bg_mask = (target == 0)
        fg_mask = (target > 0)
        
        # Swapped to prioritize Foreground (alpha=0.25 -> bg=0.25, fg=0.75*fg_boost)
        weight[bg_mask] = self.alpha
        weight[fg_mask] = (1.0 - self.alpha) * self.fg_boost

        # Combine
        loss = ce_loss * focal_weight * weight
        
        # --- 6. Stable Normalization ---
        # Normalize by sum of weights (Smooth denominator)
        normalizer = weight.sum().clamp(min=1.0)
        
        return loss.sum() / normalizer

class ClassBalancedCELoss(nn.Module):
    """
    Class-Balanced Cross-Entropy Loss.
    Reference: "Class-Balanced Loss Based on Effective Number of Samples" (Cui et al., 2019)

    Uses inverse-effective-number weights — the simplest stable option for visual-only
    learning.  No logit adjustment, no focal term, no coupled hyperparameters.
    Best used as a baseline to isolate what each extra component contributes.
    """
    def __init__(self, pred_freq, beta=0.9999):
        super().__init__()
        if not isinstance(pred_freq, torch.Tensor):
            pred_freq = torch.tensor(pred_freq, dtype=torch.float32)
        pred_freq = torch.nan_to_num(pred_freq, nan=1e-6).clamp(min=1e-6)
        # Scale proportions to pseudo-counts in [1, 1000] so beta^n is meaningful.
        # Preserves relative ordering: rare classes stay rare.
        n = (pred_freq / pred_freq.max() * 1000.0).clamp(min=1.0)
        # Effective number of samples per class
        effective_num = (1.0 - beta ** n) / (1.0 - beta + 1e-12)
        weights = 1.0 / effective_num
        weights = weights / weights.sum() * len(weights)  # normalize: mean weight = 1
        self.register_buffer('class_weights', weights.float())
        # Store priors for sampling (used in RelationHead forward)
        self.register_buffer('priors', pred_freq / (pred_freq.sum() + 1e-12))

    def forward(self, logits, target, sbj_labels=None, obj_labels=None):
        logits = logits.float()
        return F.cross_entropy(logits, target,
                               weight=self.class_weights.to(logits.device))


class CBFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss.
    Combines CB weights (effective-number inverse) with focal attention.
    No logit adjustment — avoids the LA/focal coupling that causes gradient spikes
    when learning from visual features only (no text anchors).

    The two levers are orthogonal:
      - CB weights: how much total attention each CLASS receives
      - gamma: how strongly to down-weight easy examples within a class
    """
    def __init__(self, pred_freq, gamma=1.5, beta=0.9999):
        super().__init__()
        if not isinstance(pred_freq, torch.Tensor):
            pred_freq = torch.tensor(pred_freq, dtype=torch.float32)
        pred_freq = torch.nan_to_num(pred_freq, nan=1e-6).clamp(min=1e-6)
        # Scale proportions to pseudo-counts in [1, 1000] so beta^n is meaningful.
        n = (pred_freq / pred_freq.max() * 1000.0).clamp(min=1.0)
        effective_num = (1.0 - beta ** n) / (1.0 - beta + 1e-12)
        weights = 1.0 / effective_num
        weights = weights / weights.sum() * len(weights)
        self.register_buffer('class_weights', weights.float())
        # Store priors for sampling (used in RelationHead forward)
        self.register_buffer('priors', pred_freq / (pred_freq.sum() + 1e-12))
        self.gamma = gamma

    def forward(self, logits, target, sbj_labels=None, obj_labels=None):
        logits = logits.float()
        w = self.class_weights.to(logits.device)
        # Weighted CE gives the per-sample loss
        ce = F.cross_entropy(logits, target, weight=w, reduction='none')
        # Focal weight: (1 - pt)^gamma where pt is raw model confidence
        with torch.no_grad():
            pt = torch.exp(-ce.clamp(max=20.0))
            focal_weight = (1.0 - pt) ** self.gamma
        return (ce * focal_weight).mean()


class AdaptiveRelationalBalancedLoss(nn.Module):
    """
    Robust version of Class-Balanced and Logit Adjusted loss.
    Scaled to maintain significant gradients for foreground relationship classes.
    """
    def __init__(self, pred_freq, beta=0.99, gamma=2.0, tau=0.2, fg_weight=2.0):
        super(AdaptiveRelationalBalancedLoss, self).__init__()
        # Ensure pred_freq is a count-like tensor (at least 1)
        if not isinstance(pred_freq, torch.Tensor):
            pred_freq = torch.tensor(pred_freq, dtype=torch.float32)
        
        # If pred_freq is frequencies [0, 1], scale them to be like counts
        if pred_freq.max() <= 1.0:
            pred_freq = pred_freq * 2000 # Assume a larger base for better weights
            
        pred_freq = torch.clamp(pred_freq, min=1.0)
        
        # 1. Class-Balanced Weights (Cui et al.)
        effective_num = 1.0 - torch.pow(beta, pred_freq)
        weights = (1.0 - beta) / (effective_num + 1e-8)
        # Normalize weights so that the average weight is 1.0
        weights = weights / weights.mean()
        
        # Stability: Clamp max weight to 20x the average
        self.register_buffer("weights", torch.clamp(weights, max=20.0))
        
        # 2. Priors for Logit Adjustment
        priors = pred_freq / (pred_freq.sum() + 1e-8)
        log_priors = torch.log(priors + 1e-8)
        self.register_buffer("log_priors", torch.clamp(log_priors, min=-20.0, max=20.0))
        
        self.gamma = gamma
        self.tau = tau
        self.fg_weight = fg_weight

    def forward(self, logits, target):
        # 0. NaN Guard for input logits
        if torch.isnan(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0)

        device = logits.device
        dtype = logits.dtype
        
        # 1. Logit Adjustment
        log_priors = self.log_priors.to(device=device, dtype=dtype)
        # Shift logits to encourage higher margins for tail classes
        adjusted_logits = logits + self.tau * log_priors.unsqueeze(0)
        # Guard against extreme logits
        adjusted_logits = torch.clamp(adjusted_logits, min=-50.0, max=50.0)
        
        # 2. Compute Cross Entropy on adjusted logits
        ce_loss = F.cross_entropy(adjusted_logits, target, reduction='none')
        
        # 3. Focal component
        with torch.no_grad():
            pt = torch.exp(-ce_loss)
            focal_weight = (1 - pt) ** self.gamma
        
        # 4. Class balanced weights
        weights = self.weights.to(device=device, dtype=dtype)
        batch_weights = weights[target]
        
        # 5. Foreground Boosting
        fg_mask = (target > 0).to(dtype)
        boost = fg_mask * (self.fg_weight - 1.0) + 1.0
        
        loss = ce_loss * focal_weight * batch_weights * boost
        
        # 6. Normalization: Instead of global mean (drowned by BG), 
        # we normalize by the number of foreground samples + small constant
        num_fg = torch.clamp(fg_mask.sum(), min=1.0)
        final_loss = loss.sum() / num_fg
        
        if torch.isnan(final_loss):
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
            
        return final_loss


class SemanticCompatibilityLoss(nn.Module):
    """
    A loss designed to improve generalization and handle unannotated relations.
    
    1. Smoothed Logit Adjustment (LA): Handles long-tail with a temperature 'tau'.
    2. Plausibility-Aware Background Discounting: Uses word/CLIP embeddings to
       recognize valid but unannotated relations (e.g., 'person sitting on table').
       If a triplet is semantically plausible but labeled as BG, we discount the penalty.
    3. Compositional Contrastive Learning (CCL): Aligns visual relation features
       with semantic embeddings to ensure better generalization to unseen pairs.
    """
    def __init__(self, pred_freq, tau=0.8, smoothing=0.05, bg_discount=2.0, decisive_margin=2.0, ccl_weight=0.1, feat_dim=1024,
                 fg_matrix=None, fg_weight=1.0, poly_epsilon=1.0):
        super(SemanticCompatibilityLoss, self).__init__()
        if not isinstance(pred_freq, torch.Tensor):
            pred_freq = torch.tensor(pred_freq, dtype=torch.float32)
        
        # Clean up pred_freq
        pred_freq = torch.nan_to_num(pred_freq, nan=0.0, posinf=0.0, neginf=0.0)
        pred_freq = torch.clamp(pred_freq, min=0.0)
        
        total = pred_freq.sum()
        if total > 0:
            priors = pred_freq / (total + 1e-12)
        else:
            priors = torch.ones_like(pred_freq) / (len(pred_freq) + 1e-12)
            
        self.register_buffer("priors", priors)
        self.tau = tau 
        self.bg_discount = bg_discount
        self.decisive_margin = decisive_margin
        self.ccl_weight = ccl_weight
        self.fg_weight = fg_weight
        self.poly_epsilon = poly_epsilon # Poly-Loss coefficient for improved recall
        self.gamma = 2.0
        self.embed_dim = feat_dim // 2

        # Dataset Triplet Matrix for empirical plausibility (Idea: reduce penalty for missing annotations)
        if fg_matrix is not None:
            if not isinstance(fg_matrix, torch.Tensor):
                fg_matrix = torch.as_tensor(fg_matrix)
            # Normalize or just use binary presence? Let's use log-transformed counts for smoothness
            # Shape: [num_obj, num_obj, num_rel]
            self.register_buffer("fg_matrix", fg_matrix.float())
        else:
            self.fg_matrix = None

        # CCL Alignment Layer
        self.ccl_proj = nn.Linear(feat_dim, self.embed_dim)
        nn.init.xavier_uniform_(self.ccl_proj.weight)
        nn.init.constant_(self.ccl_proj.bias, 0)

        # Semantic Embeddings for Plausibility falling back or CCL
        num_obj = fg_matrix.shape[0] if fg_matrix is not None else 151
        num_rel = len(pred_freq)
        self.obj_embed = nn.Parameter(torch.zeros(num_obj, self.embed_dim))
        self.rel_embed = nn.Parameter(torch.zeros(num_rel, self.embed_dim))
        nn.init.normal_(self.obj_embed, std=0.01)
        nn.init.normal_(self.rel_embed, std=0.01)

    def compute_plausibility(self, sbj_ids, rel_ids, obj_ids):
        """Estimate semantic plausibility of a triplet."""
        if self.fg_matrix is not None:
            # Idea: Empirical Plausibility based on dataset occurrences.
            # If triplet (s, r, o) exists anywhere in the training set, we consider it plausible.
            # Shape of fg_matrix: [num_obj, num_obj, num_rel]
            
            # rel_ids are 1-indexed predicates. 
            # We ensure we don't index beyond the matrix size if classes differ.
            s_idx = sbj_ids.clamp(max=self.fg_matrix.shape[0]-1)
            o_idx = obj_ids.clamp(max=self.fg_matrix.shape[1]-1)
            r_idx = rel_ids.clamp(max=self.fg_matrix.shape[2]-1)
            
            # Binary plausibility: 1 if it exists, 0 otherwise.
            # We can use a small log scale if we want to reward highly frequent triplets more.
            counts = self.fg_matrix[s_idx, o_idx, r_idx]
            res = (counts > 0).float()
            return res
        else:
            # Semantic fallback: dot product of subject, predicate, object embeddings
            s_idx = sbj_ids.clamp(max=self.obj_embed.shape[0]-1)
            o_idx = obj_ids.clamp(max=self.obj_embed.shape[0]-1)
            r_idx = rel_ids.clamp(max=self.rel_embed.shape[0]-1)
            
            s_emb = self.obj_embed[s_idx]
            r_emb = self.rel_embed[r_idx]
            o_emb = self.obj_embed[o_idx]
            
            # Simple triplet plausibility score
            score = torch.sum((s_emb + o_emb) * r_emb, dim=-1)
            return torch.sigmoid(score)

    def forward(self, logits, target, sbj_labels=None, obj_labels=None, rel_features=None):
        device = logits.device
        dtype = logits.dtype
        
        # 0. NaN/Inf guard for input
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=25.0, neginf=-25.0)
        
        # 1. Logit Adjustment
        priors = self.priors.to(device=device, dtype=dtype)
        if torch.isnan(priors).any() or priors.sum() == 0:
            priors = torch.ones_like(priors) / max(1, priors.size(0))
            
        # Ensure priors are non-negative and finite before log
        priors = torch.clamp(priors, min=1e-8)
        log_priors = torch.log(priors)
        
        # Guard against extremely large adjustments
        log_priors = torch.clamp(log_priors, min=-20.0, max=20.0)
        
        adjusted_logits = logits + self.tau * log_priors.unsqueeze(0)
        
        # Additional safety clamp for adjusted_logits before CE
        # float16 max is ~65k. exp(11) is ~60k. 
        # But we use log-sum-exp, so we can go higher. 50 should be safe.
        adjusted_logits = torch.clamp(adjusted_logits, min=-50.0, max=50.0)
        
        # Check if target is indices or probabilities (for PSG multi-label)
        is_prob_target = (target.dim() == 2 and target.shape == logits.shape)

        # If target is indices, check bounds
        if not is_prob_target:
            target = target.clamp(min=0, max=logits.size(-1) - 1)
        
        ce_loss = F.cross_entropy(adjusted_logits, target, reduction='none')
        
        # Robustness: F.cross_entropy can return inf if logit is small and target is 1
        if torch.isinf(ce_loss).any() or torch.isnan(ce_loss).any():
            ce_loss = torch.nan_to_num(ce_loss, nan=20.0, posinf=20.0, neginf=0.0)

        # 3. Background Discounting via Plausibility (Dataset-aware or Semantic fallback)
        if sbj_labels is not None and obj_labels is not None:
            bg_mask = (target == 0)
            if bg_mask.any() and not is_prob_target:
                # bg_mask has shape [N], ce_loss has shape [N]. 
                with torch.no_grad():
                    # Use cleaned adjusted_logits for stability
                    pred_rel = adjusted_logits[bg_mask].argmax(-1)
                    has_fg_pred = (pred_rel > 0)
                    if has_fg_pred.any():
                        fg_sub_indices = has_fg_pred.nonzero(as_tuple=True)[0]
                        sub_sbj = sbj_labels[bg_mask][fg_sub_indices]
                        sub_rel = pred_rel[fg_sub_indices]
                        sub_obj = obj_labels[bg_mask][fg_sub_indices]
                        
                        sub_plausibility = self.compute_plausibility(sub_sbj, sub_rel, sub_obj)
                        
                        batch_indices = bg_mask.nonzero(as_tuple=True)[0][fg_sub_indices]
                        # Apply empirical discount factor
                        discount = torch.exp(-self.bg_discount * sub_plausibility)
                        ce_loss[batch_indices] *= discount

        # 4. Focal
        with torch.no_grad():
            pt = torch.exp(-ce_loss.clamp(max=20.0))
            focal_weight = (1 - pt) ** self.gamma
            
            # Foreground weighting to handle background dominance
            fg_mask = (target > 0).to(dtype)
            weight = fg_mask * self.fg_weight + (1 - fg_mask)
        
        loss = ce_loss * focal_weight * weight
            
        final_loss = loss.mean()

        # 5. Decisive Margin (DM)
        # Specifically ensures Ground Truth Foreground Logit > Background Logit by a margin.
        # This helps in ranking correct relations higher than 'No Relation'.
        if not is_prob_target and self.decisive_margin > 0:
            fg_mask_bool = (target > 0)
            if fg_mask_bool.any():
                fg_target = target[fg_mask_bool]
                fg_logits_raw = logits[fg_mask_bool]
                
                # Extract logit for the correct class and for background (0)
                gt_logits = torch.gather(fg_logits_raw, 1, fg_target.unsqueeze(1)).squeeze(1)
                bg_logits = fg_logits_raw[:, 0]
                
                # Penalty if GT is not significantly higher than BG
                # weight it similarly to the main loss
                loss_dm = F.relu(bg_logits + self.decisive_margin - gt_logits).mean()
                final_loss = final_loss + 0.5 * loss_dm

        # 6. Compositional Contrastive Learning (CCL)
        if rel_features is not None and target.sum() > 0:
            fg_mask = (target > 0)
            fg_rel_features = rel_features[fg_mask]
            if torch.isnan(fg_rel_features).any():
                fg_rel_features = torch.nan_to_num(fg_rel_features, nan=0.0)
                
            fg_feats = self.ccl_proj(fg_rel_features)
            
            # Use safe normalization
            norm = torch.norm(fg_feats, p=2, dim=-1, keepdim=True)
            fg_feats = fg_feats / (norm + 1e-8)
        
            fg_targets = target[fg_mask]
            
            # Ensure rel_embed matches device and dtype
            rel_embed = self.rel_embed.to(device=device, dtype=fg_feats.dtype)
            
            # Normalize embeddings for stable contrastive learning
            rel_embed_norm = rel_embed / (torch.norm(rel_embed, p=2, dim=-1, keepdim=True) + 1e-8)
            
            # InfoNCE-like contrastive term
            logits_ccl = torch.matmul(fg_feats, rel_embed_norm.T) / 0.07

            loss_ccl = F.cross_entropy(logits_ccl, fg_targets)
            final_loss = final_loss + self.ccl_weight * loss_ccl
            
        return final_loss


class PSGWeightedBCE(nn.Module):
    """
    Implementation of the Weighted BCE loss for PSG as described.
    Penalizes false negatives (missing labels) more than false positives.
    Weight w_p = neg_samples / pos_samples for each class p.
    """
    def __init__(self, pred_freq, reduction='mean'):
        super().__init__()
        if not isinstance(pred_freq, torch.Tensor):
            pred_freq = torch.tensor(pred_freq, dtype=torch.float32)
        
        # pred_freq: frequencies (sum=1) or counts
        # w_p = (Total - Pos) / Pos
        if pred_freq.sum() > 1.1: # Counts
            total = pred_freq.sum()
            pos = torch.clamp(pred_freq, min=1.0)
            neg = total - pos
        else: # Frequencies
            pos = torch.clamp(pred_freq, min=1e-6)
            neg = 1.0 - pos
            
        pos_weight = neg / pos
        
        # Clamp for numerical stability (rare predicates)
        self.register_buffer("pos_weight", torch.clamp(pos_weight, max=1000.0))
        self.reduction = reduction

    def forward(self, logits, target):
        # target: [N] (indices)
        num_classes = logits.size(-1)
        target_one_hot = F.one_hot(target.long(), num_classes).to(logits.dtype)
        
        # BCEWithLogitsLoss with pos_weight p: - [p*y*log(sigm) + (1-y)*log(1-sigm)]
        # This matches the user's Eq (2).
        loss = F.binary_cross_entropy_with_logits(
            logits, 
            target_one_hot, 
            pos_weight=self.pos_weight, 
            reduction='none'
        )
        
        if self.reduction == 'mean':
            # loss.mean() is sum(loss) / (N * P), matching Eq (3).
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CEForSoftLabel(nn.Module):
    """
    Given a soft label, choose the class with max class as the GT.
    converting label like [0.1, 0.3, 0.6] to [0, 0, 1] and apply CrossEntropy
    """
    def __init__(self, reduction="mean"):
        super(CEForSoftLabel, self).__init__()
        self.reduction=reduction

    def forward(self, input, target, pos_weight=None):
        if target.dim() == 1:
            # If target is indices, this loss is just standard CrossEntropy
            return F.cross_entropy(input, target.long(), reduction=self.reduction)
        
        final_target = torch.zeros_like(target)
        final_target[torch.arange(0, target.size(0)), target.argmax(1)] = 1.
        target = final_target
        x = F.log_softmax(input, 1)
        loss = torch.sum(- x * target, dim=1)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'mean':
            return torch.mean(loss)
        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')


class ReweightingCE(nn.Module):
    """
    Standard CrossEntropy with class-level reweighting based on inverse frequency.
    Used to mitigate long-tail distribution bias.
    """
    def __init__(self, pred_weight, reduction="mean"):
        super(ReweightingCE, self).__init__()
        self.reduction = reduction
        # Ensure pred_weight is on the correct device later
        self.register_buffer("pred_weight", torch.tensor(pred_weight, dtype=torch.float32))

    def forward(self, input, target):
        """
        Args:
            input: the prediction [N, num_classes]
            target: the ground truth labels [N]
        """
        # Ensure weight matches input device and dtype
        weight = self.pred_weight.to(device=input.device, dtype=input.dtype)
        
        # CrossEntropyLoss can take weights directly in __init__, 
        # but here we apply them manually for more control if targets were soft
        loss = F.cross_entropy(input, target, weight=weight, reduction=self.reduction)
        return loss


# =============================================================================
# RelationLoss — primary loss for DeformableRelationHead
# =============================================================================

class RelationLoss(nn.Module):
    """
    Combined relation prediction loss for long-tail scene graph generation.

    Components
    ----------
    1. LA-Focal-CB
       Logit-Adjusted Focal Loss with class-balanced per-sample weights.
       Handles the long-tail distribution and fg/bg imbalance.

    2. Semantic smooth
       Auxiliary KL term toward the empirical distribution P(rel | s_cls, o_cls)
       derived from fg_matrix for FG-labeled pairs that have co-occurrence evidence.
       Teaches the model *which* predicates are plausible for a given object-pair
       type — structured supervision instead of uniform label noise.

    3. SupCon (class-balanced Supervised Contrastive)
       Operates on L2-normalised relation features.  Tightens intra-class clusters
       and separates inter-class clusters in feature space, which improves mean
       Recall (mR) for tail predicates without degrading overall Recall.

    4. Geo-aux
       Auxiliary CE restricted to the geometry-branch logits (W_geo × g_so) for
       spatially-grounded predicates (above, below, behind …).  Provides a clean,
       isolated gradient path to geo_logit_bias, independent of the visual path.

    All four terms are in cross-entropy / KL units so their magnitudes are
    naturally compatible; the weight hyperparameters are small multipliers.
    """

    # Substrings that identify geometry-dominated predicates.
    SPATIAL_KEYWORDS = frozenset({
        'above', 'below', 'under', 'behind', 'front', 'left', 'right',
        'near', 'next', 'beside', 'across', 'along', 'inside', 'outside',
        'between', 'over', 'beneath', 'opposite', 'around',
    })

    def __init__(
        self,
        pred_freq,
        fg_matrix=None,
        rel_classes=None,
        # LA + focal + CB
        tau=0.5,
        gamma=2.0,
        alpha=0.25,
        fg_boost=2.0,
        # semantic smoothing
        smooth_weight=0.15,
        # SupCon
        supcon_weight=0.2,
        supcon_tau=0.1,
        supcon_max_per_class=8,
        # geo auxiliary
        geo_weight=0.5,
    ):
        super().__init__()
        if not isinstance(pred_freq, torch.Tensor):
            pred_freq = torch.tensor(pred_freq, dtype=torch.float32)
        pred_freq = torch.nan_to_num(pred_freq, nan=1e-6).clamp(min=1e-12)
        num_classes = len(pred_freq)

        # Priors — also read by samp_processor via loss_evaluator.criterion.priors
        self.register_buffer('priors', pred_freq / pred_freq.sum().clamp(min=1e-12))

        # Per-class CB weights: inv-sqrt frequency, FG normalised to unit mean, capped at 3×.
        # BG (index 0) stays at 1.0 — handled separately by alpha.
        pcw = 1.0 / (pred_freq.clamp(min=1.0) ** 0.5)
        pcw[0] = 1.0
        fg_mean = pcw[1:].mean().clamp(min=1e-6)
        pcw[1:] = (pcw[1:] / fg_mean).clamp(max=3.0)
        self.register_buffer('per_class_weight', pcw)

        # fg_matrix for semantic smoothing
        if fg_matrix is not None:
            if not isinstance(fg_matrix, torch.Tensor):
                fg_matrix = torch.as_tensor(fg_matrix)
            self.register_buffer('fg_matrix', fg_matrix.float())
        else:
            self.fg_matrix = None

        # Spatial predicate mask for geo-aux loss
        spatial_mask = torch.zeros(num_classes, dtype=torch.bool)
        if rel_classes:
            for i, name in enumerate(rel_classes):
                if any(kw in name.lower() for kw in self.SPATIAL_KEYWORDS):
                    spatial_mask[i] = True
        self.register_buffer('spatial_mask', spatial_mask)

        self.num_classes        = num_classes
        self.tau                = tau
        self.gamma              = gamma
        self.alpha              = alpha
        self.fg_boost           = fg_boost
        self.smooth_weight      = smooth_weight
        self.supcon_weight      = supcon_weight
        self.supcon_tau         = supcon_tau
        self.supcon_max_per_class = supcon_max_per_class
        self.geo_weight         = geo_weight

    # ------------------------------------------------------------------
    # Component 1 : Logit-Adjusted Focal + Class-Balanced
    # ------------------------------------------------------------------
    def _la_focal_cb(self, logits, target):
        logits  = logits.float()
        priors  = self.priors.to(logits.device)

        # Logit adjustment
        log_pr = torch.log(priors + 1e-12).clamp(-20.0, 20.0)
        adj    = (logits + self.tau * log_pr.unsqueeze(0)).clamp(-50.0, 50.0)

        # Per-sample cross-entropy
        ce = F.cross_entropy(adj, target, reduction='none')

        # Focal weighting
        with torch.no_grad():
            pt = torch.exp(-ce.clamp(max=20.0))
            fw = (1.0 - pt) ** self.gamma

        # Alpha FG/BG + per-class CB
        pcw     = self.per_class_weight.to(logits.device)[target]
        bg_m    = target == 0
        alpha_w = torch.ones_like(ce)
        alpha_w[bg_m]  = self.alpha
        alpha_w[~bg_m] = (1.0 - self.alpha) * self.fg_boost

        w = fw * alpha_w * pcw
        return (ce * w).sum() / w.sum().clamp(min=1.0)

    # ------------------------------------------------------------------
    # Component 2 : Semantic Label Smoothing
    # ------------------------------------------------------------------
    def _semantic_smooth(self, logits, target, sbj_labels, obj_labels):
        """
        Auxiliary KL term toward P(rel | s_cls, o_cls) for FG pairs with
        fg_matrix evidence.  Purely additive — no overlap with the hard-label CE.
        """
        if self.fg_matrix is None:
            return logits.new_tensor(0.0)

        logits  = logits.float()
        fg_mask = target > 0
        if not fg_mask.any():
            return logits.new_tensor(0.0)

        fg_idx = fg_mask.nonzero(as_tuple=True)[0]
        s_idx  = sbj_labels[fg_idx].clamp(max=self.fg_matrix.shape[0] - 1)
        o_idx  = obj_labels[fg_idx].clamp(max=self.fg_matrix.shape[1] - 1)

        # Empirical predicate distribution (BG slot zeroed, re-normalised over FG)
        counts     = self.fg_matrix[s_idx, o_idx].clone()   # [M, C]
        counts[:, 0] = 0.0
        ev_sum     = counts.sum(dim=-1)                      # [M]
        has_ev     = ev_sum > 0
        if not has_ev.any():
            return logits.new_tensor(0.0)

        ev_local  = has_ev.nonzero(as_tuple=True)[0]
        ev_global = fg_idx[ev_local]
        p_emp     = counts[ev_local] / ev_sum[ev_local].unsqueeze(1).clamp(min=1e-6)

        log_probs = F.log_softmax(logits[ev_global], dim=-1)
        kl        = -(p_emp.to(logits.device) * log_probs).sum(dim=-1)   # [K]
        return kl.mean()

    # ------------------------------------------------------------------
    # Component 3 : Class-Balanced Supervised Contrastive
    # ------------------------------------------------------------------
    def _supcon(self, features, target):
        fg_mask = target > 0
        if fg_mask.sum() < 2:
            return features.new_tensor(0.0)

        fg_feat = F.normalize(features[fg_mask].float(), dim=-1)   # [M, D]
        fg_lbl  = target[fg_mask]                                   # [M]

        # Sample at most K per class so head classes don't dominate gradients
        K     = self.supcon_max_per_class
        parts = []
        for cls in fg_lbl.unique():
            idx = (fg_lbl == cls).nonzero(as_tuple=True)[0]
            if len(idx) > K:
                idx = idx[torch.randperm(len(idx), device=idx.device)[:K]]
            parts.append(idx)

        sel  = torch.cat(parts)
        feat = fg_feat[sel]    # [S, D]
        lbl  = fg_lbl[sel]     # [S]
        S    = feat.shape[0]
        if S < 2:
            return features.new_tensor(0.0)

        # Cosine similarity / temperature
        sim      = feat @ feat.T / self.supcon_tau                       # [S, S]
        diag     = torch.eye(S, dtype=torch.bool, device=feat.device)
        pos_mask = (lbl.unsqueeze(0) == lbl.unsqueeze(1)) & ~diag        # [S, S]

        # SupCon: -1/|P(i)| * sum_{p in P(i)} log [ exp(sim_ip) / sum_{a≠i} exp(sim_ia) ]
        log_probs = F.log_softmax(sim.masked_fill(diag, -1e4), dim=1)   # [S, S]
        num_pos   = pos_mask.float().sum(dim=1)                          # [S]
        has_pos   = num_pos > 0
        if not has_pos.any():
            return features.new_tensor(0.0)

        loss_per = -(log_probs * pos_mask.float()).sum(dim=1) / num_pos.clamp(min=1)
        return loss_per[has_pos].mean()

    # ------------------------------------------------------------------
    # Component 4 : Geometry Auxiliary CE
    # ------------------------------------------------------------------
    def _geo_aux(self, geo_logits, target):
        """CE on the geometry-branch logits for spatially-grounded predicates only."""
        spatial = self.spatial_mask.to(target.device)
        if not spatial.any():
            return geo_logits.new_tensor(0.0)
        sp_mask = spatial[target]
        if not sp_mask.any():
            return geo_logits.new_tensor(0.0)
        return F.cross_entropy(geo_logits[sp_mask].float(), target[sp_mask])

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, logits, target, sbj_labels=None, obj_labels=None,
                features=None, geo_logits=None):
        # 1. Core LA-Focal-CB classification loss
        loss = self._la_focal_cb(logits, target)

        # 2. Semantic label smoothing toward empirical P(rel | s_cls, o_cls)
        if sbj_labels is not None and obj_labels is not None and self.smooth_weight > 0:
            loss = loss + self.smooth_weight * self._semantic_smooth(
                logits, target, sbj_labels, obj_labels)

        # 3. Class-balanced SupCon on relation features
        if features is not None and self.supcon_weight > 0:
            loss = loss + self.supcon_weight * self._supcon(features, target)

        # 4. Geometry auxiliary CE for spatial predicates
        if geo_logits is not None and self.geo_weight > 0:
            loss = loss + self.geo_weight * self._geo_aux(geo_logits, target)

        return loss