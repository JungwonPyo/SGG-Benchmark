import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import numpy.random as npr
from typing import List, Dict, Any, Tuple, Optional

from sgg_benchmark.layers import smooth_l1_loss, Label_Smoothing_Regression
from sgg_benchmark.modeling.box_coder import BoxCoder
from sgg_benchmark.modeling.matcher import Matcher
from sgg_benchmark.structures.box_ops import box_iou
from sgg_benchmark.modeling.utils import cat

from .reweight_loss import ReweightingCE, CEForSoftLabel, EdgeDensityLoss, BalancedLogitAdjustedLoss, SemanticCompatibilityLoss, AdaptiveRelationalBalancedLoss, BCEWithLogitsIgnoreNegative, PSGWeightedBCE, ClassBalancedCELoss, CBFocalLoss

class RelationLossComputation(nn.Module):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
        self,
        cfg,
        attri_on,
        num_attri_cat,
        max_num_attri,
        attribute_sampling,
        attribute_bgfg_ratio,
        rel_loss,
        pred_weight,
        statistics,
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        super(RelationLossComputation, self).__init__()
        self.cfg = cfg
        self.attri_on = attri_on
        self.num_attri_cat = num_attri_cat
        self.max_num_attri = max_num_attri
        self.attribute_sampling = attribute_sampling
        self.attribute_bgfg_ratio = attribute_bgfg_ratio

        self.criterion_loss_obj = nn.CrossEntropyLoss()

        # Handle new nested loss configuration
        if not isinstance(rel_loss, str):
            loss_cfg = rel_loss
            rel_loss = getattr(loss_cfg, "loss_type", "CrossEntropyLoss")
        else:
            loss_cfg = cfg.model.roi_relation_head.get("loss", {})

        if rel_loss == "LabelSmoothingRegression":
            e = getattr(loss_cfg, "label_smoothing_epsilon", 0.01)
            self.criterion_loss = Label_Smoothing_Regression(e=e)
        elif rel_loss == 'BCEWithLogitsLoss':
            self.criterion_loss = BCEWithLogitsIgnoreNegative()
        elif rel_loss == 'CEForSoftLabel':
            self.criterion_loss = CEForSoftLabel()
        elif rel_loss == "ReweightingCE":
            self.criterion_loss = ReweightingCE(pred_weight)
        elif rel_loss == "BCEWithLogitsIgnoreNegative":
            self.criterion_loss = BCEWithLogitsIgnoreNegative()
        elif rel_loss == "CrossEntropyLoss":
            self.criterion_loss = nn.CrossEntropyLoss()
        elif rel_loss == "ClassBalancedCELoss":
            # pred_weight is proportional to class frequency; CB formula expects
            # counts where common=large, rare=small — pass directly (not inverted).
            beta = getattr(loss_cfg, "beta", 0.9999)
            self.criterion_loss = ClassBalancedCELoss(pred_weight, beta=beta)
        elif rel_loss == "CBFocalLoss":
            # Same: pass pred_weight directly so rare classes get large CB weights.
            # The old 1/pred_weight inversion made CB upweight head classes → collapse.
            gamma = getattr(loss_cfg, "gamma", 1.5)
            beta  = getattr(loss_cfg, "beta", 0.9999)
            self.criterion_loss = CBFocalLoss(pred_weight, gamma=gamma, beta=beta)
        elif rel_loss == "EdgeDensityLoss":
            self.criterion_loss = EdgeDensityLoss()
        elif rel_loss == "BalancedLogitAdjustedLoss":
            # pred_weight is typically 1/freq, so freq = 1/weight
            pred_freq = 1.0 / (pred_weight + 1e-12)
            tau       = getattr(loss_cfg, "logit_adjustment_tau", 1.0)
            gamma     = getattr(loss_cfg, "gamma",       2.0)
            alpha     = getattr(loss_cfg, "alpha",       0.4)
            fg_boost  = getattr(loss_cfg, "fg_boost",    1.2)
            bg_disc   = getattr(loss_cfg, "bg_discount", 0.5)
            fg_matrix = statistics.get("fg_matrix", None)
            self.criterion_loss = BalancedLogitAdjustedLoss(
                pred_freq, gamma=gamma, alpha=alpha, tau=tau,
                fg_boost=fg_boost, fg_matrix=fg_matrix, bg_discount=bg_disc)
        elif rel_loss == "AdaptiveRelationalBalancedLoss":
            pred_freq = 1.0 / (pred_weight + 1e-12)
            beta = getattr(loss_cfg, "beta", 0.99)
            gamma = getattr(loss_cfg, "gamma", 2.0)
            tau = getattr(loss_cfg, "logit_adjustment_tau", 0.2)
            fg_weight = getattr(loss_cfg, "fg_weight", 2.0)
            self.criterion_loss = AdaptiveRelationalBalancedLoss(pred_freq, beta=beta, gamma=gamma, tau=tau, fg_weight=fg_weight)
        elif rel_loss == "PSGWeightedBCE":
            pred_freq = 1.0 / (pred_weight + 1e-12)
            self.criterion_loss = PSGWeightedBCE(pred_freq)
        elif rel_loss == "SemanticCompatibilityLoss":
            pred_freq = 1.0 / (pred_weight + 1e-12)
            fg_matrix = statistics.get('fg_matrix', None)
            
            # Smartly infer the feature dimension for CCL alignment
            # 1. Check if Query-based head or DSFormer is used (embed_dim)
            # 2. Check if text-based predictors are used (embed_dim)
            # 3. Default to mlp_head_dim for traditional Motif-like predictors
            if cfg.model.roi_relation_head.feature_extractor == "QueryFeatureExtractor" or \
               cfg.model.roi_box_head.feature_extractor == "PatchFeatureExtractor":
                feat_dim = cfg.model.roi_relation_head.embed_dim
            elif cfg.model.roi_relation_head.use_union_features or cfg.model.roi_relation_head.use_spatial_features:
                feat_dim = cfg.model.roi_relation_head.mlp_head_dim
            else:
                feat_dim = getattr(cfg.model.roi_relation_head, "embed_dim", 200)
            
            # Use configuration parameters if available
            tau = getattr(loss_cfg, "logit_adjustment_tau", 0.3)
            bg_discount = getattr(loss_cfg, "bg_discount", 2.0)
            ccl_weight = getattr(loss_cfg, "ccl_weight", 0.1)
            smoothing = getattr(loss_cfg, "label_smoothing_epsilon", 0.0)
            fg_weight = getattr(loss_cfg, "fg_weight", 2.0)
            decisive_margin = getattr(loss_cfg, "decisive_margin", 2.0)
            poly_epsilon = getattr(loss_cfg, "poly_epsilon", 1.0)
            
            self.criterion_loss = SemanticCompatibilityLoss(
                pred_freq, feat_dim=feat_dim, tau=tau, bg_discount=bg_discount, 
                ccl_weight=ccl_weight, smoothing=smoothing,
                fg_matrix=fg_matrix, fg_weight=fg_weight,
                decisive_margin=decisive_margin, poly_epsilon=poly_epsilon
            )


    def forward(self, proposals: List[Dict[str, Any]], rel_labels, relation_logits, rel_pair_idxs=None, rel_features=None, refine_logits=None):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            proposals (list[Dict])
            rel_labels (list[Tensor])
            relation_logits (list[Tensor])
            rel_pair_idxs (list[Tensor])
            rel_features (Tensor)
            refine_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """
        if self.attri_on and refine_logits is not None:
            if isinstance(refine_logits[0], (list, tuple)):
                refine_obj_logits, refine_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attri_on = False
                refine_obj_logits = refine_logits
        else:
            refine_obj_logits = refine_logits
            refine_att_logits = None # Ensure it is defined if self.attri_on is False or refine_logits is None

        relation_logits = cat(relation_logits, dim=0)
        
        # Universal NaN/Inf Guard for Logits
        if torch.isnan(relation_logits).any() or torch.isinf(relation_logits).any():
            print("DEBUG: relation_logits contains NaNs or Infs before loss computation! Cleaning up...")
            relation_logits = torch.nan_to_num(relation_logits, nan=0.0, posinf=20.0, neginf=-20.0)

        fg_labels = cat([proposal["labels"] for proposal in proposals], dim=0)
        rel_labels = cat(rel_labels, dim=0)

        if isinstance(self.criterion_loss, BalancedLogitAdjustedLoss) or isinstance(self.criterion_loss, SemanticCompatibilityLoss) or isinstance(self.criterion_loss, AdaptiveRelationalBalancedLoss) or isinstance(self.criterion_loss, PSGWeightedBCE):
            # Extract subject and object labels for each pair to compute plausibility
            sbj_labels = []
            obj_labels = []
            for i, idxs in enumerate(rel_pair_idxs):
                img_labels = proposals[i]["labels"]
                sbj_labels.append(img_labels[idxs[:, 0]])
                obj_labels.append(img_labels[idxs[:, 1]])
            
            sbj_labels = cat(sbj_labels, dim=0)
            obj_labels = cat(obj_labels, dim=0)
            
            loss_relation = self.criterion_loss(relation_logits, rel_labels, sbj_labels, obj_labels)
        else:
            loss_relation = self.criterion_loss(relation_logits, rel_labels)

        if refine_obj_logits is not None:
            if not isinstance(refine_obj_logits, torch.Tensor):
                refine_obj_logits = cat(refine_obj_logits, dim=0)
            loss_refine_obj = self.criterion_loss_obj(refine_obj_logits, fg_labels.long())

        # The following code is used to calculate sampled attribute loss
        if self.attri_on and refine_att_logits is not None:
            if not isinstance(refine_att_logits, torch.Tensor):
                refine_att_logits = cat(refine_att_logits, dim=0)
            fg_attributes = cat([proposal["attributes"] for proposal in proposals], dim=0)

            attribute_targets, fg_attri_idx = self.generate_attributes_target(fg_attributes)
            if float(fg_attri_idx.sum()) > 0:
                # have at least one bbox got fg attributes
                refine_att_logits = refine_att_logits[fg_attri_idx > 0]
                attribute_targets = attribute_targets[fg_attri_idx > 0]
            else:
                refine_att_logits = refine_att_logits[0].view(1, -1)
                attribute_targets = attribute_targets[0].view(1, -1)

            loss_refine_att = self.attribute_loss(refine_att_logits, attribute_targets, 
                                             fg_bg_sample=self.attribute_sampling, 
                                             bg_fg_ratio=self.attribute_bgfg_ratio)
            if refine_obj_logits is not None: 
                return loss_relation, (loss_refine_obj, loss_refine_att)
            else:
                return loss_relation, loss_refine_att
        elif refine_obj_logits is not None: 
            return loss_relation, loss_refine_obj
        else:
            return loss_relation, None

    def generate_attributes_target(self, attributes):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert self.max_num_attri == attributes.shape[1]
        device = attributes.device
        num_obj = attributes.shape[0]

        fg_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=device).float()

        for idx in torch.nonzero(fg_attri_idx).squeeze(1).tolist():
            for k in range(self.max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, fg_attri_idx

    def attribute_loss(self, logits, labels, fg_bg_sample=True, bg_fg_ratio=3):
        if fg_bg_sample:
            loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').view(-1)
            fg_loss = loss_matrix[labels.view(-1) > 0]
            bg_loss = loss_matrix[labels.view(-1) <= 0]

            num_fg = fg_loss.shape[0]
            # if there is no fg, add at least one bg
            num_bg = max(int(num_fg * bg_fg_ratio), 1)   
            perm = torch.randperm(bg_loss.shape[0], device=bg_loss.device)[:num_bg]
            bg_loss = bg_loss[perm]

            return torch.cat([fg_loss, bg_loss], dim=0).mean()
        else:
            attri_loss = F.binary_cross_entropy_with_logits(logits, labels)
            attri_loss = attri_loss * self.num_attri_cat / 20.0
            return attri_loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        target = target.view(-1)

        logpt = F.log_softmax(input)
        logpt = logpt.index_select(-1, target).diag()
        logpt = logpt.view(-1)
        pt = logpt.exp()

        logpt = logpt * self.alpha * (target > 0).float() + logpt * (1 - self.alpha) * (target <= 0).float()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class FocalLossFGBGNormalization(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, logits=True, fgbgnorm=True):
        super(FocalLossFGBGNormalization, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_loss = FocalLoss(alpha, gamma, logits, reduce=False)

    def forward(self, inputs, targets, reduce=True):
        loss = self.focal_loss(inputs, targets)
        
        loss = loss.sum(-1)
        loss /= (len(torch.nonzero(targets)) + 1)

        return loss.mean(-1)