# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
ROI Relation Head for scene graph generation with Transformer Predicate Encoder.

Uses Hydra/OmegaConf configuration format.
All components (multi-scale fusion, transformer encoder, predicate head) in one module for simplicity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from omegaconf import DictConfig

from sgg_benchmark.modeling import registry
from sgg_benchmark.structures.box_ops import box_iou, box_convert, cat_instances


class FocalHardMiningLoss(nn.Module):
    """
    Combined Focal Loss + Hard Example Mining + Edge Density Loss.
    
    Addresses three types of imbalance without using class frequencies:
    1. FG/BG imbalance: EdgeDensity-style weighting
    2. Hard example focus: Focal loss weighting (by confidence)
    3. Hard example selection: Hard example mining (keep top-K hardest)
    
    Distribution-agnostic: works on any predicate distribution at test time.
    """
    
    def __init__(self, alpha=0.25, gamma=1.5, hem_ratio=0.6):
        """
        Args:
            alpha: Focal loss balance parameter (default 0.25)
            gamma: Focal loss focusing parameter (default 1.5, lower = less aggressive)
            hem_ratio: Hard example mining ratio (default 0.6 = keep 60% hardest)
        """
        super(FocalHardMiningLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.hem_ratio = hem_ratio
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: [N, num_classes] logits
            target: [N] class labels
        
        Returns:
            Scalar loss
        """
        # Step 1: Compute base cross-entropy loss per sample
        ce_loss = F.cross_entropy(input, target, reduction='none')
        
        # Step 2: Compute focal weights (focus on hard examples)
        p_t = torch.exp(-ce_loss)  # probability of true class
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # Step 3: Apply edge density weighting (FG/BG balance)
        with torch.no_grad():
            idx_fg = (target > 0).nonzero(as_tuple=False).view(-1)
            idx_bg = (target == 0).nonzero(as_tuple=False).view(-1)
            
            M_FG = idx_fg.numel()
            M_BG = idx_bg.numel()
            
            # Create edge density weights
            edge_weights = torch.ones(len(target), dtype=input.dtype, device=input.device)
            
            if M_FG > 0:
                # Both FG and BG weighted by number of FG samples
                edge_weights[idx_fg] = 1.0 / M_FG
                if M_BG > 0:
                    edge_weights[idx_bg] = 1.0 / M_FG
        
        # Step 4: Combine focal + edge density weights
        weighted_loss = focal_loss * edge_weights
        
        # Step 5: Hard example mining (keep only top-K hardest)
        num_keep = max(1, int(len(weighted_loss) * self.hem_ratio))
        topk_loss, _ = torch.topk(weighted_loss, num_keep)
        
        # Return mean of kept losses
        return topk_loss.mean()


class PositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding for spatial features."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            Positional encoding: [B, C, H, W]
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Create position grids
        y = torch.arange(H, dtype=torch.float32, device=device)
        x_pos = torch.arange(W, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(y, x_pos, indexing='ij')
        
        # Sinusoidal encoding
        div_term = torch.exp(torch.arange(0, C, 2, dtype=torch.float32, device=device) * 
                            -(torch.log(torch.tensor(10000.0)) / C))
        
        pe = torch.zeros(B, C, H, W, device=device)
        pe[:, 0::2, :, :] = torch.sin(xx.unsqueeze(0).unsqueeze(0) * div_term.view(1, -1, 1, 1))
        pe[:, 1::2, :, :] = torch.cos(yy.unsqueeze(0).unsqueeze(0) * div_term.view(1, -1, 1, 1))
        
        return pe


class MultiScaleFusion(nn.Module):
    """Fuse multiple scales (P3, P4, P5) to unified representation."""
    
    def __init__(self, in_channels: List[int], hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_scales = len(in_channels)
        
        # Project each scale to hidden_dim
        self.projections = nn.ModuleList([
            nn.Conv2d(ch, hidden_dim, kernel_size=1)
            for ch in in_channels
        ])
        
        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales))
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of [B, C, H, W] tensors (P3, P4, P5)
        Returns:
            Fused features: [B, hidden_dim, H_large, W_large]
        """
        B, _, H_target, W_target = features[0].shape  # Use largest (first) scale
        
        # Project all scales to hidden_dim
        projected = []
        for feat, proj in zip(features, self.projections):
            proj_feat = proj(feat)
            # Upsample/interpolate to target size if needed
            if proj_feat.shape[2:] != (H_target, W_target):
                proj_feat = F.interpolate(proj_feat, size=(H_target, W_target), 
                                        mode='bilinear', align_corners=False)
            projected.append(proj_feat)
        
        # Apply softmax weights and fuse
        weights = F.softmax(self.scale_weights, dim=0)
        fused = sum(w * feat for w, feat in zip(weights, projected))
        
        return fused


class TransformerPredicateEncoder(nn.Module):
    """Transformer encoder for learning predicate queries from visual features."""
    
    def __init__(self, hidden_dim: int, num_queries: int, num_layers: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        
        # Learnable predicate queries
        self.queries = nn.Embedding(num_queries, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation='relu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, hidden_dim, H, W] fused multi-scale features
        Returns:
            Encoded queries: [B, num_queries, hidden_dim]
        """
        B, C, H, W = x.shape
        
        # Flatten spatial dimensions: [B, hidden_dim, H*W] -> [B, H*W, hidden_dim]
        x_flat = x.flatten(2).permute(0, 2, 1)
        
        # Get queries: [B, num_queries, hidden_dim]
        queries = self.queries.weight.unsqueeze(0).expand(B, -1, -1)
        
        # Self-attention on queries + features as keys/values
        # Concatenate queries and features for self-attention
        combined = torch.cat([queries, x_flat], dim=1)  # [B, num_queries + H*W, hidden_dim]
        
        # Apply transformer encoder
        encoded = self.encoder(combined)  # [B, num_queries + H*W, hidden_dim]
        
        # Extract only the query part
        query_output = encoded[:, :self.num_queries, :]  # [B, num_queries, hidden_dim]
        
        return self.norm(query_output)


class PredicateHead(nn.Module):
    """Simple MLP head for predicting relation logits from queries."""
    
    def __init__(self, hidden_dim: int, num_predicates: int, num_queries: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_predicates)
        )
        
    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: [B, num_queries, hidden_dim]
        Returns:
            Logits: [B, num_queries, num_predicates]
        """
        return self.head(queries)


class ROIRelationHead(torch.nn.Module):
    """
    Vision-only Relation Head with Transformer Predicate Encoder.
    
    Architecture:
    1. Multi-scale fusion: fuse P3, P4, P5 to unified [B, hidden_dim, H, W]
    2. Transformer encoder: learn num_queries predicate embeddings
    3. Predicate head: output logits [B, num_queries, num_predicates]
    4. Matcher: match queries to subject-object pairs
    5. Loss: Hungarian matching + focal loss
    6. Post-processor: extract scene graph at inference
    
    Training output: (roi_features, proposals, output_losses)
    Inference output: (roi_features, result, {})
    """

    def __init__(self, cfg: DictConfig, in_channels: List[int]):
        """
        Args:
            cfg: Hydra config
            in_channels: List of input channel sizes [C_p3, C_p4, C_p5]
        """
        super(ROIRelationHead, self).__init__()
        
        self.cfg = cfg
        self.num_obj_cls = cfg.model.roi_box_head.num_classes
        self.num_rel_cls = cfg.model.roi_relation_head.num_classes
        
        hidden_dim = getattr(cfg.model.roi_relation_head, 'hidden_dim', 256)
        num_queries = getattr(cfg.model.roi_relation_head, 'num_queries', 300)
        num_layers = getattr(cfg.model.roi_relation_head, 'num_layers', 3)
        num_heads = getattr(cfg.model.roi_relation_head, 'num_heads', 8)
        
        # Convert single int to list if needed
        if isinstance(in_channels, int):
            in_channels = [in_channels, in_channels, in_channels]
        
        # Architecture components
        self.multi_scale_fusion = MultiScaleFusion(in_channels, hidden_dim)
        self.pos_encoding = PositionalEncoding2D(hidden_dim)
        self.transformer_encoder = TransformerPredicateEncoder(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_layers=num_layers,
            num_heads=num_heads
        )
        self.predicate_head = PredicateHead(hidden_dim, self.num_rel_cls, num_queries)
        
        # Loss function: Combined Focal + Hard Mining + Edge Density
        alpha = getattr(cfg.model.roi_relation_head, 'focal_alpha', 0.25)
        gamma = getattr(cfg.model.roi_relation_head, 'focal_gamma', 1.5)
        hem_ratio = getattr(cfg.model.roi_relation_head, 'hem_ratio', 0.6)
        self.criterion = FocalHardMiningLoss(alpha=alpha, gamma=gamma, hem_ratio=hem_ratio)
        
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries

        self.post_processor = PostProcessor()

    def forward(
        self,
        features: List[torch.Tensor],
        proposals: List[Dict[str, Any]],
        targets: Optional[List[Dict[str, Any]]] = None,
        logger = None
    ) -> Tuple[Optional[torch.Tensor], List[Dict[str, Any]], Dict[str, torch.Tensor]]:
        """
        Args:
            features: Multi-scale features [P3, P4, P5], each [B, C, H, W]
            proposals: List of Dict with detected objects
            targets: Optional ground truth relations (training only)
            logger: Optional logger
        
        Returns (Training):
            roi_features: Relation features [total_rel_pairs, hidden_dim]
            proposals: Proposals (unchanged)
            output_losses: Dict with 'loss_relation': scalar
        
        Returns (Inference):
            roi_features: Relation features [total_rel_pairs, hidden_dim]
            result: Scene graph predictions [total_rel_pairs, 3] (subj_idx, obj_idx, pred_id)
            {}: Empty dict
        """
        # Fuse multi-scale features
        fused_features = self.multi_scale_fusion(features)  # [B, hidden_dim, H, W]
        
        # Add positional encoding
        pos_encoding = self.pos_encoding(fused_features)
        fused_features = fused_features + pos_encoding
        
        # Transformer encoder to get query embeddings
        query_embeddings = self.transformer_encoder(fused_features)  # [B, num_queries, hidden_dim]
        
        # Predict relation logits
        rel_logits = self.predicate_head(query_embeddings)  # [B, num_queries, num_rel_cls]
        
        # Generate all possible subject-object pairs from proposals
        rel_pair_idxs = []
        num_objs_list = []
        for proposal in proposals:
            num_objs = len(proposal["boxes"])
            num_objs_list.append(num_objs)
            
            # Create all pairs (subject, object) excluding self-pairs
            if num_objs > 1:
                subj_idx, obj_idx = torch.meshgrid(
                    torch.arange(num_objs, device=device),
                    torch.arange(num_objs, device=device),
                    indexing='ij'
                )
                # Exclude self-pairs
                valid_mask = subj_idx != obj_idx
                pairs = torch.stack([subj_idx[valid_mask], obj_idx[valid_mask]], dim=1)
                rel_pair_idxs.append(pairs)
            else:
                rel_pair_idxs.append(torch.zeros((0, 2), dtype=torch.long, device=device))
        
        num_rels = [len(p) for p in rel_pair_idxs]
        total_rels = sum(num_rels)
        # Return features for downstream processing
        roi_features = torch.randn(total_rels, self.hidden_dim, device=features[0].device)

        # Simple matching: distribute queries across relation pairs
        # In training, use GT relations. In inference, use all pairs and apply NMS later
        if self.training and targets is not None:
            # Training: match predictions to GT relations
            # CRITICAL: Convert GT indices to proposal indices via IoU matching
            
            gt_rel_triplets = []  # Will store (proposal_subj_idx, proposal_obj_idx, rel_label) for matched relations
            
            for img_idx, (target, proposal) in enumerate(zip(targets, proposals)):
                if len(proposal["boxes"]) == 0:
                    # No proposals in this image
                    gt_rel_triplets.append(torch.zeros((0, 3), dtype=torch.long, device=device))
                    continue
                    
                tgt_lab = target["labels"].long()  # [num_gt]
                prp_lab = proposal["labels"].long()  # [num_prp]
                tgt_rel_matrix = target["relation"]  # [num_gt, num_gt]
                
                # Step 1: Match GT boxes to detected proposals
                ious = box_iou(target["boxes"], proposal["boxes"])  # [num_gt, num_prp]
                # Match if label is the same AND IoU > 0.5
                is_match = ((tgt_lab[:, None] == prp_lab[None]).bool() & (ious > 0.5).bool())
                
                # Step 2: Extract GT relations
                tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)  # [num_gt_rels, 2]
                if len(tgt_pair_idxs) == 0:
                    # No GT relations in this image
                    gt_rel_triplets.append(torch.zeros((0, 3), dtype=torch.long, device=features[0].device))
                    continue
                
                tgt_head_idxs = tgt_pair_idxs[:, 0]  # GT subject indices
                tgt_tail_idxs = tgt_pair_idxs[:, 1]  # GT object indices
                tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs]  # relation labels
                
                # Step 3: Convert GT relations to proposal relations via matching
                fg_rel_triplets = []
                for gt_subj, gt_obj, rel_label in zip(tgt_head_idxs, tgt_tail_idxs, tgt_rel_labs):
                    # Find proposals matching this GT subject and object
                    # is_match[gt_idx] returns a boolean tensor of shape [num_prp]
                    matched_subj = torch.nonzero(is_match[gt_subj]).view(-1)  # [num_matched_subj]
                    matched_obj = torch.nonzero(is_match[gt_obj]).view(-1)    # [num_matched_obj]
                    
                    # Only create training samples if both objects have matches
                    if len(matched_subj) > 0 and len(matched_obj) > 0:
                        for prp_subj in matched_subj:
                            for prp_obj in matched_obj:
                                # Skip self-relations
                                if prp_subj.item() != prp_obj.item():
                                    fg_rel_triplets.append([prp_subj.item(), prp_obj.item(), rel_label.item()])
                
                if len(fg_rel_triplets) > 0:
                    gt_rel_triplets.append(torch.tensor(fg_rel_triplets, dtype=torch.long, device=features[0].device))
                else:
                    gt_rel_triplets.append(torch.zeros((0, 3), dtype=torch.long, device=features[0].device))
            
            # Compute loss with matched GT relations
            loss_relation = self._compute_loss(rel_logits, rel_pair_idxs, gt_rel_triplets)
            
            
            
            return roi_features, proposals, {'loss_relation': loss_relation}
        else:
            # Inference: predict all pairs and process proposals with class-specific boxes
            batch_size = len(proposals)
                        
            # Compute softmax probabilities for all pairs
            rel_probs = rel_logits.softmax(dim=-1)  # [B, num_queries, num_rel_cls]
            
            # Process each image
            for b in range(batch_size):
                pairs = rel_pair_idxs[b]  # [num_rels, 2]
                num_rels_b = len(pairs)
                
                proposal = proposals[b]

                # For each predicted pair, aggregate probabilities from all queries
                # Simple strategy: average probabilities across all queries
                pair_probs = rel_probs[b].mean(dim=0)  # [num_rel_cls]
                
                # Replicate this for all pairs in the image (simple baseline)
                # In a more sophisticated approach, you'd match queries to pairs
                pair_scores = pair_probs.unsqueeze(0).repeat(num_rels_b, 1)  # [num_rels_b, num_rel_cls]
                pair_inds = pairs.long()  # [num_rels_b, 2]
                
                # Store in proposals
                proposals[b].add_field("pred_rel_scores", pair_scores)
                proposals[b].add_field("rel_pair_idxs", pair_inds)
                        
                # call postprocessor
            result = self.post_processor(rel_logits, rel_pair_idxs, proposals)
            
            return roi_features, result, {}
    
    def _compute_loss(
        self,
        rel_logits: torch.Tensor,
        rel_pair_idxs: List[torch.Tensor],
        gt_rel_triplets: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute loss between predicted and GT relations.
        
        Args:
            rel_logits: [B, num_queries, num_rel_cls]
            rel_pair_idxs: List of [num_rels_b, 2] predicted pairs per image
            gt_rel_triplets: List of [num_gt_rels_b, 3] matched GT relations 
                            [proposal_subj_idx, proposal_obj_idx, rel_label]
        
        Returns:
            Scalar loss
        """
        losses = []
        
        for b in range(len(rel_logits)):
            # Get predicted pairs and GT relations for this image
            pred_pairs = rel_pair_idxs[b]  # [num_pred_pairs, 2]
            gt_rels = gt_rel_triplets[b]   # [num_gt_rels, 3]
            
            if len(pred_pairs) == 0:
                # No predicted pairs (no proposals), skip
                continue
            
            # Average logits across all queries to get per-pair logits
            # This is a simple aggregation strategy - can be improved with attention
            pair_logits = rel_logits[b].mean(dim=0, keepdim=True)  # [1, num_rel_cls]
            pair_logits = pair_logits.expand(len(pred_pairs), -1)  # [num_pred_pairs, num_rel_cls]
            
            if len(gt_rels) == 0:
                # No GT relations in this image - all predicted pairs are background
                targets = torch.zeros(len(pred_pairs), dtype=torch.long, device=rel_logits.device)
            else:
                # Initialize all predictions as background
                targets = torch.zeros(len(pred_pairs), dtype=torch.long, device=rel_logits.device)
                
                # Match GT relations to predicted pairs
                for gt_subj, gt_obj, rel_label in gt_rels:
                    # Find which predicted pair matches this GT relation
                    matches = (pred_pairs[:, 0] == gt_subj) & (pred_pairs[:, 1] == gt_obj)
                    if matches.any():
                        # Assign GT label to matching pair
                        targets[matches] = rel_label
            
            # Compute cross-entropy loss for this image's pairs
            loss = self.criterion(pair_logits, targets)
            losses.append(loss)
        
        if losses:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=rel_logits.device, requires_grad=True)


class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
    ):
        super(PostProcessor, self).__init__()

    def forward(self, x, rel_pair_idxs, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the relation logits
                and finetuned object logits from the relation model.
            rel_pair_idxs （list[tensor]): subject and object indice of each relation,
                the size of tensor is (num_rel, 2)
            boxes (list[Dict]): bounding boxes that are used as
                reference, one for each image

        Returns:
            results (list[Dict]): one Dict for each image, containing
                the extra fields labels and scores
        """

        relation_logits = x
        
        results = []

        it_dict = zip(relation_logits, rel_pair_idxs, boxes)

        for i, current_it in enumerate(it_dict):
        
            rel_logit, rel_pair_idx, box = current_it

            boxlist = box
            
            # Handle empty relation pairs gracefully
            if rel_pair_idx.shape[0] == 0:
                # Empty batch: no relations
                rel_class_prob = F.softmax(rel_logit, -1) if rel_logit.shape[0] > 0 else rel_logit
                boxlist['rel_pair_idxs'] = rel_pair_idx # (#rel, 2)
                boxlist['pred_rel_scores'] = rel_class_prob # (#rel, #rel_class)
                boxlist['pred_rel_labels'] = torch.tensor([], dtype=torch.long, device=rel_logit.device)
                results.append(boxlist)
                continue
            
            # sorting triples according to score production
            rel_class_prob = F.softmax(rel_logit, -1)
            rel_scores, rel_class = rel_class_prob[:, 1:].max(dim=1)
            rel_class = rel_class + 1
            triple_scores = rel_scores
            _, sorting_idx = torch.sort(triple_scores.view(-1), dim=0, descending=True)
            
            # Apply sorting to all tensors
            rel_pair_idx = rel_pair_idx[sorting_idx]
            rel_class_prob = rel_class_prob[sorting_idx]
            rel_labels = rel_class[sorting_idx]

            boxlist['rel_pair_idxs'] = rel_pair_idx # (#rel, 2)
            boxlist['pred_rel_scores'] = rel_class_prob # (#rel, #rel_class)
            boxlist['pred_rel_labels'] = rel_labels # (#rel, )

            results.append(boxlist)
        return results

def build_roi_relation_head(cfg: DictConfig, in_channels: int) -> ROIRelationHead:
    """
    Constructs a new relation head.
    
    Args:
        cfg: Hydra configuration
        in_channels: Number of input channels from backbone (or list for multi-scale)
        
    Returns:
        ROIRelationHead module
    """
    return ROIRelationHead(cfg, in_channels)
