"""
Visualization of DeformableUnionSampler sampling points during training.

Layout: 4 rows (one per image) × 2 columns.
  LEFT  panel: original dataset image + subject box (blue) + object box (orange)
               for the TOP predicted relation only, with sub/obj/rel class labels.
  RIGHT panel: letterboxed P3 feature heatmap + K deformable sampling points
               for that same top relation. Colour = relative attention weight
               (cyan low → magenta high, normalised per-image so convergence is
               visible even when absolute weights are small at epoch 0).
"""

import os
import torch
import torch.nn.functional as F
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────────────

def visualize_sampling_points(
    model,
    viz_batch,
    output_dir: str,
    epoch: int,
    cfg,
    device,
    pred_classes=None,
    obj_classes=None,
    num_samples: int = 8,
) -> None:
    """
    Run a fixed batch through the model (eval mode) and save num_samples separate
    PNG files, one per top-scoring relation pair selected globally across the batch.
    Files: viz/deformable_points_epoch_{epoch:03d}_s{k:02d}.png  (k=0..num_samples-1)
    """
    images, targets, _ = viz_batch

    # ── Locate sampler & P3 proj (handle DDP) ───────────────────────────────
    m = model.module if hasattr(model, "module") else model
    try:
        sampler = m.roi_heads.relation.deformable_union
        p3_proj = m.roi_heads.relation.level_projs[0]
    except AttributeError:
        return

    was_training = model.training
    sampler.last_abs_x = sampler.last_abs_y = None
    sampler.last_weights = sampler.last_b_ids = None
    sampler.store_points = True
    model.eval()

    _p3_feats: list = []
    def _p3_hook(module, inp, out):
        _p3_feats.clear()
        _p3_feats.append(out.detach().cpu())
    hook = p3_proj.register_forward_hook(_p3_hook)

    images_dev = images.to(device)
    targets_dev = [
        {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
        for t in targets
    ]

    try:
        with torch.no_grad():
            out = model(images_dev, targets_dev)
        proposals_out = out if isinstance(out, list) else (
            out[1] if isinstance(out, tuple) and len(out) >= 2 else []
        )
        raw_boxes_list = getattr(m.roi_heads.relation, '_viz_letterboxed_boxes', None)
        # Measure how much delta_proj has moved from zero (proxy for learning progress)
        try:
            _dp = m.roi_heads.relation.deformable_union.delta_proj
            delta_l2 = float(_dp.weight.detach().abs().mean().item())
        except Exception:
            delta_l2 = float('nan')
    except Exception as exc:
        print(f"[viz_deformable_points] Forward pass failed: {exc}")
        hook.remove(); sampler.store_points = False
        if was_training:
            model.train(); (m.module if hasattr(m, "module") else m).backbone.eval()
        return
    finally:
        hook.remove()

    sampler.store_points = False

    abs_x          = sampler.last_abs_x        # [N_total, K]
    abs_y          = sampler.last_abs_y
    weights        = sampler.last_weights       # [N_total, K, L]
    b_ids          = sampler.last_b_ids         # [N_total]
    last_delta_all = getattr(sampler, 'last_delta', None)  # [N_total, K, 2] or None

    if abs_x is None or abs_x.shape[0] == 0:
        sampler.last_abs_x = None
        if was_training:
            model.train(); m.backbone.eval()
        return

    p3_act = None
    if _p3_feats:
        p3_act = _p3_feats[0].float().norm(dim=1)   # [B, H3, W3]

    _K = abs_x.shape[1]                                   # sampling points per pair
    _L = weights.shape[2] if weights.ndim == 3 else 1     # feature levels

    # Letterbox-space boxes (for RIGHT panel heatmap): abs coords in 640×640 space.
    # These are set from p["lb_boxes"] in deformable_relation_head at inference time.
    # Original-image boxes (for LEFT panel source image): abs coords in original-image space.
    orig_boxes_list = getattr(m.roi_heads.relation, '_viz_original_boxes', None)
    # raw_boxes_list already retrieved above as _viz_letterboxed_boxes

    B = images.tensors.shape[0]
    pixel_mean = torch.tensor(cfg.input.pixel_mean, dtype=torch.float32).view(3, 1, 1)

    # ── Collect all (img_idx, pair_rank, combined_score) across the batch ────
    # Rank globally so num_samples best pairs across ALL images are visualized —
    # not just the top-1 per image.  Exposes diverse relation types per epoch.
    all_samples = []  # [(img_idx, pair_rank, combined_score), ...]
    for i in range(B):
        if i >= len(proposals_out) or proposals_out[i] is None:
            continue
        prop       = proposals_out[i]
        rel_pairs  = prop.get("rel_pair_idxs",   None)
        rel_scores = prop.get("pred_rel_scores",  None)
        obj_scores = prop.get("pred_scores",      None)
        if rel_pairs is None or rel_scores is None or obj_scores is None or len(rel_pairs) == 0:
            continue
        fg_scores = rel_scores[:, 1:].max(dim=-1).values
        s_idxs    = rel_pairs[:, 0]
        o_idxs    = rel_pairs[:, 1]
        combined  = fg_scores * obj_scores[s_idxs] * obj_scores[o_idxs]
        for rank in range(len(rel_pairs)):
            all_samples.append((i, rank, combined[rank].item()))

    all_samples.sort(key=lambda x: x[2], reverse=True)
    selected = all_samples[:num_samples]

    if not selected:
        if was_training:
            model.train(); m.backbone.eval()
        return

    viz_dir = os.path.join(output_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    for k, (img_idx, pair_rank, score) in enumerate(selected):
        prop         = proposals_out[img_idx]
        rel_pairs_p  = prop.get("rel_pair_idxs",   None)
        rel_labels_p = prop.get("pred_rel_labels",  None)
        obj_preds_i  = prop.get("pred_labels",      None)

        s_idx     = rel_pairs_p[pair_rank, 0].item()
        o_idx     = rel_pairs_p[pair_rank, 1].item()
        rel_label = rel_labels_p[pair_rank].item() if rel_labels_p is not None else 0
        s_lbl = _obj_name(obj_classes, int(obj_preds_i[s_idx])) if obj_preds_i is not None else "sub"
        o_lbl = _obj_name(obj_classes, int(obj_preds_i[o_idx])) if obj_preds_i is not None else "obj"
        r_lbl = _rel_name(pred_classes, rel_label)

        # ── Letterbox image ───────────────────────────────────────────────
        img_t = images.tensors[img_idx].float().cpu()
        lb_np = (img_t + pixel_mean).clamp(0, 255).byte().numpy()
        lb_np = lb_np[::-1].transpose(1, 2, 0).copy()   # BGR→RGB, CHW→HWC
        H_lb, W_lb = lb_np.shape[:2]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            f"Epoch {epoch}  —  [{s_lbl}]  {r_lbl}  [{o_lbl}]"
            f"  (sample {k + 1}/{len(selected)}, score={score:.3f})",
            fontsize=11, y=1.002,
        )
        ax_left, ax_right = axes[0], axes[1]
        ax_left.axis("off"); ax_right.axis("off")

        # ════════════════════════════════════════════════════════════════════
        # LEFT PANEL
        # ════════════════════════════════════════════════════════════════════
        orig_img_np = None
        img_path = targets[img_idx].get("image_path", None) if img_idx < len(targets) else None
        if img_path and os.path.isfile(str(img_path)):
            try:
                orig_img_np = np.array(Image.open(img_path).convert("RGB"))
            except Exception:
                pass

        if orig_img_np is not None:
            ax_left.imshow(orig_img_np)
            H_disp, W_disp = orig_img_np.shape[:2]
            _imsz = targets[img_idx].get("image_size", (W_disp, H_disp)) if img_idx < len(targets) else (W_disp, H_disp)
            orig_W, orig_H = _imsz
            sx = W_disp / float(orig_W); sy = H_disp / float(orig_H)
            ax_left.set_title(f"img {img_idx}  (original)", fontsize=9, pad=3)
        else:
            ax_left.imshow(lb_np)
            H_disp, W_disp = H_lb, W_lb
            sx, sy = 1.0, 1.0
            ax_left.set_title(f"img {img_idx}  (letterboxed)", fontsize=9, pad=3)

        if orig_boxes_list is not None and img_idx < len(orig_boxes_list):
            o_raw = orig_boxes_list[img_idx].cpu().numpy()
            s_box = o_raw[s_idx] * [sx, sy, sx, sy]
            o_box = o_raw[o_idx] * [sx, sy, sx, sy]
            _draw_box(ax_left, s_box, "dodgerblue", lw=2.5)
            _draw_box(ax_left, o_box, "darkorange",  lw=2.5)
            ax_left.text(s_box[0]+2, s_box[1]-4, s_lbl, fontsize=8, color="white", va="bottom",
                         bbox=dict(facecolor="dodgerblue", alpha=0.85, pad=1.5, boxstyle="round,pad=0.2"))
            ax_left.text(o_box[0]+2, o_box[1]-4, o_lbl, fontsize=8, color="white", va="bottom",
                         bbox=dict(facecolor="darkorange", alpha=0.85, pad=1.5, boxstyle="round,pad=0.2"))
            u_cx = (min(s_box[0], o_box[0]) + max(s_box[2], o_box[2])) * 0.5
            u_cy = (min(s_box[1], o_box[1]) + max(s_box[3], o_box[3])) * 0.5
            ax_left.text(u_cx, u_cy, r_lbl, fontsize=10, color="white", ha="center", va="center",
                         fontweight="bold",
                         bbox=dict(facecolor="black", alpha=0.65, pad=2, boxstyle="round,pad=0.3"))

        # ════════════════════════════════════════════════════════════════════
        # RIGHT PANEL: letterboxed image + P3 heatmap + sampling points
        # ════════════════════════════════════════════════════════════════════
        ax_right.imshow(lb_np)
        if p3_act is not None and img_idx < p3_act.shape[0]:
            heat = p3_act[img_idx].float().numpy()                  # [H3, W3]
            heat_up = F.interpolate(
                torch.from_numpy(heat).unsqueeze(0).unsqueeze(0),
                size=(H_lb, W_lb), mode='bilinear', align_corners=False,
            ).squeeze().numpy()                                     # [H_lb, W_lb]
            _vmax = float(heat_up.max()) if heat_up.max() > 0 else 1.0
            ax_right.imshow(heat_up, alpha=0.45, cmap='inferno',
                            vmin=0.0, vmax=_vmax, zorder=2)
        _n_sv = max(1, _K // 3); _n_ov = max(1, _K // 3); _n_uv = _K - _n_sv - _n_ov
        ax_right.set_title(
            f"img {img_idx}  K={_K} ({_n_sv}s+{_n_ov}o+{_n_uv}u) × L={_L}  |  Δ|w|={delta_l2:.4f}",
            fontsize=9, pad=3)

        if raw_boxes_list is not None and img_idx < len(raw_boxes_list):
            lb_boxes = raw_boxes_list[img_idx].cpu().numpy()
            _draw_box(ax_right, lb_boxes[s_idx], "dodgerblue", lw=1.5)
            _draw_box(ax_right, lb_boxes[o_idx], "darkorange",  lw=1.5)

        img_pair_indices = (b_ids == img_idx).nonzero(as_tuple=False).squeeze(1)
        # pair_rank is the index in post-processor SORTED proposals; sampler stores
        # points in the ORIGINAL unsorted order.  Map (s_idx, o_idx) → original rank.
        g_idx = None
        orig_pairs_list = getattr(m.roi_heads.relation, '_viz_rel_pair_idxs', None)
        if orig_pairs_list is not None and img_idx < len(orig_pairs_list):
            op = orig_pairs_list[img_idx]  # [n_orig, 2] — original unsorted pairs
            match = ((op[:, 0] == s_idx) & (op[:, 1] == o_idx)).nonzero(as_tuple=False)
            if match.numel() > 0:
                orig_rank = int(match[0, 0].item())
                if orig_rank < len(img_pair_indices):
                    g_idx = img_pair_indices[orig_rank]
        if g_idx is None and pair_rank < len(img_pair_indices):
            g_idx = img_pair_indices[pair_rank]   # fallback (old behaviour)
        if g_idx is not None:
            pts_wkl = weights[g_idx].numpy()              # [K, L]
            if pts_wkl.ndim == 1:
                pts_wkl = pts_wkl[:, None]
            K_pts, L_pts = pts_wkl.shape

            _ax = abs_x[g_idx].numpy()  # [K, L] or [K]
            _ay = abs_y[g_idx].numpy()
            if _ax.ndim == 1:
                _ax = _ax[:, None].repeat(L_pts, axis=1)
                _ay = _ay[:, None].repeat(L_pts, axis=1)
            pts_x = _ax * W_lb   # [K, L] letterbox px
            pts_y = _ay * H_lb   # [K, L]

            drift = None
            if last_delta_all is not None and g_idx < len(last_delta_all):
                _d = last_delta_all[g_idx].float()
                drift = float(_d.norm(dim=-1).mean().item())

            _n_s = max(1, K_pts // 3); _n_o = max(1, K_pts // 3); _n_u = K_pts - _n_s - _n_o
            _ec  = ['dodgerblue'] * _n_s + ['darkorange'] * _n_o + ['limegreen'] * _n_u
            _rl  = ['s'] * _n_s + ['o'] * _n_o + ['u'] * _n_u
            _MK  = ['^', 'o', 's', 'D']
            _LVL = ['P3', 'P4', 'P5', 'P6']
            _vmax = float(pts_wkl.max()) if pts_wkl.max() > 0 else 1.0 / max(K_pts * L_pts, 1)
            _dy   = np.linspace(-3, 3, L_pts) if L_pts > 1 else np.array([0.0])
            for l_i in range(L_pts):
                ax_right.scatter(
                    pts_x[:, l_i], pts_y[:, l_i] + _dy[l_i],
                    c=pts_wkl[:, l_i], cmap='cool',
                    s=110, alpha=0.95,
                    marker=_MK[l_i] if l_i < len(_MK) else 'o',
                    vmin=0.0, vmax=_vmax,
                    zorder=8 + l_i, linewidths=1.8, edgecolors=_ec,
                    label=_LVL[l_i] if l_i < len(_LVL) else f'L{l_i}',
                )
            for k_i in range(K_pts):
                ax_right.text(pts_x[k_i].mean() + 5, pts_y[k_i].mean() - 5,
                              f'{_rl[k_i]}{k_i}',
                              fontsize=7, color=_ec[k_i], zorder=12, fontweight='bold')

            from matplotlib.lines import Line2D
            _lvl_h = [
                Line2D([0], [0], marker=_MK[l], color='w',
                       markerfacecolor='gray', markeredgecolor='white',
                       markersize=7, label=_LVL[l])
                for l in range(L_pts)
            ]
            _role_h = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                       markeredgecolor='dodgerblue', markeredgewidth=2,
                       markersize=7, label=f'sub ({_n_s})'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                       markeredgecolor='darkorange', markeredgewidth=2,
                       markersize=7, label=f'obj ({_n_o})'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                       markeredgecolor='limegreen', markeredgewidth=2,
                       markersize=7, label=f'union ({_n_u})'),
            ]
            ax_right.legend(handles=_lvl_h + _role_h,
                            fontsize=7, loc='upper left', framealpha=0.7, markerscale=0.8)
            try:
                sm = ScalarMappable(cmap='cool', norm=Normalize(0.0, _vmax))
                sm.set_array([])
                cb = plt.colorbar(sm, ax=ax_right, fraction=0.03, pad=0.02)
                cb.ax.tick_params(labelsize=6)
                _drift_str = f"  drift={drift:.4f}" if drift is not None else ""
                cb.set_label(f"attn/level (max={_vmax:.3f}){_drift_str}", fontsize=6)
            except Exception:
                pass

        plt.tight_layout(pad=1.0)
        out_path = os.path.join(viz_dir, f"deformable_points_epoch_{epoch:03d}_s{k:02d}.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[viz_deformable_points] Saved → {out_path}")

    sampler.last_abs_x = sampler.last_abs_y = None
    sampler.last_weights = sampler.last_b_ids = None
    if was_training:
        model.train()
        m.backbone.eval()


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _draw_box(ax, box, color, lw=1.5, alpha=0.85):
    """Draw a single xyxy box as a Rectangle patch."""
    x1, y1, x2, y2 = box
    ax.add_patch(patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=lw, edgecolor=color, facecolor="none", alpha=alpha,
    ))


def _rel_name(pred_classes, idx: int) -> str:
    if pred_classes is None:
        return str(idx)
    if idx == 0:
        return "bg"
    if idx < len(pred_classes):
        return pred_classes[idx]
    return str(idx)


def _obj_name(obj_classes, idx: int) -> str:
    if obj_classes is None:
        return str(idx)
    if idx < len(obj_classes):
        return obj_classes[idx]
    return str(idx)


def get_viz_batch(data_loader, device=None, max_images: int = 4):
    """
    Extract a small fixed batch from a data_loader for visualization.
    Called once before the training loop and cached.

    Returns (images, targets, img_ids) with at most max_images samples.
    """
    for images, targets, img_ids in data_loader:
        if images.tensors.shape[0] > max_images:
            images_tensor = images.tensors[:max_images]
            try:
                from sgg_benchmark.structures.image_list import ImageList
                images = ImageList(images_tensor, images.image_sizes[:max_images])
            except Exception:
                images.tensors = images_tensor
            targets = targets[:max_images]
            img_ids = img_ids[:max_images] if hasattr(img_ids, "__getitem__") else img_ids
        return images, targets, img_ids
    return None

