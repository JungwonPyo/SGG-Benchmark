#!/usr/bin/env python3
"""Evaluate an ONNX SGG model on the PSG test set and compare accuracy with the PyTorch baseline.

The script:
  1. Loads the checkpoint config and patches remote paths to local ones.
  2. Builds the PSG test dataset via the existing data pipeline.
  3. Runs ONNX Runtime inference on every test image (letterbox pre-processing,
     identical to the webcam demo).
  4. Converts raw ONNX outputs (boxes + rels) to the prediction-dict format
     expected by do_sgg_evaluation.
  5. Computes R@K, mR@K, F1@K, zR@K and bbox mAP.

Usage:
    # Evaluate the already-exported model.onnx in the checkpoint folder:
    python tools/eval_onnx_psg.py \\
        --run-dir checkpoints/PSG/react++_yolo12m

    # Override data or ONNX paths explicitly:
    python tools/eval_onnx_psg.py \\
        --run-dir checkpoints/PSG/react++_yolo12m \\
        --onnx-path checkpoints/PSG/react++_yolo12m/model.onnx \\
        --data-dir  datasets/PSG/coco_format \\
        --provider  CUDAExecutionProvider
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# ── Project root ───────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from omegaconf import OmegaConf, open_dict

from sgg_benchmark.config import get_cfg
from sgg_benchmark.data import make_data_loader
from sgg_benchmark.data.datasets.evaluation.sgg_eval import do_sgg_evaluation
from sgg_benchmark.utils.env import setup_environment
from sgg_benchmark.utils.logger import setup_logger
from sgg_benchmark.utils.miscellaneous import mkdir

setup_environment()

# ── Default local paths ────────────────────────────────────────────────────────
_DEFAULT_DATA_DIR  = str(project_root / "datasets" / "PSG" / "coco_format")
_DEFAULT_GLOVE_DIR = str(project_root)


# ══════════════════════════════════════════════════════════════════════════════
# Image pre-processing  (matches onnx_model._preprocess_for_onnx exactly)
# ══════════════════════════════════════════════════════════════════════════════

def letterbox(bgr: np.ndarray, size: int = 640):
    """Letterbox-pad *bgr* to a square of side *size*.

    Returns
    -------
    padded : np.ndarray  – (size, size, 3) uint8 BGR image
    gain   : float       – scaling factor applied to the original image
    pad_left : int       – pixels of padding on the left
    pad_top  : int       – pixels of padding on the top
    """
    h, w = bgr.shape[:2]
    gain = min(size / h, size / w)
    nw = int(round(w * gain))
    nh = int(round(h * gain))

    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_top    = int(round((size - nh) / 2 - 0.1))
    pad_bottom = int(round((size - nh) / 2 + 0.1))
    pad_left   = int(round((size - nw) / 2 - 0.1))
    pad_right  = int(round((size - nw) / 2 + 0.1))

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return padded, gain, pad_left, pad_top


def preprocess(bgr: np.ndarray, size: int = 640):
    """Letterbox → BGR→RGB → CHW → float32 /255.  Returns (1, 3, H, W) array."""
    padded, gain, pad_left, pad_top = letterbox(bgr, size)
    rgb = padded[:, :, ::-1]                          # BGR → RGB
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))  # HWC → CHW
    inp = chw.astype(np.float32) / 255.0
    return inp[None, ...], gain, pad_left, pad_top    # (1, 3, H, W)


def undo_letterbox(boxes: np.ndarray, gain: float, pad_left: int, pad_top: int,
                   orig_w: int, orig_h: int) -> np.ndarray:
    """Undo the letterbox transform on (N, 4+) boxes [x1, y1, x2, y2, ...]."""
    if len(boxes) == 0:
        return boxes
    b = boxes.copy()
    b[:, [0, 2]] = (b[:, [0, 2]] - pad_left) / gain
    b[:, [1, 3]] = (b[:, [1, 3]] - pad_top)  / gain
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, orig_w)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, orig_h)
    return b


# ══════════════════════════════════════════════════════════════════════════════
# Convert ONNX outputs → prediction dict for do_sgg_evaluation
# ══════════════════════════════════════════════════════════════════════════════

def build_pred_dict(
    boxes_raw: np.ndarray,   # (N, 6) [x1,y1,x2,y2, label(1-idx), score]
    rels_raw:  np.ndarray,   # (M, 5) [subj, obj, label(1-idx), tri_score, rel_score]
    orig_w: int,
    orig_h: int,
    num_rel_classes: int,    # = cfg.model.roi_relation_head.num_classes (e.g. 57 for PSG)
) -> dict:
    """Build a prediction dict compatible with evaluate_relation_of_one_image.

    Expected keys and shapes
    ------------------------
    boxes           : (N, 4)  float  x1y1x2y2  in original-image pixels
    mode            : str     'xyxy'
    pred_labels     : (N,)    int64  1-indexed object class
    pred_scores     : (N,)    float32 confidence
    rel_pair_idxs   : (M, 2)  int64
    pred_rel_scores : (M, num_rel_classes)  float32
                      col-0 = background, col-k = score for predicate k
    pred_rel_labels : (M,)    int64  (convenience; not strictly required)
    image_size      : (w, h)  tuple  original image dimensions
    """
    empty = dict(
        boxes=torch.zeros((0, 4), dtype=torch.float32),
        mode="xyxy",
        pred_labels=torch.zeros(0, dtype=torch.int64),
        pred_scores=torch.zeros(0, dtype=torch.float32),
        rel_pair_idxs=torch.zeros((0, 2), dtype=torch.int64),
        pred_rel_scores=torch.zeros((0, num_rel_classes), dtype=torch.float32),
        pred_rel_labels=torch.zeros(0, dtype=torch.int64),
        image_size=(orig_w, orig_h),
    )

    if len(boxes_raw) == 0:
        return empty

    boxes  = torch.from_numpy(boxes_raw[:, :4].astype(np.float32))
    labels = torch.from_numpy(boxes_raw[:, 4].astype(np.int64))
    scores = torch.from_numpy(boxes_raw[:, 5].astype(np.float32))

    if len(rels_raw) == 0:
        return {**empty,
                "boxes": boxes, "pred_labels": labels, "pred_scores": scores}

    subj_idx  = rels_raw[:, 0].astype(np.int64)
    obj_idx   = rels_raw[:, 1].astype(np.int64)
    rel_label = rels_raw[:, 2].astype(np.int64)   # 1-indexed predicate
    rel_score = rels_raw[:, 4].astype(np.float32)  # per-predicate confidence

    rel_pair_idxs = torch.from_numpy(np.stack([subj_idx, obj_idx], axis=1))

    # Build (M, num_rel_classes) score matrix.
    # rel_scores[:, 0] = background (left as 0).
    # rel_scores[:, label] = rel_score  →  argmax(rel_scores[:,1:]) + 1 == label  ✓
    pred_rel_scores = torch.zeros(len(rels_raw), num_rel_classes, dtype=torch.float32)
    for i, (lbl, sc) in enumerate(zip(rel_label, rel_score)):
        if 1 <= lbl < num_rel_classes:
            pred_rel_scores[i, lbl] = float(sc)

    return dict(
        boxes=boxes,
        mode="xyxy",
        pred_labels=labels,
        pred_scores=scores,
        rel_pair_idxs=rel_pair_idxs,
        pred_rel_scores=pred_rel_scores,
        pred_rel_labels=torch.from_numpy(rel_label),
        image_size=(orig_w, orig_h),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Config helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_and_patch_config(run_dir: Path, local_data_dir: str, local_glove_dir: str):
    """Load the checkpoint config and replace remote cluster paths with local ones."""
    for candidate in ("config.yml", "hydra_config.yaml", "config.yaml"):
        cfg_file = run_dir / candidate
        if cfg_file.exists():
            break
    else:
        raise FileNotFoundError(f"No config file found in {run_dir}")

    yaml_cfg = OmegaConf.load(str(cfg_file))

    with open_dict(yaml_cfg):
        # Patch catalog data_dir  (works for both PSG and RelationDataset catalogs)
        try:
            for key in yaml_cfg.datasets.catalog:
                yaml_cfg.datasets.catalog[key]["data_dir"] = local_data_dir
        except Exception:
            pass

        # Patch glove_dir if the stored path doesn't exist
        glove = OmegaConf.select(yaml_cfg, "glove_dir")
        if glove and not os.path.isdir(str(glove)):
            yaml_cfg.glove_dir = local_glove_dir

        # Force output_dir to the run_dir so stats cache lands there
        yaml_cfg.output_dir = str(run_dir)

        # Ensure we evaluate in sgdet mode (both flags = False)
        try:
            yaml_cfg.model.roi_relation_head.use_gt_box          = False
            yaml_cfg.model.roi_relation_head.use_gt_object_label = False
        except Exception:
            pass

    return get_cfg(yaml_cfg)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ONNX model on PSG test set (R@K, mR@K, F1@K, zR@K, mAP)"
    )
    parser.add_argument("--run-dir", required=True,
                        help="Checkpoint folder containing config.yml and model.onnx")
    parser.add_argument("--onnx-path", default=None,
                        help="Explicit ONNX file path (default: <run-dir>/model.onnx)")
    parser.add_argument("--data-dir", default=None,
                        help="Local PSG coco_format directory "
                             f"(default: {_DEFAULT_DATA_DIR})")
    parser.add_argument("--provider", default="CUDAExecutionProvider",
                        choices=["CUDAExecutionProvider", "CPUExecutionProvider", "TensorrtExecutionProvider"],
                        help="ONNX Runtime execution provider")
    parser.add_argument("--input-size", type=int, default=640,
                        help="Letterbox size used at export (default: 640)")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Maximum number of relations per image to keep (default: 100)")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write logs and result JSONs "
                             "(default: <run-dir>/inference_onnx)")
    parser.add_argument("--num-images", type=int, default=-1,
                        help="Limit evaluation to the first N images (-1 = all)")
    args = parser.parse_args()

    run_dir    = Path(args.run_dir).resolve()
    onnx_path  = args.onnx_path or str(run_dir / "model.onnx")
    output_dir = Path(args.output_dir).resolve() if args.output_dir \
                 else run_dir / "inference_onnx"
    mkdir(str(output_dir))

    logger     = setup_logger("eval_onnx", str(output_dir), 0, filename="eval_onnx.log")

    logger.info(f"Run dir  : {run_dir}")
    logger.info(f"ONNX     : {onnx_path}")

    # ── Config ────────────────────────────────────────────────────────────────
    local_data_dir  = args.data_dir or _DEFAULT_DATA_DIR
    local_glove_dir = _DEFAULT_GLOVE_DIR
    logger.info(f"Data dir : {local_data_dir}")

    cfg = load_and_patch_config(run_dir, local_data_dir, local_glove_dir)

    num_rel_classes = cfg.model.roi_relation_head.num_classes  # e.g. 57
    input_size      = args.input_size
    top_k           = args.top_k

    logger.info(f"num_rel_classes={num_rel_classes}, input_size={input_size}, top_k={top_k}")
    logger.info(f"Output   : {output_dir}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    logger.info("Building PSG test data loader …")
    data_loaders = make_data_loader(cfg, mode="test", is_distributed=False)
    loader  = data_loaders[0] if isinstance(data_loaders, (list, tuple)) else data_loaders
    dataset = loader.dataset
    n_total = len(dataset)
    n_eval  = n_total if args.num_images <= 0 else min(args.num_images, n_total)
    logger.info(f"Test set : {n_total} images  (evaluating {n_eval})")

    # ── ONNX Runtime session ──────────────────────────────────────────────────
    import onnxruntime as ort

    logger.info(f"Loading ONNX model from {onnx_path} with {args.provider} …")
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=[args.provider, "CPUExecutionProvider"],
        )
    except Exception as exc:
        logger.error(f"Failed to create ONNX session: {exc}")
        raise

    input_name = session.get_inputs()[0].name
    logger.info(f"ONNX session ready.  Providers: {session.get_providers()}")
    logger.info(f"  Input : {session.get_inputs()[0].name}  "
                f"{session.get_inputs()[0].shape}")
    for out in session.get_outputs():
        logger.info(f"  Output: {out.name}  {out.shape}")

    # ── Warmup ────────────────────────────────────────────────────────────────
    logger.info("Warming up ONNX session …")
    dummy = np.random.rand(1, 3, input_size, input_size).astype(np.float32)
    for _ in range(3):
        session.run(None, {input_name: dummy})
    logger.info("Warmup done.")

    # ── Inference loop ────────────────────────────────────────────────────────
    predictions = []
    timings     = []
    timings_full = []

    try:
        from tqdm import tqdm as _tqdm
        indices = _tqdm(range(n_eval), desc="ONNX Inference", dynamic_ncols=True)
    except ImportError:
        indices = range(n_eval)

    for i in indices:
        img_info = dataset.get_img_info(i)
        orig_h   = int(img_info.get("height", input_size))
        orig_w   = int(img_info.get("width",  input_size))

        # Load raw BGR image
        t_full_start = time.perf_counter()
        img_path = dataset.filenames[i]
        bgr = cv2.imread(img_path)
        if bgr is None:
            logger.warning(f"Cannot read image: {img_path} — using empty prediction")
            predictions.append(build_pred_dict(
                np.zeros((0, 6), np.float32),
                np.zeros((0, 5), np.float32),
                orig_w, orig_h, num_rel_classes,
            ))
            continue

        # Pre-process
        inp, gain, pad_left, pad_top = preprocess(bgr, input_size)

        # ONNX inference
        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: inp})
        timings.append((time.perf_counter() - t0) * 1000.0)

        # Raw outputs:  boxes (N,6)  rels (M,5)
        boxes_raw = outputs[0].copy() if len(outputs) > 0 else np.zeros((0, 6), np.float32)
        rels_raw  = outputs[1].copy() if len(outputs) > 1 else np.zeros((0, 5), np.float32)

        # Undo letterbox on boxes
        if len(boxes_raw) > 0:
            boxes_raw = undo_letterbox(boxes_raw, gain, pad_left, pad_top, orig_w, orig_h)

        # Keep top-k relations by triplet score (col 3)
        if len(rels_raw) > top_k:
            order    = np.argsort(rels_raw[:, 3])[::-1][:top_k]
            rels_raw = rels_raw[order]

        pred = build_pred_dict(boxes_raw, rels_raw, orig_w, orig_h, num_rel_classes)
        predictions.append(pred)

        timings_full.append((time.perf_counter() - t_full_start) * 1000.0)

    # ── SGG Evaluation ────────────────────────────────────────────────────────
    logger.info("Running SGG evaluation (R@K / mR@K / F1@K / zR@K + bbox mAP) …")

    if cfg.datasets.test:
        dataset_name = cfg.datasets.test[0]
    elif cfg.datasets.name:
        dataset_name = cfg.datasets.name
    elif cfg.datasets.catalog:
        # Derive a test-split name from the catalog keys (prefer a key ending in 'test')
        catalog_keys = list(cfg.datasets.catalog.keys())
        test_keys = [k for k in catalog_keys if "test" in k.lower()]
        dataset_name = test_keys[0] if test_keys else catalog_keys[0]
    else:
        dataset_name = ""

    result_dict = do_sgg_evaluation(
        cfg=cfg,
        dataset=dataset,
        dataset_name=dataset_name,
        predictions=predictions,
        output_folder=str(output_dir),
        logger=logger,
        iou_types=["relations", "bbox"],
    )

    # ── Timing summary ────────────────────────────────────────────────────────
    mean_ms   = float(np.mean(timings))   if timings else 0.0
    median_ms = float(np.median(timings)) if timings else 0.0
    fps       = 1000.0 / mean_ms          if mean_ms > 0 else 0.0
    logger.info(
        f"Inference timing (ONNX forward only):  "
        f"mean={mean_ms:.1f} ms  median={median_ms:.1f} ms  FPS={fps:.1f}"
    )

    # ── Full Timing summary ────────────────────────────────────────────────────────
    mean_full_ms   = float(np.mean(timings_full))   if timings_full else 0.0
    median_full_ms = float(np.median(timings_full)) if timings_full else 0.0
    fps_full       = 1000.0 / mean_full_ms          if mean_full_ms > 0 else 0.0
    logger.info(
        f"Full timing (load + pre-process + ONNX forward):  "
        f"mean={mean_full_ms:.1f} ms  median={median_full_ms:.1f} ms  FPS={fps_full:.1f}"
    )

    # ── Save summary JSON ─────────────────────────────────────────────────────
    summary = {
        "onnx_path":       onnx_path,
        "num_images_eval": n_eval,
        "mean_latency_ms": round(mean_ms,   2),
        "median_latency_ms": round(median_ms, 2),
        "median_full_latency_ms": round(median_full_ms, 2),
        "mean_full_latency_ms": round(mean_full_ms, 2),
        "fps":             round(fps,       2),
        "fps_full":        round(fps_full,  2),
    }

    def _to_serialisable(v):
        if isinstance(v, (np.integer, np.floating)):
            return float(v)
        if isinstance(v, torch.Tensor):
            return v.item() if v.numel() == 1 else v.tolist()
        return v

    if isinstance(result_dict, dict):
        for k, v in result_dict.items():
            if isinstance(v, dict):
                summary[k] = {kk: _to_serialisable(vv) for kk, vv in v.items()}
            else:
                summary[k] = _to_serialisable(v)

    summary_path = output_dir / "onnx_eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary JSON saved to: {summary_path}")


if __name__ == "__main__":
    main()
