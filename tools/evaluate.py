#!/usr/bin/env python3
"""Unified SGG evaluation and latency benchmark.

Modes
-----
  Full eval (PyTorch):     --model-dir <dir>
  Full eval (ONNX):        --model-dir <dir> --onnx <path>
  Eval + latency:          --model-dir <dir> [--onnx <path>] --latency
  Latency only (N images): --model-dir <dir> [--onnx <path>] --skip-eval

Examples
--------
  python tools/evaluate.py --model-dir checkpoints/PSG/react++_yolo12m
  python tools/evaluate.py --model-dir checkpoints/PSG/react++_yolo12m \\
      --onnx checkpoints/PSG/react++_yolo12m/model.onnx
  python tools/evaluate.py --model-dir checkpoints/PSG/react++_yolo12m --latency
  python tools/evaluate.py --model-dir checkpoints/PSG/react++_yolo12m \\
      --onnx checkpoints/PSG/react++_yolo12m/model.onnx --skip-eval --num-images 200
  # Evaluate both PyTorch and ONNX in one call:
  python tools/evaluate.py --model-dir checkpoints/PSG/react++_yolo12m --compare
  python tools/evaluate.py --model-dir checkpoints/PSG/react++_yolo12m --compare --latency
"""

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict

# ── Fix LD_LIBRARY_PATH for cuDNN / TensorRT before importing ort ─────────
try:
    import site
    _site_dirs = site.getsitepackages()
    if hasattr(site, "getusersitepackages"):
        _site_dirs.append(site.getusersitepackages())
    _extra = []
    for _sp in _site_dirs:
        if not os.path.exists(_sp):
            continue
        for _item in os.listdir(_sp):
            if _item.endswith("_libs") or _item.endswith(".libs"):
                _extra.append(os.path.join(_sp, _item))
        _nv = os.path.join(_sp, "nvidia")
        if os.path.exists(_nv):
            for _sub in os.listdir(_nv):
                _lib = os.path.join(_nv, _sub, "lib")
                if os.path.exists(_lib):
                    _extra.append(_lib)
                    if _sub == "cudnn":
                        for _so in ["libcudnn_ops.so.9", "libcudnn_adv.so.9",
                                    "libcudnn_cnn.so.9", "libcudnn.so.9"]:
                            _p = os.path.join(_lib, _so)
                            if os.path.exists(_p):
                                try:
                                    ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
                                except Exception:
                                    pass
    if _extra:
        _cur = os.environ.get("LD_LIBRARY_PATH", "")
        _new = ":".join(dict.fromkeys(_extra))
        os.environ["LD_LIBRARY_PATH"] = f"{_new}:{_cur}" if _cur else _new
except Exception:
    pass

import onnxruntime as ort  # noqa: E402  (after LD fix)

# ── Project root ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from sgg_benchmark.config import get_cfg
from sgg_benchmark.data import make_data_loader
from sgg_benchmark.data.datasets.evaluation.sgg_eval import do_sgg_evaluation
from sgg_benchmark.modeling.detector import build_detection_model
from sgg_benchmark.structures.image_list import ImageList
from sgg_benchmark.utils.checkpoint import DetectronCheckpointer
from sgg_benchmark.utils.comm import get_rank
from sgg_benchmark.utils.env import setup_environment
from sgg_benchmark.utils.logger import setup_logger
from sgg_benchmark.utils.miscellaneous import mkdir

setup_environment()


# ══════════════════════════════════════════════════════════════════════════════
# Image helpers (ONNX pre / post-processing)
# ══════════════════════════════════════════════════════════════════════════════

def letterbox(bgr: np.ndarray, size: int = 640):
    h, w = bgr.shape[:2]
    gain = min(size / h, size / w)
    nw, nh = int(round(w * gain)), int(round(h * gain))
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
    padded, gain, pad_left, pad_top = letterbox(bgr, size)
    rgb = padded[:, :, ::-1]
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))
    inp = chw.astype(np.float32) / 255.0
    return inp[None, ...], gain, pad_left, pad_top


def undo_letterbox(boxes: np.ndarray, gain: float, pad_left: int, pad_top: int,
                   orig_w: int, orig_h: int) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    b = boxes.copy()
    b[:, [0, 2]] = (b[:, [0, 2]] - pad_left) / gain
    b[:, [1, 3]] = (b[:, [1, 3]] - pad_top)  / gain
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, orig_w)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, orig_h)
    return b


def build_pred_dict(boxes_raw: np.ndarray, rels_raw: np.ndarray,
                    orig_w: int, orig_h: int, num_rel_classes: int) -> dict:
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
        return {**empty, "boxes": boxes, "pred_labels": labels, "pred_scores": scores}

    subj_idx  = rels_raw[:, 0].astype(np.int64)
    obj_idx   = rels_raw[:, 1].astype(np.int64)
    rel_label = rels_raw[:, 2].astype(np.int64)
    rel_score = rels_raw[:, 4].astype(np.float32)
    rel_pair_idxs = torch.from_numpy(np.stack([subj_idx, obj_idx], axis=1))

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
# Config / checkpoint helpers
# ══════════════════════════════════════════════════════════════════════════════

def resolve_checkpoint_from_dir(model_dir: str) -> str:
    last_ckpt_file = os.path.join(model_dir, "last_checkpoint")
    if not os.path.exists(last_ckpt_file):
        raise FileNotFoundError(f"No last_checkpoint file found in {model_dir}")
    with open(last_ckpt_file) as f:
        ckpt_path = f.read().strip()
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.abspath(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path


def load_and_patch_config(run_dir: Path, local_data_dir=None):
    for candidate in ("config.yml", "config.yaml", "hydra_config.yaml"):
        cfg_file = run_dir / candidate
        if cfg_file.exists():
            break
    else:
        raise FileNotFoundError(f"No config file found in {run_dir}")

    yaml_cfg = OmegaConf.load(str(cfg_file))
    with open_dict(yaml_cfg):
        if local_data_dir is not None:
            try:
                for key in yaml_cfg.datasets.catalog:
                    yaml_cfg.datasets.catalog[key]["data_dir"] = local_data_dir
            except Exception:
                pass
            try:
                yaml_cfg.datasets.data_dir = local_data_dir
            except Exception:
                pass
        # Patch glove_dir if stored path doesn't exist locally
        glove = OmegaConf.select(yaml_cfg, "glove_dir")
        if glove and not os.path.isdir(str(glove)):
            yaml_cfg.glove_dir = str(_PROJECT_ROOT)
        yaml_cfg.output_dir = str(run_dir)
        # Always evaluate in sgdet mode
        try:
            yaml_cfg.model.roi_relation_head.use_gt_box          = False
            yaml_cfg.model.roi_relation_head.use_gt_object_label = False
        except Exception:
            pass
    return get_cfg(yaml_cfg)


def get_dataset_name(cfg) -> str:
    if OmegaConf.select(cfg, "datasets.test"):
        return cfg.datasets.test[0]
    if OmegaConf.select(cfg, "datasets.name"):
        return cfg.datasets.name
    if OmegaConf.select(cfg, "datasets.catalog"):
        keys = list(cfg.datasets.catalog.keys())
        test_keys = [k for k in keys if "test" in k.lower()]
        return test_keys[0] if test_keys else keys[0]
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# PyTorch helpers
# ══════════════════════════════════════════════════════════════════════════════

def convert_to_dict(obj):
    if isinstance(obj, dict):
        return {k: v.to("cpu") if hasattr(v, "to") else v for k, v in obj.items()}
    if hasattr(obj, "to"):
        return obj.to("cpu")
    return obj


def load_pytorch_model(cfg, checkpoint: str, device: str):
    model = build_detection_model(cfg)
    model.to(device)
    DetectronCheckpointer(cfg, model).load(checkpoint)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# ONNX session helper
# ══════════════════════════════════════════════════════════════════════════════

def load_onnx_session(onnx_path: str, provider: str, logger) -> ort.InferenceSession:
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_opts,
        providers=[provider, "CPUExecutionProvider"],
    )
    logger.info(f"ONNX session ready. Providers: {session.get_providers()}")
    for inp in session.get_inputs():
        logger.info(f"  Input : {inp.name}  {inp.shape}")
    for out in session.get_outputs():
        logger.info(f"  Output: {out.name}  {out.shape}")
    return session


# ══════════════════════════════════════════════════════════════════════════════
# Full evaluation (all test images)
# ══════════════════════════════════════════════════════════════════════════════

def eval_pytorch(model, dataset, device: str, logger) -> tuple:
    """Run full PyTorch inference on all dataset images. Returns (predictions, timing)."""
    dev = torch.device(device)
    model.to(dev)
    model.eval()
    if hasattr(model, "cfg"):
        with open_dict(model.cfg):
            model.cfg.model.device = str(dev)
    if hasattr(model, "backbone") and hasattr(model.backbone, "device"):
        model.backbone.device = str(dev)

    # Warmup on first image
    logger.info(f"Warming up PyTorch on {dev}…")
    img0, _, _ = dataset[0]
    if isinstance(img0, (list, tuple)):
        img0 = img0[0]
    img_info0 = dataset.get_img_info(0)
    w0, h0 = img_info0["width"], img_info0["height"]
    with torch.no_grad():
        wlist = ImageList(img0.unsqueeze(0).to(dev), [(h0, w0)])
        for _ in range(5):
            _ = model(wlist)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    predictions = []
    fwd_times = []
    logger.info(f"Running PyTorch inference on {len(dataset)} images…")
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="PyTorch Eval", dynamic_ncols=True):
            img_info = dataset.get_img_info(i)
            w, h = img_info["width"], img_info["height"]
            img, _, _ = dataset[i]
            if isinstance(img, (list, tuple)):
                img = img[0]
            img_list = ImageList(img.unsqueeze(0).to(dev), [(h, w)])
            t0 = time.perf_counter()
            output = model(img_list)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            fwd_times.append((time.perf_counter() - t0) * 1000.0)
            if isinstance(output, (list, tuple)):
                for ob in output:
                    predictions.append(convert_to_dict(ob))
            else:
                predictions.append(convert_to_dict(output))

    timing = {
        "mean_forward_ms":   float(np.mean(fwd_times)),
        "median_forward_ms": float(np.median(fwd_times)),
    }
    logger.info(f"PyTorch eval done. Mean forward: {timing['mean_forward_ms']:.1f} ms")
    return predictions, timing


def eval_onnx(session, dataset, num_rel_classes: int, input_size: int,
              top_k: int, logger) -> tuple:
    """Run full ONNX inference (cv2 + letterbox) on all dataset images."""
    input_name = session.get_inputs()[0].name

    # Warmup with real images to avoid zero-box CUBLAS errors
    logger.info("Warming up ONNX session…")
    for wi in range(min(5, len(dataset))):
        bgr_w = cv2.imread(dataset.filenames[wi])
        if bgr_w is None:
            continue
        inp_w, _, _, _ = preprocess(bgr_w, input_size)
        session.run(None, {input_name: inp_w})

    predictions = []
    fwd_times, e2e_times = [], []
    logger.info(f"Running ONNX inference on {len(dataset)} images…")
    for i in tqdm(range(len(dataset)), desc="ONNX Eval", dynamic_ncols=True):
        img_info = dataset.get_img_info(i)
        orig_h = int(img_info.get("height", input_size))
        orig_w = int(img_info.get("width",  input_size))

        t_e2e = time.perf_counter()
        bgr = cv2.imread(dataset.filenames[i])
        if bgr is None:
            logger.warning(f"Cannot read image: {dataset.filenames[i]}")
            predictions.append(build_pred_dict(
                np.zeros((0, 6), np.float32), np.zeros((0, 5), np.float32),
                orig_w, orig_h, num_rel_classes,
            ))
            continue

        inp, gain, pad_left, pad_top = preprocess(bgr, input_size)

        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: inp})
        fwd_times.append((time.perf_counter() - t0) * 1000.0)
        e2e_times.append((time.perf_counter() - t_e2e) * 1000.0)

        boxes_raw = outputs[0].copy() if len(outputs) > 0 else np.zeros((0, 6), np.float32)
        rels_raw  = outputs[1].copy() if len(outputs) > 1 else np.zeros((0, 5), np.float32)

        if len(boxes_raw) > 0:
            boxes_raw = undo_letterbox(boxes_raw, gain, pad_left, pad_top, orig_w, orig_h)
        if len(rels_raw) > top_k:
            order    = np.argsort(rels_raw[:, 3])[::-1][:top_k]
            rels_raw = rels_raw[order]

        predictions.append(build_pred_dict(boxes_raw, rels_raw, orig_w, orig_h, num_rel_classes))

    timing = {
        "mean_forward_ms":   float(np.mean(fwd_times))   if fwd_times else 0.0,
        "median_forward_ms": float(np.median(fwd_times)) if fwd_times else 0.0,
        "mean_e2e_ms":       float(np.mean(e2e_times))   if e2e_times else 0.0,
        "median_e2e_ms":     float(np.median(e2e_times)) if e2e_times else 0.0,
    }
    logger.info(
        f"ONNX eval done. "
        f"Mean forward: {timing['mean_forward_ms']:.1f} ms  "
        f"Mean E2E: {timing['mean_e2e_ms']:.1f} ms"
    )
    return predictions, timing


# ══════════════════════════════════════════════════════════════════════════════
# Dedicated latency benchmark (subset with warmup)
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_latency_pytorch(model, dataset, device: str, n: int,
                               num_warmup: int = 10) -> dict:
    """Warmup + latency benchmark on first n images. Returns timing dict."""
    dev = torch.device(device)
    model.to(dev)
    model.eval()
    n = min(n, len(dataset))

    # Pre-load images (dataset transforms already applied)
    images, orig_sizes = [], []
    logger_print = print
    logger_print(f"Pre-loading {n} images for latency benchmark…")
    for i in range(n):
        img_info = dataset.get_img_info(i)
        w, h = img_info["width"], img_info["height"]
        orig_sizes.append((h, w))
        img, _, _ = dataset[i]
        if isinstance(img, (list, tuple)):
            img = img[0]
        images.append(img.unsqueeze(0))

    # Warmup
    print(f"Warming up PyTorch on {dev} ({num_warmup} iterations)…")
    with torch.no_grad():
        wlist = ImageList(images[0].to(dev), [orig_sizes[0]])
        for _ in range(num_warmup):
            _ = model(wlist)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    pre_times, fwd_times = [], []
    print(f"Benchmarking PyTorch on {n} images…")
    with torch.no_grad():
        for i, img in enumerate(tqdm(images, desc="PyTorch Latency")):
            t_pre = time.perf_counter()
            img_dev  = img.to(dev)
            img_list = ImageList(img_dev, [orig_sizes[i]])
            pre_times.append((time.perf_counter() - t_pre) * 1000.0)

            t_fwd = time.perf_counter()
            _ = model(img_list)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            fwd_times.append((time.perf_counter() - t_fwd) * 1000.0)

    e2e = [p + f for p, f in zip(pre_times, fwd_times)]
    return {
        "count":           n,
        "avg_pre_ms":      float(np.mean(pre_times)),
        "avg_forward_ms":  float(np.mean(fwd_times)),
        "avg_e2e_ms":      float(np.mean(e2e)),
        "median_e2e_ms":   float(np.median(e2e)),
        "throughput_fps":  1000.0 / float(np.mean(e2e)) if np.mean(e2e) > 0 else 0.0,
    }


def benchmark_latency_onnx(session, dataset, input_size: int, n: int,
                            num_warmup: int = 10) -> dict:
    """Warmup + E2E latency benchmark (cv2 load + letterbox + forward) on first n images."""
    input_name = session.get_inputs()[0].name
    n = min(n, len(dataset))

    # Warmup with real images so the model sees actual detections (a random dummy
    # can produce zero boxes which causes CUBLAS_STATUS_INVALID_VALUE in Einsum).
    print(f"Warming up ONNX ({num_warmup} iterations, using real images)…")
    for wi in range(min(num_warmup, len(dataset))):
        bgr_w = cv2.imread(dataset.filenames[wi])
        if bgr_w is None:
            continue
        inp_w, _, _, _ = preprocess(bgr_w, input_size)
        session.run(None, {input_name: inp_w})

    pre_times, fwd_times = [], []
    print(f"Benchmarking ONNX on {n} images (E2E: load + letterbox + forward)…")
    for i in tqdm(range(n), desc="ONNX Latency"):
        img_path = dataset.filenames[i]
        t_pre = time.perf_counter()
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        inp, _, _, _ = preprocess(bgr, input_size)
        pre_times.append((time.perf_counter() - t_pre) * 1000.0)

        t_fwd = time.perf_counter()
        session.run(None, {input_name: inp})
        fwd_times.append((time.perf_counter() - t_fwd) * 1000.0)

    e2e = [p + f for p, f in zip(pre_times, fwd_times)]
    return {
        "count":          len(e2e),
        "avg_pre_ms":     float(np.mean(pre_times)),
        "avg_forward_ms": float(np.mean(fwd_times)),
        "avg_e2e_ms":     float(np.mean(e2e)),
        "median_e2e_ms":  float(np.median(e2e)),
        "throughput_fps": 1000.0 / float(np.mean(e2e)) if np.mean(e2e) > 0 else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Output helpers
# ══════════════════════════════════════════════════════════════════════════════

def print_latency_table(name: str, lat: dict):
    print(f"\n{'─' * 60}")
    print(f"  Latency benchmark : {name}")
    print(f"{'─' * 60}")
    print(f"  Images benchmarked : {lat['count']}")
    print(f"  Pre-process  (avg) : {lat['avg_pre_ms']:.2f} ms")
    print(f"  Forward      (avg) : {lat['avg_forward_ms']:.2f} ms")
    print(f"  E2E          (avg) : {lat['avg_e2e_ms']:.2f} ms")
    print(f"  E2E       (median) : {lat['median_e2e_ms']:.2f} ms")
    print(f"  Throughput         : {lat['throughput_fps']:.1f} FPS")
    print(f"{'─' * 60}\n")


def _serialise(v):
    if isinstance(v, Path):            return str(v)
    if isinstance(v, (np.integer, np.floating)): return float(v)
    if isinstance(v, torch.Tensor):
        return v.item() if v.numel() == 1 else v.tolist()
    return v


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Unified SGG evaluation and latency benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="Model directory containing config.yml "
             "(and last_checkpoint for PyTorch eval)",
    )
    parser.add_argument(
        "--onnx", default=None,
        help="Path to .onnx file; if provided, evaluates the ONNX model "
             "instead of the PyTorch checkpoint. Defaults to <model-dir>/model.onnx "
             "when --compare is used.",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run both PyTorch and ONNX evaluation sequentially in a single call. "
             "ONNX path defaults to <model-dir>/model.onnx (override with --onnx).",
    )
    parser.add_argument(
        "--latency", action="store_true",
        help="Run a dedicated warmup+latency benchmark in addition to eval",
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip evaluation; run latency benchmark only "
             "(implies --latency, uses --num-images images)",
    )
    parser.add_argument(
        "--num-images", type=int, default=200,
        help="Number of test-set images for the latency benchmark (default: 200)",
    )
    parser.add_argument("--device",     default="cuda",
                        help="PyTorch device (default: cuda)")
    parser.add_argument(
        "--provider", default="CUDAExecutionProvider",
        choices=["CUDAExecutionProvider", "CPUExecutionProvider",
                 "TensorrtExecutionProvider"],
        help="ONNX Runtime execution provider (default: CUDAExecutionProvider)",
    )
    parser.add_argument("--input-size", type=int, default=640,
                        help="ONNX letterbox input size (default: 640)")
    parser.add_argument("--top-k",      type=int, default=100,
                        help="Max relations per image for ONNX eval (default: 100)")
    parser.add_argument("--data-dir",   default=None,
                        help="Override dataset paths from the checkpoint config")
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for logs and JSONs "
             "(default: <model-dir>/inference_sgdet or inference_onnx)",
    )
    args = parser.parse_args()

    run_dir = Path(args.model_dir).resolve()

    if args.compare:
        _run_compare(args, run_dir)
    else:
        _run_single(args, run_dir, use_onnx=args.onnx is not None)


def _resolve_onnx_path(args, run_dir: Path) -> str:
    """Return absolute ONNX path, falling back to <model-dir>/model.onnx."""
    raw = args.onnx or str(run_dir / "model.onnx")
    onnx_path = str(Path(raw).resolve())
    if not os.path.exists(onnx_path):
        candidates = list(run_dir.glob("react_pp_*.onnx"))
        if candidates:
            return str(candidates[0])
    return onnx_path


def _run_compare(args, run_dir: Path):
    """Evaluate both PyTorch and ONNX sequentially, writing separate output dirs."""
    # Build shared infrastructure (config, dataset) once
    cfg = load_and_patch_config(run_dir, args.data_dir)
    data_loaders = make_data_loader(cfg, mode="test", is_distributed=False)
    loader  = data_loaders[0] if isinstance(data_loaders, (list, tuple)) else data_loaders
    dataset = loader.dataset

    num_obj_classes = len(dataset.ind_to_classes)    if hasattr(dataset, "ind_to_classes")    else 0
    num_rel_classes = len(dataset.ind_to_predicates) if hasattr(dataset, "ind_to_predicates") else 0
    with open_dict(cfg):
        cfg.model.roi_box_head.num_classes      = num_obj_classes
        cfg.model.roi_relation_head.num_classes = num_rel_classes
    dataset_name = get_dataset_name(cfg)

    all_summaries = {}

    # ── 1. PyTorch ──────────────────────────────────────────────────────────
    pt_dir = (Path(args.output_dir).resolve() / "pytorch") if args.output_dir \
             else run_dir / "inference_sgdet"
    mkdir(str(pt_dir))
    pt_logger = setup_logger("sgg_eval_pt", str(pt_dir), get_rank(), filename="eval.log")
    pt_logger.info("=== PyTorch evaluation ===")

    checkpoint = resolve_checkpoint_from_dir(str(run_dir))
    pt_logger.info(f"Loading PyTorch checkpoint from {checkpoint}…")
    model = load_pytorch_model(cfg, checkpoint, args.device)

    pt_summary: dict = {"model_dir": str(run_dir), "mode": "pytorch"}

    if not args.skip_eval:
        predictions, eval_timing = eval_pytorch(model, dataset, args.device, pt_logger)
        pt_logger.info("Running SGG evaluation for PyTorch…")
        result_dict = do_sgg_evaluation(
            cfg=cfg, dataset=dataset, dataset_name=dataset_name,
            predictions=predictions, output_folder=str(pt_dir),
            logger=pt_logger, iou_types=["relations", "bbox"],
        )
        pt_summary["eval_timing"] = eval_timing
        if isinstance(result_dict, dict):
            for k, v in result_dict.items():
                pt_summary[k] = (
                    {kk: _serialise(vv) for kk, vv in v.items()}
                    if isinstance(v, dict) else _serialise(v)
                )

    if args.latency or args.skip_eval:
        lat = benchmark_latency_pytorch(model, dataset, args.device, args.num_images)
        print_latency_table("PyTorch", lat)
        with open(pt_dir / "latency.json", "w") as f:
            json.dump(lat, f, indent=2)
        pt_summary["latency"] = lat

    with open(pt_dir / "eval_summary.json", "w") as f:
        json.dump(pt_summary, f, indent=2)
    pt_logger.info(f"PyTorch summary saved to: {pt_dir / 'eval_summary.json'}")
    all_summaries["pytorch"] = pt_summary

    # Free GPU memory before loading ONNX
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 2. ONNX ─────────────────────────────────────────────────────────────
    onnx_path = _resolve_onnx_path(args, run_dir)
    onnx_dir  = (Path(args.output_dir).resolve() / "onnx") if args.output_dir \
                else run_dir / "inference_onnx"
    mkdir(str(onnx_dir))
    onnx_logger = setup_logger("sgg_eval_onnx", str(onnx_dir), get_rank(), filename="eval.log")
    onnx_logger.info("=== ONNX evaluation ===")
    onnx_logger.info(f"Loading ONNX from {onnx_path}…")
    session = load_onnx_session(onnx_path, args.provider, onnx_logger)

    onnx_summary: dict = {"model_dir": str(run_dir), "mode": "onnx", "onnx_path": onnx_path}

    if not args.skip_eval:
        predictions, eval_timing = eval_onnx(
            session, dataset, num_rel_classes, args.input_size, args.top_k, onnx_logger)
        onnx_logger.info("Running SGG evaluation for ONNX…")
        result_dict = do_sgg_evaluation(
            cfg=cfg, dataset=dataset, dataset_name=dataset_name,
            predictions=predictions, output_folder=str(onnx_dir),
            logger=onnx_logger, iou_types=["relations", "bbox"],
        )
        onnx_summary["eval_timing"] = eval_timing
        if isinstance(result_dict, dict):
            for k, v in result_dict.items():
                onnx_summary[k] = (
                    {kk: _serialise(vv) for kk, vv in v.items()}
                    if isinstance(v, dict) else _serialise(v)
                )

    if args.latency or args.skip_eval:
        lat = benchmark_latency_onnx(session, dataset, args.input_size, args.num_images)
        print_latency_table("ONNX", lat)
        with open(onnx_dir / "latency.json", "w") as f:
            json.dump(lat, f, indent=2)
        onnx_summary["latency"] = lat

    with open(onnx_dir / "eval_summary.json", "w") as f:
        json.dump(onnx_summary, f, indent=2)
    onnx_logger.info(f"ONNX summary saved to: {onnx_dir / 'eval_summary.json'}")
    all_summaries["onnx"] = onnx_summary

    # ── Combined summary ────────────────────────────────────────────────────
    combined_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir
    with open(combined_dir / "comparison_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nComparison summary saved to: {combined_dir / 'comparison_summary.json'}")


def _run_single(args, run_dir: Path, use_onnx: bool):
    """Original single-backend evaluation path."""
    # ── Output directory ────────────────────────────────────────────────────
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = run_dir / ("inference_onnx" if use_onnx else "inference_sgdet")
    mkdir(str(output_dir))

    logger = setup_logger("sgg_eval", str(output_dir), get_rank(), filename="eval.log")
    logger.info(f"Model dir  : {run_dir}")
    logger.info(f"Output dir : {output_dir}")
    logger.info(f"Backend    : {'ONNX  (' + args.provider + ')' if use_onnx else 'PyTorch (' + args.device + ')'}")
    logger.info(f"skip-eval  : {args.skip_eval}  latency: {args.latency}  num-images: {args.num_images}")

    # ── Config ──────────────────────────────────────────────────────────────
    cfg = load_and_patch_config(run_dir, args.data_dir)

    # ── Dataset ─────────────────────────────────────────────────────────────
    logger.info("Building test data loader…")
    data_loaders = make_data_loader(cfg, mode="test", is_distributed=False)
    loader  = data_loaders[0] if isinstance(data_loaders, (list, tuple)) else data_loaders
    dataset = loader.dataset

    num_obj_classes = len(dataset.ind_to_classes)    if hasattr(dataset, "ind_to_classes")    else 0
    num_rel_classes = len(dataset.ind_to_predicates) if hasattr(dataset, "ind_to_predicates") else 0
    with open_dict(cfg):
        cfg.model.roi_box_head.num_classes      = num_obj_classes
        cfg.model.roi_relation_head.num_classes = num_rel_classes

    dataset_name = get_dataset_name(cfg)
    logger.info(
        f"Dataset: {dataset_name}  "
        f"({len(dataset)} images, {num_obj_classes} obj classes, {num_rel_classes} rel classes)"
    )

    # ── Load model / session (once) ─────────────────────────────────────────
    model   = None
    session = None

    if use_onnx:
        onnx_path = _resolve_onnx_path(args, run_dir)
        logger.info(f"Loading ONNX from {onnx_path}…")
        session = load_onnx_session(onnx_path, args.provider, logger)
    else:
        checkpoint = resolve_checkpoint_from_dir(str(run_dir))
        logger.info(f"Loading PyTorch checkpoint from {checkpoint}…")
        model = load_pytorch_model(cfg, checkpoint, args.device)

    # ── Evaluation (full test set) ───────────────────────────────────────────
    eval_timing  = {}
    result_dict  = {}

    if not args.skip_eval:
        if use_onnx:
            predictions, eval_timing = eval_onnx(
                session, dataset, num_rel_classes, args.input_size, args.top_k, logger)
        else:
            predictions, eval_timing = eval_pytorch(model, dataset, args.device, logger)

        logger.info("Running SGG evaluation (R@K / mR@K / F1@K / zR@K + bbox mAP)…")
        result_dict = do_sgg_evaluation(
            cfg=cfg,
            dataset=dataset,
            dataset_name=dataset_name,
            predictions=predictions,
            output_folder=str(output_dir),
            logger=logger,
            iou_types=["relations", "bbox"],
        )

    # ── Latency benchmark ───────────────────────────────────────────────────
    latency_result = None

    if args.latency or args.skip_eval:
        n = args.num_images
        if use_onnx:
            latency_result = benchmark_latency_onnx(session, dataset, args.input_size, n)
        else:
            latency_result = benchmark_latency_pytorch(model, dataset, args.device, n)

        print_latency_table("ONNX" if use_onnx else "PyTorch", latency_result)

        lat_path = output_dir / "latency.json"
        with open(lat_path, "w") as f:
            json.dump(latency_result, f, indent=2)
        logger.info(f"Latency results saved to: {lat_path}")

    # ── Save summary JSON ────────────────────────────────────────────────────
    summary: dict = {
        "model_dir":   str(run_dir),
        "mode":        "onnx" if use_onnx else "pytorch",
        "eval_timing": eval_timing,
    }
    if latency_result:
        summary["latency"] = latency_result
    if isinstance(result_dict, dict):
        for k, v in result_dict.items():
            summary[k] = (
                {kk: _serialise(vv) for kk, vv in v.items()}
                if isinstance(v, dict) else _serialise(v)
            )

    summary_path = output_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
