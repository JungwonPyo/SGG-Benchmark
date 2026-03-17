import numpy as np
import time
import os
import cv2
import json
import argparse
import ctypes
import glob
import site


def _preload_cuda_libs():
    """Preload CUDA/cuDNN shared libs from the venv into the process global
    symbol table (RTLD_GLOBAL) so that onnxruntime's dlopen can find them.

    This is the most reliable approach: once a library is in the global table
    it is found by any subsequent dlopen, regardless of LD_LIBRARY_PATH.
    """
    site_dirs = list(site.getsitepackages())
    if hasattr(site, 'getusersitepackages'):
        try:
            site_dirs.append(site.getusersitepackages())
        except Exception:
            pass

    # Collect every nvidia sub-package lib dir
    lib_dirs = []
    for pkg in site_dirs:
        if not os.path.exists(pkg):
            continue
        nvidia = os.path.join(pkg, 'nvidia')
        if os.path.exists(nvidia):
            for sub in sorted(os.listdir(nvidia)):
                lib_dir = os.path.join(nvidia, sub, 'lib')
                if os.path.exists(lib_dir):
                    lib_dirs.append(lib_dir)
        # Also grab *_libs / *.libs wheels (e.g. nvidia_cublas_cu12-*.libs)
        for item in os.listdir(pkg):
            if item.endswith('_libs') or item.endswith('.libs'):
                lib_dirs.append(os.path.join(pkg, item))

    if not lib_dirs:
        return  # nothing found, give up silently

    # Update LD_LIBRARY_PATH so child processes also benefit
    current_ld = os.environ.get('LD_LIBRARY_PATH', '')
    new_ld = ':'.join(dict.fromkeys(lib_dirs))
    os.environ['LD_LIBRARY_PATH'] = f"{new_ld}:{current_ld}" if current_ld else new_ld

    # Load priority order: cublas → cudnn (cudnn depends on cublas)
    priority_patterns = [
        'libcublas.so.*',
        'libcublasLt.so.*',
        'libcudart.so.*',
        'libcudnn.so.*',
    ]

    # Build an ordered load list: priority libs first, then everything else
    ordered = []
    other = []
    for lib_dir in lib_dirs:
        for pat in priority_patterns:
            for path in sorted(glob.glob(os.path.join(lib_dir, pat))):
                if path not in ordered:
                    ordered.append(path)
        for path in sorted(glob.glob(os.path.join(lib_dir, 'lib*.so*'))):
            if path not in ordered and path not in other:
                other.append(path)

    loaded, failed = [], []
    for path in ordered + other:
        # Skip version-symlinks that point to already-loaded files
        real = os.path.realpath(path)
        if real in [os.path.realpath(p) for p in loaded]:
            continue
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            loaded.append(path)
        except OSError:
            failed.append(path)

    if loaded:
        print(f"[cuda preload] Loaded {len(loaded)} CUDA libs from venv.")
    if failed:
        print(f"[cuda preload] Could not load {len(failed)} libs (may be normal): "
              + ', '.join(os.path.basename(p) for p in failed[:5]))


import onnxruntime as ort  # noqa: E402 – must be after preload definition


class SGG_ONNX_Standalone:
    def __init__(self, onnx_path, provider='CUDAExecutionProvider', rel_conf=0.1, box_conf=0.5):
        self.rel_conf = rel_conf
        self.box_conf = box_conf

        # 1. Pre-load CUDA libs so onnxruntime can find libcudnn etc.
        _preload_cuda_libs()
        providers = [provider, 'CPUExecutionProvider']
        print(f"Loading ONNX model from {onnx_path} with {provider}...")
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        print(f"Active providers: {self.session.get_providers()}")
        
        self.input_name = self.session.get_inputs()[0].name
        
        # 2. Extract Metadata (Class Names)
        _meta = self.session.get_modelmeta().custom_metadata_map
        if 'obj_classes' in _meta and 'rel_classes' in _meta:
            obj_list = json.loads(_meta['obj_classes'])
            rel_list = json.loads(_meta['rel_classes'])
            # Model outputs 1-based labels for objects (0 is background, usually stripped)
            self.obj_classes = {i: v for i, v in enumerate(obj_list, start=1)}
            self.rel_classes = {i: v for i, v in enumerate(rel_list)}
            print(f"Loaded {len(obj_list)} object classes and {len(rel_list)} relation classes from metadata.")
        else:
            raise ValueError("ONNX model is missing 'obj_classes' or 'rel_classes' metadata. "
                             "This standalone demo requires embedded metadata.")

    def preprocess(self, image, size=640):
        """Letterbox + BGR to RGB + CHW + Norm"""
        h, w = image.shape[:2]
        r = min(size / h, size / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        
        # Padding
        top = (size - nh) // 2
        bottom = size - nh - top
        left = (size - nw) // 2
        right = size - nw - left
        
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # BGR to RGB, HWC to CHW, Normalize
        img = padded[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img).astype(np.float32) / 255.0
        return img[None, ...], r, (left, top)

    def predict(self, frame):
        # Preprocess
        img_input, ratio, (pad_x, pad_y) = self.preprocess(frame)
        
        # Inference
        outputs = self.session.run(None, {self.input_name: img_input})
        
        # Post-process: boxes (N, 6), rels (M, 5)
        # boxes_raw: x1, y1, x2, y2, label, score
        # rels_raw: subj_idx, obj_idx, label, triplet_score, rel_score
        boxes_raw, rels_raw = outputs[0], outputs[1]
        
        # Rescale boxes
        boxes_raw[:, [0, 2]] = (boxes_raw[:, [0, 2]] - pad_x) / ratio
        boxes_raw[:, [1, 3]] = (boxes_raw[:, [1, 3]] - pad_y) / ratio
        
        # Filter relations
        keep_rels = rels_raw[rels_raw[:, 3] >= self.rel_conf]
        
        # Identify boxes involved in relations
        rel_box_indices = set()
        if len(keep_rels) > 0:
            rel_box_indices = set(keep_rels[:, 0].astype(int)) | set(keep_rels[:, 1].astype(int))
            
        # Filter boxes (high confidence OR involved in a relation)
        keep_box_mask = (boxes_raw[:, 5] >= self.box_conf)
        for idx in rel_box_indices:
            keep_box_mask[idx] = True
            
        keep_box_indices = np.where(keep_box_mask)[0]
        
        # Remap relation indices because we are filtering boxes
        old_to_new = {old: new for new, old in enumerate(keep_box_indices)}
        final_boxes = boxes_raw[keep_box_indices]
        
        final_rels = []
        for rel in keep_rels:
            s, o = int(rel[0]), int(rel[1])
            if s in old_to_new and o in old_to_new:
                new_rel = rel.copy()
                new_rel[0], new_rel[1] = old_to_new[s], old_to_new[o]
                final_rels.append(new_rel)
        
        return final_boxes, np.array(final_rels)

    def _get_color(self, idx):
        """Deterministic per-class color via golden-ratio hue spacing."""
        import colorsys
        h = (idx * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.9)
        return (int(b * 255), int(g * 255), int(r * 255))

    def _draw_bbox(self, img, bbox, label, cls_id):
        """Fancy box with corner accents and a tight label background."""
        left, top, right, bottom = [int(b) for b in bbox]
        color = self._get_color(cls_id)

        # Thin full rectangle
        cv2.rectangle(img, (left, top), (right, bottom), color, 1)
        # Corner accents
        length = min(15, int((right - left) * 0.2), int((bottom - top) * 0.2))
        for (x, y), (dx, dy) in [
            ((left,  top),    (1,  1)),
            ((right, top),    (-1, 1)),
            ((left,  bottom), (1, -1)),
            ((right, bottom), (-1,-1)),
        ]:
            cv2.line(img, (x, y), (x + dx * length, y), color, 3)
            cv2.line(img, (x, y), (x, y + dy * length), color, 3)

        # Label background + text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        y_text = top - th - 5
        if y_text < 0:
            cv2.rectangle(img, (left, top), (left + tw + 4, top + th + 5), color, -1)
            cv2.putText(img, label, (left + 2, top + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (left, top - th - 5), (left + tw + 4, top), color, -1)
            cv2.putText(img, label, (left + 2, top - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        return (left + right) // 2, (top + bottom) // 2

    def visualize(self, img, boxes, rels):
        """Draw boxes and relations with the same style as demo_model.py."""
        if boxes is None or len(boxes) == 0:
            return img

        # --- Boxes ---
        centers = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2, cls_id, score = box
            label = f"{i}: {self.obj_classes.get(int(cls_id), '?')}"
            cx, cy = self._draw_bbox(img, (x1, y1, x2, y2), label, int(cls_id))
            centers.append((cx, cy))

        # --- Relations ---
        if rels is not None and len(rels) > 0 and rels.ndim == 2:
            for rel in rels:
                s_idx, o_idx, rel_id = int(rel[0]), int(rel[1]), int(rel[2])
                p1, p2 = centers[s_idx], centers[o_idx]

                dist = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
                if dist > 40:
                    # Shorten endpoints slightly so line starts/ends near box edges
                    alpha = 20.0 / (dist + 1e-6)
                    p1s = (int(p1[0] * (1-alpha) + p2[0] * alpha),
                           int(p1[1] * (1-alpha) + p2[1] * alpha))
                    p2s = (int(p2[0] * (1-alpha) + p1[0] * alpha),
                           int(p2[1] * (1-alpha) + p1[1] * alpha))

                    # Orange glow + white line
                    cv2.line(img, p1s, p2s, (255, 128, 0), 2, cv2.LINE_AA)
                    cv2.line(img, p1s, p2s, (255, 255, 255), 1, cv2.LINE_AA)

                    # Arrowhead pointing toward object
                    angle = np.arctan2(p1s[1] - p2s[1], p1s[0] - p2s[0])
                    for da in (0.5, -0.5):
                        tip = (int(p2s[0] + 8 * np.cos(angle + da)),
                               int(p2s[1] + 8 * np.sin(angle + da)))
                        cv2.line(img, p2s, tip, (255, 255, 255), 1, cv2.LINE_AA)

                    # Relation label at 1/3 from subject toward object
                    mid = (int(p1[0] * 0.65 + p2[0] * 0.35),
                           int(p1[1] * 0.65 + p2[1] * 0.35))
                    r_label = self.rel_classes.get(rel_id, "rel")
                    (tw, th), _ = cv2.getTextSize(r_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                    cv2.rectangle(img,
                                  (mid[0] - 2, mid[1] - th - 2),
                                  (mid[0] + tw + 2, mid[1] + 2),
                                  (20, 20, 20), -1)
                    cv2.putText(img, r_label, (mid[0], mid[1] - 1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--source', type=str, default='0', help='Camera index or image/video path')
    parser.add_argument('--provider', type=str, default='CUDAExecutionProvider',
                        choices=['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CPUExecutionProvider'],
                        help='ONNX Runtime execution provider')
    parser.add_argument('--rel_conf', type=float, default=0.05, help='Relation triplet-score threshold')
    parser.add_argument('--box_conf', type=float, default=0.4, help='Box confidence threshold')
    args = parser.parse_args()

    model = SGG_ONNX_Standalone(args.onnx, provider=args.provider, rel_conf=args.rel_conf, box_conf=args.box_conf)
    
    # Check if source is image or camera
    if os.path.isfile(args.source):
        frame = cv2.imread(args.source)
        boxes, rels = model.predict(frame)
        vis = model.visualize(frame.copy(), boxes, rels)
        cv2.imshow("Result", vis)
        cv2.waitKey(0)
    else:
        cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            t0 = time.time()
            boxes, rels = model.predict(frame)
            t1 = time.time()
            
            vis = model.visualize(frame, boxes, rels)
            cv2.putText(vis, f"FPS: {1/(t1-t0):.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("SGG ONNX Standalone", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
