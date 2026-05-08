# yolo_seg.py
import torch
from pathlib import Path
from omegaconf import DictConfig

from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import nms
from ultralytics.utils.plotting import feature_visualization

# Replace these imports with the exact segment-model classes used by
# your installed Ultralytics version.
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.nn.modules.head import Segment as _Segment


class YoloSegModel(SegmentationModel):
    def __init__(self, cfg: DictConfig, ch: int = 3, nc: int | None = None, verbose: bool = True):
        yolo_cfg = cfg.model.yolo.size + ".yaml"
        super().__init__(yolo_cfg, nc=nc, verbose=True)

        self.conf_thres = cfg.model.backbone.nms_thresh
        self.iou_thres = cfg.model.roi_heads.nms
        self.device = cfg.model.device
        self.input_size = cfg.input.img_size
        self.input_w = int(self.input_size[0])
        self.input_h = int(self.input_size[1])
        self.nc = nc
        self.max_det = cfg.model.roi_heads.detections_per_img

        if "12" in yolo_cfg:
            self.layers_to_extract = [14, 17, 20]
        elif "11" in yolo_cfg or "26" in yolo_cfg:
            self.layers_to_extract = [16, 19, 22]
        else:
            self.layers_to_extract = [15, 18, 21]

        freeze = cfg.model.backbone.freeze
        freeze_at = getattr(cfg.model.backbone, "freeze_at", -1) if not freeze else -1
        self._freeze_backbone(freeze, freeze_at)

    def _freeze_backbone(self, freeze: bool, freeze_at: int = -1):
        if freeze:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()
        elif freeze_at >= 0:
            for i, m in enumerate(self.model):
                for p in m.parameters():
                    p.requires_grad = (i >= freeze_at)
            self.eval()
        else:
            for p in self.parameters():
                p.requires_grad = True

    def forward_sgg(self, x, profile=False, visualize=False, embed=None):
        y, feature_maps = [], []
        det_out = None
        proto = None
        mc = None

        for i, m in enumerate(self.model):
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

            x = m(x)

            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=Path("./demo/test_custom/results"))

            if embed and i in self.layers_to_extract:
                feature_maps.append(x)

            y.append(x if m.i in self.save else None)

        if isinstance(x, (tuple, list)):
            if len(x) == 3:
                det_out, mc, proto = x
            elif len(x) == 2:
                det_out, proto = x
            else:
                det_out = x[0]
        else:
            det_out = x

        if embed:
            return {"det_out": det_out, "proto": proto, "mc": mc}, feature_maps
        return {"det_out": det_out, "proto": proto, "mc": mc}

    def load(self, weights_path: str, task=None):
        weights, _ = load_checkpoint(weights_path)
        if weights:
            super().load(weights)

    def postprocess(self, preds, image_sizes):
        det_preds = preds["det_out"]
        proto = preds.get("proto", None)

        det_preds, indices = nms.non_max_suppression(
            det_preds,
            nc=self.nc,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            max_det=self.max_det,
            return_idxs=True,
        )

        results = []
        for i, (pred, idx) in enumerate(zip(det_preds, indices)):
            orig_h, orig_w = float(image_sizes[i][0]), float(image_sizes[i][1])

            if pred.shape[0] == 0:
                results.append({
                    "boxes": torch.zeros((0, 4), device=self.device).float(),
                    "lb_boxes": torch.zeros((0, 4), device=self.device).float(),
                    "lb_input_size": self.input_h,
                    "lb_gain": 1.0,
                    "lb_pad_w": 0.0,
                    "lb_pad_h": 0.0,
                    "image_size": (int(orig_w), int(orig_h)),
                    "mode": "xyxy",
                    "pred_labels": torch.zeros((0,), device=self.device).long(),
                    "pred_scores": torch.zeros((0,), device=self.device).float(),
                    "labels": torch.zeros((0,), device=self.device).long(),
                    "feat_idx": torch.zeros((0,), device=self.device).long(),
                    # "pred_masks": torch.zeros((0, self.input_h, self.input_w), device=self.device).float(),
                })
                continue

            boxes = pred[:, :4]
            gain = min(self.input_h / orig_h, self.input_w / orig_w)
            pad_w = (self.input_w - orig_w * gain) / 2
            pad_h = (self.input_h - orig_h * gain) / 2
            offset_w = torch.round(torch.as_tensor(pad_w - 0.1, device=boxes.device))
            offset_h = torch.round(torch.as_tensor(pad_h - 0.1, device=boxes.device))

            lb_boxes = boxes.clamp(min=0)

            b0 = ((boxes[:, 0] - offset_w) / gain).clamp(0, orig_w)
            b1 = ((boxes[:, 1] - offset_h) / gain).clamp(0, orig_h)
            b2 = ((boxes[:, 2] - offset_w) / gain).clamp(0, orig_w)
            b3 = ((boxes[:, 3] - offset_h) / gain).clamp(0, orig_h)
            boxes = torch.stack([b0, b1, b2, b3], dim=1)

            scores = pred[:, 4]
            labels = pred[:, 5].long()
            labels_plus_1 = labels + 1

            results.append({
                "boxes": boxes,
                "lb_boxes": lb_boxes,
                "lb_input_size": self.input_h,
                "lb_gain": float(gain),
                "lb_pad_w": float(offset_w.item()),
                "lb_pad_h": float(offset_h.item()),
                "image_size": (int(orig_w), int(orig_h)),
                "mode": "xyxy",
                "pred_labels": labels_plus_1.detach().clone(),
                "pred_scores": scores,
                "labels": labels_plus_1,
                "feat_idx": idx.long(),
                # "pred_masks": None,   # fill later when you wire mask decoding
                # "proto": proto[i] if proto is not None else None,
                "proto": None,
            })

        return results