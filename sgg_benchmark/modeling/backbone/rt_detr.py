import torch
from torch import nn
import torch.nn.functional as F
from ultralytics.nn.tasks import DetectionModel
from sgg_benchmark.data.transforms import LetterBox

from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import ops
from ultralytics.engine.results import Results

import numpy as np
from PIL import Image


class RTDetr(DetectionModel):
    def __init__(self, cfg, ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        yolo_cfg = cfg.model.yolo.size+'.yaml'
        super().__init__(yolo_cfg, nc=nc, verbose=verbose)
        # self.features_layers = [len(self.model) - 2]
        self.conf_thres = cfg.model.backbone.nms_thresh
        self.iou_thres = cfg.model.roi_heads.nms
        self.device = cfg.model.device
        self.input_size = cfg.input.img_size  # (W, H)
        self.input_w = int(self.input_size[0])
        self.input_h = int(self.input_size[1])
        self.nc = nc

    def forward(self, x, profile=False, visualize=False, embed=None):
        y, feature_maps = [], []  # outputs
        for i, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            """
            We extract features from the following layers:
            15: 80x80
            18: 40x40
            21: 20x20
            For different object scales, as in original YOLOV8 implementation.
            """
            if embed:
                if i in {15, 18, 21}:  # if current layer is one of the feature extraction layers
                    feature_maps.append(x)
        if embed:
            return x, feature_maps
        else:
            return x

    def load(self, weights_path: str, task=None):
        """
        Initializes a new model and infers the task type from the model head.

        Args:
            weights (str): model checkpoint to be loaded
            task (str | None): model task
        """

        weights, _ = attempt_load_one_weight(weights_path)

        if weights:
            super().load(weights)

    def prepare_input(self, image, input_shape=(640,640), stride=32, auto=True):
        not_tensor = not isinstance(image, torch.Tensor)
        if not_tensor:
            same_shapes = all(x.shape == im[0].shape for x in image)
            letterbox = LetterBox(input_shape, auto=same_shapes, stride=self.model.stride)(image=image)
            im = np.stack([letterbox(image=x) for x in im])
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device).float()
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0

        return im
    
    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        if not(torch.is_tensor(batch)):
            batch['img'] = batch['img'].to(self.device, non_blocking=True)
            batch['img'] = (batch['img'].half() if self.half else batch['img'].float()) / 255
            for k in ['batch_idx', 'cls', 'bboxes']:
                batch[k] = batch[k].to(self.device)

            nb = len(batch['img'])
            self.lb = [torch.cat([batch['cls'], batch['bboxes']], dim=-1)[batch['batch_idx'] == i]
                    for i in range(nb)] if self.save_hybrid else []  # for autolabelling
        else:
            batch = batch.to(self.device, non_blocking=True)
            batch = (batch.half() if self.half else batch.float()) / 255
            nb = len(batch)

        return batch
    
    def visualize(self, preds, orig_imgs):
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
    
        # get model input size
        imgsz = (self.input_h, self.input_w)
        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(imgsz, pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
    
    def postprocess(self, preds, image_sizes):

        nd = preds[0].shape[-1]
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)

        results = []
        for i, bbox in enumerate(bboxes):  # (300, 4)
            out_img_size = image_sizes[i]
            # flip
            out_img_size = (out_img_size[1], out_img_size[0])

            bbox = ops.xywh2xyxy(bbox)
            score, cls = scores[i].max(-1, keepdim=True)  # (300, 1)
            idx = score.squeeze(-1) > self.conf_thres  # (300, )

            pred = torch.cat([bbox, score, cls], dim=-1)[idx]  # filter
            oh, ow = image_sizes[i]
            pred[..., [0, 2]] *= ow
            pred[..., [1, 3]] *= oh

            scores = pred[:, 4]
            labels = pred[:, 5].long()
            
            box_dict = {
                "boxes": pred[:, :4],
                "image_size": out_img_size,
                "mode": "xyxy",
                "pred_labels": labels.detach().clone(),
                "pred_scores": scores,
                "labels": labels + 1
            }

            assert len(box_dict["pred_labels"]) == len(box_dict["pred_scores"])

            results.append(box_dict)

        return results

    
    @staticmethod
    def _reset_ckpt_args(args):
        """Reset arguments when loading a PyTorch model."""
        include = {"imgsz", "data", "task", "single_cls"}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}