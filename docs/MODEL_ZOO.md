# MODEL ZOO

All SGG models are trained without any debiasing or re-weighting methods (e.g. TDE or reweight loss); performance could likely be further improved with such techniques.

Download weights to a `checkpoints/` folder at the repository root, then evaluation with `tools/relation_eval_hydra.py` or `tools/evaluate.py`.

---

## SGG Models

### PSG

SGDet results on the **PSG test set** (2,177 images).

#### Benchmark (SGDet, CUDA)

> **E2E Latency** = image load + letterbox pre-process + model forward, averaged over 200 test images on CUDA (RTX A4000 GPU, batch size 1). Both PyTorch and ONNX models are evaluated over all 2,177 PSG test images with `tools/evaluate.py --compare`.

| Model | Backbone | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 | F1@K | mAP@50 | E2E Lat. (ms) |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| REACT++ PyTorch | [YOLOv12n](https://huggingface.co/maelic/REACTPlusPlus_PSG/tree/main/yolo12n) | 27.04 | 31.96 | 35.12 | 17.65 | 20.95 | 22.74 | 24.76 | 42.95 | 15.0 |
| REACT++ ONNX | [YOLOv12n](https://huggingface.co/maelic/REACTPlusPlus_PSG/tree/main/yolo12n) | 27.29 | 32.26 | 35.43 | 17.33 | 20.38 | 22.19 | 24.49 | 42.94 | **11.8** |
| REACT++ PyTorch | [YOLOv12s](https://huggingface.co/maelic/REACTPlusPlus_PSG/tree/main/yolo12s) | 29.78 | 35.19 | 38.22 | 22.73 | 25.35 | 27.05 | 28.98 | 48.69 | 14.8 |
| REACT++ ONNX | [YOLOv12s](https://huggingface.co/maelic/REACTPlusPlus_PSG/tree/main/yolo12s) | 29.71 | 35.24 | 38.29 | 21.86 | 24.68 | 26.38 | 28.49 | 48.67 | **12.8** |
| REACT++ PyTorch | [YOLOv8m](https://huggingface.co/maelic/REACTPlusPlus_PSG/tree/main/yolov8m) | 30.96 | 36.76 | 39.97 | 22.97 | 26.10 | 27.73 | 29.88 | 54.26 | 17.2 |
| REACT++ ONNX | [YOLOv8m](https://huggingface.co/maelic/REACTPlusPlus_PSG/tree/main/yolov8m) | 30.83 | 36.65 | 39.85 | 22.13 | 25.37 | 27.00 | 29.31 | 54.26 | 16.0 |
| REACT++ PyTorch | [**YOLOv12m**](https://huggingface.co/maelic/REACTPlusPlus_PSG/tree/main/yolo12m) | **32.68** | **38.36** | **41.39** | **23.92** | **27.19** | **28.94** | **31.17** | **58.03** | 18.4 |
| REACT++ ONNX | [**YOLOv12m**](https://huggingface.co/maelic/REACTPlusPlus_PSG/tree/main/yolo12m) | **32.80** | **38.51** | **41.49** | 23.60 | 26.99 | 28.71 | 31.04 | **58.05** | 16.3 |
| REACT++ PyTorch | [YOLOv12l](https://huggingface.co/maelic/REACTPlusPlus_PSG/tree/main/yolo12l) | 31.37 | 37.07 | 40.18 | 24.06 | 27.11 | 28.94 | 30.73 | 52.95 | 23.9 |
| REACT++ ONNX | [YOLOv12l](https://huggingface.co/maelic/REACTPlusPlus_PSG/tree/main/yolo12l) | 31.39 | 37.07 | 40.21 | 23.31 | 26.36 | 28.13 | 30.22 | 52.95 | 20.2 |

> **R@K** = Recall@K (mean over images) · **mR@K** = mean Recall@K (macro-average over predicates) · **F1@K** = average of F1@20, F1@50 and F1@100 (harmonic mean of R@K and mR@K at each K) · **mAP@50** = detection mAP at IoU = 0.50 · Latency measured on RTX A4000, batch size 1.

---

### VG150

SGDet results on the **VG150 test set**.

#### Benchmark (SGDet, CUDA)

| Model | Backbone | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 | F1@K | mAP@50 | E2E Lat. (ms) |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| REACT++ PyTorch | [**YOLOv8m**](https://huggingface.co/maelic/REACTPlusPlus_VG150/resolve/main/yolov8m/best_model.pth?download=true) | **22.89** | **29.96** | **34.09** | **12.22** | **16.31** | **18.45** | **20.33** | **36.82** | 18.8 |
| REACT++ ONNX | [**YOLOv8m**](https://huggingface.co/maelic/REACTPlusPlus_VG150/resolve/main/yolov8m/model.onnx?download=true) | **22.90** | **29.97** | **34.10** | **12.20** | **16.33** | **18.44** | **20.33** | **36.80** | 17.1 |
| REACT++ PyTorch | [YOLOv26m](https://huggingface.co/maelic/REACTPlusPlus_VG150/resolve/main/yolo26m/best_model.pth?download=true) | 21.12 | 28.34 | 33.70 | 10.81 | 14.60 | 18.36 | 19.12 | 32.70 | 19.7 |
| REACT++ ONNX | [YOLOv26m](https://huggingface.co/maelic/REACTPlusPlus_VG150/resolve/main/yolo26m/model.onnx?download=true) | 21.12 | 28.33 | 33.69 | 10.94 | 14.75 | 18.45 | 19.22 | 32.70 | 17.0 |
| REACT++ PyTorch | [YOLOv12m](https://huggingface.co/maelic/REACTPlusPlus_VG150/resolve/main/yolo12m/best_model.pth?download=true) | 18.76 | 24.63 | 28.47 | 10.81 | 14.42 | 16.78 | 17.67 | 32.52 | 20.7 |
| REACT++ ONNX | [YOLOv12m](https://huggingface.co/maelic/REACTPlusPlus_VG150/resolve/main/yolo12m/model.onnx?download=true) | 18.71 | 24.59 | 28.42 | 10.75 | 14.38 | 16.71 | 17.61 | 32.51 | 19.0 |

### IndoorVG

SGDet results on the **IndoorVG test set**.

#### Benchmark (SGDet, CUDA)

| Model | Backbone | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 | F1@K | mAP@50 | E2E Lat. (ms) |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| REACT++ PyTorch | [**YOLOv8m**](https://huggingface.co/maelic/REACTPlusPlus_IndoorVG/resolve/main/yolov8m/best_model.pth?download=true) | **23.1** | **30.27** | **35.1** | **17.54** | **22.3** | **24.97** | **24.93** | **44.8** | 19.1 |
| REACT++ ONNX | [**YOLOv8m**](https://huggingface.co/maelic/REACTPlusPlus_IndoorVG/resolve/main/yolov8m/model.onnx?download=true) | **23.05** | **30.19** | **35.03** | **17.70** | **22.37** | **25.08** | **24.97** | **44.8** | 14.7 |

#### PyTorch vs ONNX evaluation

Both backends are evaluated identically — same PSG test images, same letterbox pre-processing, same NMS settings (`conf_thres=0.001`, `max_det=100`) — so the numbers are directly comparable.

The ONNX export bakes the full detection pipeline (backbone NMS, relation scoring) into a single graph, removing Python overhead and enabling graph-level optimisations by ONNX Runtime with `CUDAExecutionProvider`. This typically yields a **2–7 ms lower E2E latency** with no meaningful accuracy loss: R@K and mAP@50 are within ±0.2 pp of their PyTorch counterparts. The small residual gap comes from minor numerical differences between PyTorch and ONNX Runtime kernels (e.g. attention, softmax), not from a systematic pipeline difference.

If you see a larger discrepancy when re-exporting a model, ensure `export_obj_thres` is set to `0.0` in `tools/export_onnx.py` (so no second confidence filter is applied after NMS), and that `backbone.conf_thres` and `backbone.max_det` are left at their config values and not overridden before tracing.

Once you have downloaded a model in .onnx you can run the inference as follow:
```bash
python demo/webcam_demo_onnx.py --onnx checkpoints/PSG/react++_yolo12l/model.onnx
```

## Backbones

Before training an SGG model, you need a pre-trained YOLO backbone. I recommand to create a dedicated folder to not confuse backbone models with SGG models:
```bash
mkdir -p ./checkpoints/BACKBONES/PSG
```

Then you can either train your own YOLO model using the [official ultralytics](https://github.com/ultralytics/ultralytics) codebase or you can download our set of pre-trained YOLO backbone:

| Dataset | YOLO Backbone | mAP@50 |
| --- | --- | --- |
| PSG | [YOLO12n](https://huggingface.co/maelic/REACTPlusPlus_PSG/resolve/main/yolo12n/PSG_yolo12n_backbone.pt?download=true) | 42.95
| PSG | [YOLO12s](https://huggingface.co/maelic/REACTPlusPlus_PSG/resolve/main/yolo12s/PSG_yolo12s_backbone.pt?download=true) | 48.69
| PSG | [YOLO12m](https://huggingface.co/maelic/REACTPlusPlus_PSG/resolve/main/yolo12m/PSG_yolo12m_backbone.pt?download=true) | 58.03
| PSG | [YOLO12l](https://huggingface.co/maelic/REACTPlusPlus_PSG/resolve/main/yolo12l/best_model.pth?download=true) | 52.95
| PSG | [YOLOv8m](https://huggingface.co/maelic/REACTPlusPlus_PSG/resolve/main/yolov8m/PSG_yolov8m_backbone.pt?download=true) | 54.26
| VG150 | [YOLOv8m](https://huggingface.co/maelic/REACTPlusPlus_VG150/resolve/main/yolov8m/VG150_yolov8m_backbone.pt?download=true) | 36.82
| VG150 | [YOLO12m](https://huggingface.co/maelic/REACTPlusPlus_VG150/resolve/main/yolo12m/VG150_yolo12m_backbone.pt?download=true) | 32.52
| VG150 | [YOLO26m](https://huggingface.co/maelic/REACTPlusPlus_VG150/resolve/main/yolo26m/VG150_yolo26m_backbone.pt?download=true) | 32.70
| IndoorVG | [YOLOv8m](https://huggingface.co/maelic/REACTPlusPlus_IndoorVG/resolve/main/yolov8m/IndoorVG_yolov8m_backbone.pt?download=true) | 44.80
