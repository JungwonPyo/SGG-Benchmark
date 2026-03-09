# MODEL ZOO

All SGG models are trained without any debiasing or re-weighting methods (e.g. TDE or reweight loss); performance could likely be further improved with such techniques.

Download weights to a `checkpoints/` folder at the repository root, then evaluation with `tools/relation_eval_hydra.py`.

---

## Backbones

Before training an SGG model, you need a pre-trained YOLO backbone. I recommand to create a dedicated folder to not confuse backbone models with SGG models:
```bash
mkdir -p ./checkpoints/BACKBONES/PSG
```

Then you can either train your own YOLO model using the [official ultralytics](https://github.com/ultralytics/ultralytics) codebase or you can download our set of pre-trained YOLO backbone:

| Dataset | YOLO Backbone |
| --- | --- |
| PSG | [YOLO12n](https://huggingface.co/maelic/REACTPlusPlus_PSG/resolve/main/yolo12n/PSG_yolo12n_backbone.pt?download=true)
| PSG | [YOLO12s](https://huggingface.co/maelic/REACTPlusPlus_PSG/resolve/main/yolo12s/PSG_yolo12s_backbone.pt?download=true)
| PSG | [YOLO12m](https://huggingface.co/maelic/REACTPlusPlus_PSG/resolve/main/yolo12m/PSG_yolo12m_backbone.pt?download=true)
| PSG | [YOLO12l](https://huggingface.co/maelic/REACTPlusPlus_PSG/resolve/main/yolo12l/best_model.pth?download=true)
| PSG | [YOLOv8m](https://huggingface.co/maelic/REACTPlusPlus_PSG/resolve/main/yolov8m/PSG_yolov8m_backbone.pt?download=true)

You can also directly used one of our pre-trained model on the PSG dataset, please see below.

## SGG Models

### PSG

SGDet results on the **PSG test set** (2,177 images).

<!-- #### PyTorch Baseline

| Model | Weights | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 | mAP@50 | Latency (ms) |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| REACT (YOLOv8m) | [Download](https://drive.google.com/file/d/1uxohHdeh4eZ-FG81DS-ooJZFcWEHd3uX/view?usp=sharing) | 27.5 | 30.9 | 32.3 | 18.3 | 20.1 | 20.9 | 53.1 | 32.5 |
| **REACT++ (YOLOv12m)** 🚀 | [Download](https://huggingface.co/maelic/REACT-pp-PSG) | **31.11** | **36.29** | **39.44** | **22.73** | **25.75** | **27.55** | **52.60** | 26.3 | -->

#### ONNX Benchmark (SGDet, CUDA)

> **E2E Latency** = image load + letterbox pre-process + ONNX forward, averaged over all 2,177 PSG test images on CUDA. Evaluated with `tools/eval_onnx_psg.py`.

| Model | Backbone | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 | zR@20 | zR@50 | zR@100 | F1@20 | F1@50 | F1@100 | mAP@50 | E2E Lat. (ms) |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| REACT++ ONNX | [YOLOv12n](https://huggingface.co/maelic/REACTPlusPlus_PSG/tree/main/yolo12n) | 26.88 | 30.61 | 31.80 | 16.88 | 18.65 | 19.50 | 1.34 | 1.87 | 1.87 | 20.74 | 23.17 | 24.17 | 40.44 | **11.4** |
| REACT++ ONNX | [YOLOv12s](https://huggingface.co/maelic/REACTPlusPlus_PSG/tree/main/yolo12s) | 29.28 | 33.48 | 34.74 | 21.12 | 23.21 | 23.77 | 1.34 | 1.98 | 2.51 | 24.54 | 27.41 | 28.23 | 46.23 | **12.2** |
| REACT++ ONNX | [YOLOv8m](https://huggingface.co/maelic/REACTPlusPlus_PSG/tree/main/yolov8m)  | 30.69 | 35.68 | 37.43 | 22.75 | 25.46 | 26.40 | 1.65 | 2.58 | 2.96  | 26.13 | 29.72 | 30.96 | 52.49 | 15.3 |
| REACT++ ONNX | [**YOLOv12m**](https://huggingface.co/maelic/REACTPlusPlus_PSG/tree/main/yolo12m) | **32.69** | **37.20** | **38.58** | 22.74 | 25.21 | 26.08 | **2.77** | **4.36** | **4.82** | 26.82 | 30.05 | 31.12 | **55.39** | 15.7 |
| REACT++ ONNX | [YOLOv12l](https://huggingface.co/maelic/REACTPlusPlus_PSG/tree/main/yolo12l) | 30.99 | 35.30 | 36.68 | **23.20** | **25.49** | **26.45** | 1.90 | 2.97 | 2.97 | **26.53** | **29.60** | **30.74** | 50.55 | 19.6 |

> **R@K** = Recall@K (mean over images) · **mR@K** = mean Recall@K (macro-average over predicates) · **zR@K** = zero-shot Recall@K · **F1@K** = harmonic mean of R@K and mR@K · **mAP@50** = detection mAP at IoU = 0.50, Latency is computed on RTX A4000 GPU with batch size 1.

Once you have downloaded a model in .onnx you can run the inference as follow:
```bash
python demo/webcam_demo_onnx.py --onnx checkpoints/PSG/react++_yolo12l/model.onnx --config checkpoints/PSG/react++_yolo12l/config.yml
```

<!-- ---

### VG150

SGDet results on the **VG150 test set**.

| Model | Weights | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 | mAP@50 | Latency (ms) |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| REACT (YOLOv8m) | [Download](https://drive.google.com/file/d/1q7WAcJ9XS5ilt3Cf3ysBjwcz5CcaUzdJ/view?usp=sharing) | 21.04 | 26.16 | 28.75 | 9.78 | 12.26 | 13.63 | 31.8 | 23.9 |

---

### IndoorVG

SGDet results on the **IndoorVG test set**.

| Model | Weights | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 | mAP@50 | Latency (ms) |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **REACT++ (YOLOv12m)** 🚀 | [Download](https://drive.google.com/file/d/1XX2SbR1P_67B6y3cdrKn9ISCFFgImZk_/view?usp=sharing) | 20.17 | 25.48 | 28.43 | 14.91 | 18.67 | 20.53 | 35.10 | — | -->
