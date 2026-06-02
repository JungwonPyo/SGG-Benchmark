# Robot Scene Graph Planner

## Set environment
```bash

chmod +x scripts/install_uv_custom.sh
./scripts/install_uv_custom.sh

```

## To train YOLO seperately
```bash

source .venv/bin/activate

# Prepare dataset
uv run python ./for_tips/tools/convert/jsonl_to_yolo.py 
## In case there is multiple jsonl
uv run python ./for_tips/tools/convert/jsonl_to_yolo_recursive.py 

uv run yolo detect train \
  data=datasets/custom/ultralytics_yolo/data.yaml \
  model=yolo12m.yaml \
  epochs=100 \
  imgsz=640 \
  batch=8 \
  device=0
uv run yolo detect val \
  model=runs/detect/train/weights/best.pt \
  data=datasets/custom/ultralytics_yolo/data.yaml \
  device=0
uv run yolo segment train \
  data=datasets/custom/ultralytics_yolo/data.yaml \
  model=yolo12m-seg.yaml \
  epochs=100 \
  imgsz=640 \
  batch=8 \
  device=0
uv run yolo segment val \
  model=runs/segment/train/weights/best.pt \
  data=datasets/custom/ultralytics_yolo/data.yaml \
  device=0
  
```

## To train relation network
```bash

source .venv/bin/activate

# Prepare dataset
uv run python ./for_tips/tools/convert/jsonl_to_format.py 
## In case there is multiple jsonl
uv run python ./for_tips/tools/convert/jsonl_to_format_recursive.py 

# Train
CUDA_VISIBLE_DEVICES=0 uv run python tools/relation_train_net_hydra.py \
  --config-name PSG/REACT++_custom \
  --task sgdet --save-best

# Evaluation
CUDA_VISIBLE_DEVICES=0 uv run python tools/relation_eval_hydra.py \
--checkpoint-dir checkpoints/CUSTOM/react_pp_yolo12m --task sgdet  --save-predictions --visualize

# Export to ONNX
CUDA_VISIBLE_DEVICES=0 uv run python tools/export_onnx.py \
--run-dir checkpoints/CUSTOM/react_pp_yolo12m

# Evaluation with ONNX
# Results are saved in: checkpoints/CUSTOM/react_pp_yolo12m/inference_onnx/onnx_eval_summary.json
CUDA_VISIBLE_DEVICES=0 uv run python tools/eval_onnx_psg.py \
--run-dir checkpoints/CUSTOM/react_pp_yolo12m --provider CUDAExecutionProvider


```

## To train situation network only
```bash

cd for_tips/situation_gnn
chmod +x ./scripts/*
# Train binary (situation or not)
./scripts/train_binary.sh
# Train only situation (classes only meaningful situations)
./scripts/train_meaningful_multiclass.sh
# Train all classes
./scripts/train_multiclass.sh
# Train all classes with augmentation
./scripts/train_multiclass_with_aug.sh

uv run python infer_situation_gnn.py \
  --checkpoint runs/situation_multiclass/best_model.pt \
  --input /home/dxr-labtop/SGG-Benchmark/datasets/custom/annotations.jsonl \
  --output runs/situation_multiclass/predictions.jsonl

# About many jsonl
uv run python infer_situation_gnn_recursive.py \
  --checkpoint runs/situation_multiclass/best_model.pt \
  --input /home/dxr-labtop/SGG-Benchmark/datasets/custom \
  --output runs/situation_multiclass/predictions.jsonl

# For total metrics
uv run python infer_situation_gnn_recursive_with_metrics.py \
  --checkpoint runs/situation_multiclass/best_model.pt \
  --input /home/dxr-labtop/SGG-Benchmark/datasets/custom \
  --output runs/situation_multiclass/predictions.jsonl

# With SGG
uv run python ./for_tips/situation_gnn/infer_end_to_end_sgg_gnn.py \
  --input /home/dxr-labtop/SGG-Benchmark/datasets/custom \
  --jsonl-pattern manual_labeled.jsonl \
  --sgg-config /home/dxr-labtop/SGG-Benchmark/checkpoints/CUSTOM/react_pp_yolo12m/config.yml \
  --sgg-weights /home/dxr-labtop/SGG-Benchmark/checkpoints/CUSTOM/react_pp_yolo12m/model.onnx \
  --gnn-checkpoint /home/dxr-labtop/SGG-Benchmark/for_tips/situation_gnn/runs/situation_multiclass/best_model.pt \
  --output /home/dxr-labtop/SGG-Benchmark/for_tips/situation_gnn/runs/situation_multiclass/e2e_predictions.jsonl \
  --save-vis-dir /home/dxr-labtop/SGG-Benchmark/for_tips/situation_gnn/runs/situation_multiclass/e2e_vis \
  --box-conf 0.5 \
  --rel-conf 0.1

```

## To train situation network with SGG
```bash

cd for_tips/situation_gnn
./scripts/train_multiclass_with_aug_sgg.sh

uv run python ./for_tips/situation_gnn/infer_end_to_end_sgg_gnn.py \
  --input /home/dxr-labtop/SGG-Benchmark/datasets/custom \
  --jsonl-pattern manual_labeled.jsonl \
  --sgg-config /home/dxr-labtop/SGG-Benchmark/checkpoints/CUSTOM/react_pp_yolo12m/config.yml \
  --sgg-weights /home/dxr-labtop/SGG-Benchmark/checkpoints/CUSTOM/react_pp_yolo12m/model.onnx \
  --gnn-checkpoint /home/dxr-labtop/SGG-Benchmark/for_tips/situation_gnn/runs/situation_multiclass/best_model.pt \
  --output /home/dxr-labtop/SGG-Benchmark/for_tips/situation_gnn/runs/situation_multiclass/e2e_predictions.jsonl \
  --save-vis-dir /home/dxr-labtop/SGG-Benchmark/for_tips/situation_gnn/runs/situation_multiclass/e2e_vis \
  --box-conf 0.5 \
  --rel-conf 0.1

```

## To run as inference
```bash

# Run Gemini335L
env -u LD_LIBRARY_PATH ROS_LOG_DIR="$HOME/.ros/log" LD_LIBRARY_PATH=/opt/ros/humble/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/gazebo-11/plugins ros2 launch orbbec_camera gemini_330_series.launch.py enable_colored_point_cloud:=true

uv run python for_tips/tools/ros2_scene_understanding_node.py \
  --sgg-config checkpoints/CUSTOM/react_pp_yolo12m/config.yml \
  --sgg-weights checkpoints/CUSTOM/react_pp_yolo12m/model.onnx \
  --gnn-checkpoint for_tips/situation_gnn/runs/situation_multiclass/best_model.pt \
  --rgb-topic /camera/color/image_raw \
  --depth-topic /camera/depth/image_raw \
  --camera-info-topic /camera/color/camera_info \
  --viz-topic /scene_graph/debug_image \
  --result-topic /scene_graph/result \
  --box-conf 0.5 \
  --rel-conf 0.1

# If numpy version error
uv init --name sgg-benchmark
uv add "numpy<2"

```

## To build custom msgs
```bash

cd tips_ws
colcon build --packages-select scene_understanding_msgs
source install/setup.bash # or source install/local_setup.bash
ros2 interface show scene_understanding_msgs/msg/SceneContext

```