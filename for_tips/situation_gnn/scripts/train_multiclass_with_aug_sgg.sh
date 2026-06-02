#!/usr/bin/env bash
set -e

DATA="/home/dxr-labtop/SGG-Benchmark/datasets/custom"
IMG_DATA="/home/dxr-labtop/SGG-Benchmark/datasets/custom"
OUTDIR="./runs/situation_multiclass"

uv run python train_situation_gnn_recursive_e2e.py \
  --data "${DATA}" \
  --obj-classes ../../datasets/custom/object_classes.txt \
  --rel-classes ../../datasets/custom/relation_classes.txt \
  --outdir "${OUTDIR}" \
  --pred-source onnx \
  --sgg-onnx /home/dxr-labtop/SGG-Benchmark/checkpoints/CUSTOM/react_pp_yolo12m/model.onnx \
  --sgg-config /home/dxr-labtop/SGG-Benchmark/checkpoints/CUSTOM/react_pp_yolo12m/config.yml \
  --image-root "${IMG_DATA}" \
  --use-inverse-relations \
  --aug-enabled \
  --split-mode group \
  --group-mode segment \
  --mix-clean-train \
  --eval-on-pred-graphs