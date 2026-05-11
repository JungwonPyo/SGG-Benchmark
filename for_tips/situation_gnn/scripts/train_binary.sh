#!/usr/bin/env bash
set -e

DATA="/home/dxr-labtop/SGG-Benchmark/datasets/custom/annotations.jsonl"
OUTDIR="./runs/situation_binary"

uv run python train_situation_gnn.py \
  --data "${DATA}" \
  --outdir "${OUTDIR}" \
  --task binary \
  --epochs 100 \
  --batch-size 16 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --hidden-dim 128 \
  --num-layers 3 \
  --dropout 0.2 \
  --seed 42 \
  --obj-classes ../../datasets/custom/object_classes.txt \
  --rel-classes ../../datasets/custom/relation_classes.txt