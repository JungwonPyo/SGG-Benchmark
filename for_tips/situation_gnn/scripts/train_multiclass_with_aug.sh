#!/usr/bin/env bash
set -e

# DATA="/home/dxr-labtop/SGG-Benchmark/datasets/custom/annotations.jsonl"
DATA="/home/dxr-labtop/SGG-Benchmark/datasets/custom"
OUTDIR="./runs/situation_multiclass"

uv run python train_situation_gnn_recursive_e2e.py \
  --data "${DATA}" \
  --obj-classes ../../datasets/custom/object_classes.txt \
  --rel-classes ../../datasets/custom/relation_classes.txt \
  --outdir "${OUTDIR}" \
  --use-inverse-relations \
  --aug-enabled \
  --split-mode group \
  --group-mode segment