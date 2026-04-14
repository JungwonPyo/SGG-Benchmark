#!/bin/bash
# run_all.sh – 전체 파이프라인 한번에 실행
set -e
echo "=== Step 1: Auto Labeling ==="
python tools/auto_label/auto_detect.py --img_dir data/raw_images --situation S2

echo "=== Step 3: Convert ==="
python tools/convert/jsonl_to_h5.py   --jsonl data/jsonl/auto_labeled.jsonl
python tools/convert/jsonl_to_coco.py --jsonl data/jsonl/auto_labeled.jsonl

echo "=== Step 5: GCN Train ==="
python graph_net/train.py --jsonl data/jsonl/auto_labeled.jsonl --epochs 30

echo "=== Step 6: Realtime Inference ==="
python graph_net/infer_realtime.py

echo "DONE"
