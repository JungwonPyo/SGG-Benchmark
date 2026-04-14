#!/bin/bash
# reactpp/scripts/train.sh
# Usage: bash reactpp/scripts/train.sh [predcls|sgcls|sgdet]
MODE=${1:-predcls}

# Step1: PredCls (GT box+class → 관계만 학습)
if [ "$MODE" = "predcls" ]; then
  python -m torch.distributed.launch \
    --nproc_per_node=1 \
    tools/relation_train_net.py \
    --config-file reactpp/configs/robot_react.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    TEST.RELATION_MODE predcls \
    OUTPUT_DIR output/robot_predcls

# Step2: SGCls (GT box만 → class+관계 학습)
elif [ "$MODE" = "sgcls" ]; then
  python -m torch.distributed.launch \
    --nproc_per_node=1 \
    tools/relation_train_net.py \
    --config-file reactpp/configs/robot_react.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    TEST.RELATION_MODE sgcls \
    OUTPUT_DIR output/robot_sgcls

# Step3: SGDet (end-to-end)
elif [ "$MODE" = "sgdet" ]; then
  python -m torch.distributed.launch \
    --nproc_per_node=1 \
    tools/relation_train_net.py \
    --config-file reactpp/configs/robot_react.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    TEST.RELATION_MODE sgdet \
    OUTPUT_DIR output/robot_sgdet
fi
