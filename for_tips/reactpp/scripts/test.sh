#!/bin/bash
# reactpp/scripts/test.sh
python tools/relation_test_net.py \
  --config-file reactpp/configs/robot_react.yaml \
  MODEL.WEIGHT output/robot_sgdet/model_final.pth \
  TEST.RELATION_MODE sgdet
