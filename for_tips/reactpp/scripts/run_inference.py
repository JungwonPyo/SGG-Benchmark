# reactpp/scripts/run_inference.py
# SGG-Benchmark clone 후 경로 내에서 실행
import cv2, json, torch, numpy as np
from sgg_benchmark.config import cfg
from sgg_benchmark.data.datasets.evaluation import evaluate
from sgg_benchmark.modeling.detector import build_detection_model
from sgg_benchmark.utils.checkpoint import DetectronCheckpointer

def load_react(config_path, weight_path):
    cfg.merge_from_file(config_path)
    cfg.freeze()
    model = build_detection_model(cfg)
    model.eval()
    checkpointer = DetectronCheckpointer(cfg, model)
    checkpointer.load(weight_path)
    return model

def infer_scene(model, image_path):
    img = cv2.imread(image_path)
    img_tensor = torch.from_numpy(img).permute(2,0,1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        output = model(img_tensor)
    boxes     = output[0].bbox.cpu().numpy()
    labels    = output[0].get_field("labels").cpu().numpy()
    rel_pairs = output[0].get_field("rel_pair_idxs").cpu().numpy()
    rel_preds = output[0].get_field("pred_rel_scores").cpu().numpy()
    rel_labels= rel_preds.argmax(-1)
    return boxes, labels, rel_pairs, rel_labels

if __name__ == "__main__":
    model = load_react(
        "reactpp/configs/robot_react.yaml",
        "output/robot_sgdet/model_final.pth"
    )
    boxes, labels, pairs, rels = infer_scene(model, "data/raw_images/test.jpg")
    print("Boxes:", boxes[:5])
    print("Relations:", list(zip(pairs[:5,0], rels[:5], pairs[:5,1])))
