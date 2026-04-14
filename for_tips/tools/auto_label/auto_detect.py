# tools/auto_label/auto_detect.py
import cv2, json, numpy as np, os
from pathlib import Path
from groundingdino.util.inference import load_model, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import supervision as sv

CLASSES_EN = [
    "part box", "plastic tray", "process part", "screwdriver",
    "human hand", "assembly jig", "scrap box", "wrench", "cable bundle", "safety goggles"
]
CLASSES_KO = [
    "부품 박스", "플라스틱 트레이", "공정 부품", "드라이버",
    "작업자 손", "조립 지그", "폐기 박스", "렌치", "케이블 묶음", "보호 고글"
]
CLASS_MAP = dict(zip(CLASSES_EN, CLASSES_KO))

DINO_CONFIG  = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_WEIGHTS = "weights/groundingdino_swint_ogc.pth"
SAM2_CONFIG  = "sam2_hiera_l.yaml"
SAM2_WEIGHTS = "weights/sam2_hiera_large.pt"

def load_models():
    dino  = load_model(DINO_CONFIG, DINO_WEIGHTS)
    sam2  = build_sam2(SAM2_CONFIG, SAM2_WEIGHTS)
    predictor = SAM2ImagePredictor(sam2)
    return dino, predictor

def match_class(phrase):
    for en in CLASSES_EN:
        if en.lower() in phrase.lower():
            return CLASS_MAP[en]
    return None

def bbox_iou(b1, b2):
    b1, b2 = np.array(b1), np.array(b2)
    xi1, yi1 = max(b1[0],b2[0]), max(b1[1],b2[1])
    xi2, yi2 = min(b1[2],b2[2]), min(b1[3],b2[3])
    inter = max(0,xi2-xi1)*max(0,yi2-yi1)
    union = (b1[2]-b1[0])*(b1[3]-b1[1])+(b2[2]-b2[0])*(b2[3]-b2[1])-inter
    return inter/union if union>0 else 0

def auto_relations(objects):
    rels = []
    for i, o1 in enumerate(objects):
        for j, o2 in enumerate(objects):
            if i==j: continue
            b1, b2 = np.array(o1["bbox"]), np.array(o2["bbox"])
            cx1, cy1 = (b1[0]+b1[2])/2, (b1[1]+b1[3])/2
            cx2, cy2 = (b2[0]+b2[2])/2, (b2[1]+b2[3])/2
            dist = np.sqrt((cx1-cx2)**2+(cy1-cy2)**2)
            iou  = bbox_iou(b1, b2)

            if cy1 < cy2-30:
                rels.append({"subject":o1["id"],"predicate":"above","object":o2["id"]})
            elif iou > 0.4:
                rels.append({"subject":o1["id"],"predicate":"blocking","object":o2["id"]})
            elif b1[2] > b2[0] and b1[0] < b2[2] and abs(b1[3]-b2[1]) <= 5:
                rels.append({"subject":o1["id"],"predicate":"on","object":o2["id"]})
            elif iou > 0.05:
                rels.append({"subject":o1["id"],"predicate":"inside","object":o2["id"]})
            elif abs(b1[2]-b2[0]) <= 5 or abs(b2[2]-b1[0]) <= 5:
                rels.append({"subject":o1["id"],"predicate":"touching","object":o2["id"]})
            elif dist < 30:
                rels.append({"subject":o1["id"],"predicate":"next_to","object":o2["id"]})
            elif 30 <= dist < 80:
                rels.append({"subject":o1["id"],"predicate":"near","object":o2["id"]})
    return rels[:12]

def detect_objects(dino, sam_predictor, image_path):
    image = cv2.imread(image_path)
    h, w  = image.shape[:2]
    sam_predictor.set_image(image)

    caption = ". ".join(CLASSES_EN)
    boxes, logits, phrases = predict(
        model=dino, image=image,
        caption=caption, box_threshold=0.30, text_threshold=0.25
    )
    objects = []
    for idx, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
        ko = match_class(phrase)
        if ko is None: continue
        # box: cx,cy,w,h (0~1) → x1y1x2y2 pixel
        cx,cy,bw,bh = box.tolist()
        x1=int((cx-bw/2)*w); y1=int((cy-bh/2)*h)
        x2=int((cx+bw/2)*w); y2=int((cy+bh/2)*h)

        # SAM2 refine
        with torch.inference_mode():
            masks,_,_ = sam_predictor.predict(
                point_coords=None, point_labels=None,
                box=np.array([[x1,y1,x2,y2]]), multimask_output=False
            )
        if masks is not None and masks.sum() > 0:
            ys,xs = np.where(masks[0])
            x1,y1,x2,y2 = int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())

        objects.append({
            "id": f"O{idx+1}", "class": ko,
            "bbox": [x1,y1,x2,y2], "confidence": round(float(logit),3)
        })
    return objects

def process_folder(img_dir, out_jsonl, situation="S2"):
    import torch
    dino, sam_pred = load_models()
    scenes = []
    for fn in sorted(os.listdir(img_dir)):
        if not fn.lower().endswith((".jpg",".jpeg",".png")): continue
        path = os.path.join(img_dir, fn)
        sid  = f"{situation}_{Path(fn).stem}_auto"
        objects = detect_objects(dino, sam_pred, path)
        rels    = auto_relations(objects)
        scenes.append({
            "scene_id": sid, "situation": situation,
            "image_path": path, "objects": objects,
            "relationships": rels,
            "path_modification": "pending",
            "goal_position": [320, 240], "goal_changed": False
        })
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    with open(out_jsonl,"w",encoding="utf-8") as f:
        for s in scenes: f.write(json.dumps(s,ensure_ascii=False)+"\n")
    print(f"✅ {len(scenes)} scenes → {out_jsonl}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir",  default="data/raw_images")
    ap.add_argument("--out",      default="data/jsonl/auto_labeled.jsonl")
    ap.add_argument("--situation",default="S2")
    args = ap.parse_args()
    process_folder(args.img_dir, args.out, args.situation)
