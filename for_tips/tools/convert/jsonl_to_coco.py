# tools/convert/jsonl_to_coco.py
import json, os, argparse

CLASSES = [
    "부품 박스","플라스틱 트레이","공정 부품","드라이버","작업자 손",
    "조립 지그","폐기 박스","렌치","케이블 묶음","보호 고글"
]
CAT2ID = {c:i+1 for i,c in enumerate(CLASSES)}

def convert(jsonl_path, out_path, img_w=640, img_h=480):
    images, anns, ann_id = [], [], 1
    with open(jsonl_path, encoding="utf-8") as f:
        for img_id, line in enumerate(f, start=1):
            data = json.loads(line)
            images.append({
                "id": img_id, "file_name": data["image_path"],
                "width": img_w, "height": img_h
            })
            for obj in data["objects"]:
                b = obj["bbox"]  # [x1,y1,x2,y2]
                bw, bh = b[2]-b[0], b[3]-b[1]
                anns.append({
                    "id": ann_id, "image_id": img_id,
                    "category_id": CAT2ID.get(obj["class"], 1),
                    "bbox": [b[0], b[1], bw, bh],  # COCO: x,y,w,h
                    "area": bw*bh, "iscrowd": 0
                })
                ann_id += 1
    cats = [{"id":v,"name":k} for k,v in CAT2ID.items()]
    coco = {"images":images,"annotations":anns,"categories":cats}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path,"w",encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
    print(f"✅ COCO JSON → {out_path} ({len(images)} imgs, {len(anns)} anns)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="data/jsonl/manual_labeled.jsonl")
    ap.add_argument("--out",   default="data/coco/train.json")
    args = ap.parse_args()
    convert(args.jsonl, args.out)
