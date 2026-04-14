# tools/convert/jsonl_to_h5.py
import json, h5py, numpy as np, os, argparse

CLASSES = [
    "부품 박스","플라스틱 트레이","공정 부품","드라이버","작업자 손",
    "조립 지그","폐기 박스","렌치","케이블 묶음","보호 고글"
]
RELATIONS = ["on","inside","next_to","above","touching","blocking","near","beside"]
CLS2ID  = {c:i+1 for i,c in enumerate(CLASSES)}
REL2ID  = {r:i+1 for i,r in enumerate(RELATIONS)}

def convert(jsonl_path, out_h5, img_w=640, img_h=480):
    all_boxes, all_cls, all_rels = [], [], []
    img_to_first_box, img_to_last_box = [], []
    img_to_first_rel, img_to_last_rel = [], []
    box_ptr, rel_ptr = 0, 0

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            obj_list = data["objects"]
            rel_list = data.get("relationships", [])
            id2idx = {o["id"]:i for i,o in enumerate(obj_list)}

            img_to_first_box.append(box_ptr)
            for obj in obj_list:
                b = np.array(obj["bbox"], dtype=np.float32)
                # normalize 0~1
                b = b / np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
                all_boxes.append(b)
                all_cls.append(CLS2ID.get(obj["class"], 1))
                box_ptr += 1
            img_to_last_box.append(box_ptr)

            img_to_first_rel.append(rel_ptr)
            for rel in rel_list:
                si = id2idx.get(rel["subject"], -1)
                oi = id2idx.get(rel["object"], -1)
                ri = REL2ID.get(rel["predicate"], 1)
                if si >= 0 and oi >= 0:
                    all_rels.append([si, ri, oi])
                    rel_ptr += 1
            img_to_last_rel.append(rel_ptr)

    os.makedirs(os.path.dirname(out_h5), exist_ok=True)
    with h5py.File(out_h5, "w") as hf:
        hf.create_dataset("boxes",              data=np.array(all_boxes, dtype=np.float32))
        hf.create_dataset("gt_classes",         data=np.array(all_cls, dtype=np.int64))
        hf.create_dataset("gt_relations",       data=np.array(all_rels, dtype=np.int64))
        hf.create_dataset("img_to_first_box",   data=np.array(img_to_first_box, dtype=np.int64))
        hf.create_dataset("img_to_last_box",    data=np.array(img_to_last_box, dtype=np.int64))
        hf.create_dataset("img_to_first_rel",   data=np.array(img_to_first_rel, dtype=np.int64))
        hf.create_dataset("img_to_last_rel",    data=np.array(img_to_last_rel, dtype=np.int64))
    print(f"✅ H5 → {out_h5} | boxes:{len(all_boxes)}, rels:{len(all_rels)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="data/jsonl/manual_labeled.jsonl")
    ap.add_argument("--out",   default="data/h5/dataset.h5")
    args = ap.parse_args()
    convert(args.jsonl, args.out)
