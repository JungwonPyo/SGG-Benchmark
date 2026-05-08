import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# =========================
# User settings
# =========================
INPUT_JSONL = "datasets/custom/annotations.jsonl"
OUTPUT_ROOT = Path("datasets/custom/sgg_coco")

SEED = 42

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

FIXED_CLASS_NAMES = [
    "__background__",
    "부품 박스", "플라스틱 트레이", "공정 부품", "드라이버", "작업자 손",
    "조립 지그", "폐기 박스", "렌치", "케이블 묶음", "보호 고글", "엔드 이펙터"
]

CLASS_NAME_MAP = {
    "__background__": "__background__",
    "부품 박스": "parts box",
    "플라스틱 트레이": "plastic tray",
    "공정 부품": "workpiece",
    "드라이버": "screwdriver",
    "작업자 손": "hand",
    "조립 지그": "assembly jig",
    "폐기 박스": "waste box",
    "렌치": "wrench",
    "케이블 묶음": "cable bundle",
    "보호 고글": "safety goggles",
    "엔드 이펙터": "end effector"
}

FIXED_PREDICATE_NAMES = [
    # "__background__", 
    "on", "inside", "beside", "above", "touching", "near"
    ]

MASK_VALUE_MODE = "object_id_number"

# INCLUDE_SEGMENTATION = True
INCLUDE_SEGMENTATION = False
MIN_AREA = 10
APPROX_EPSILON_RATIO = 0.002
KEEP_ONLY_LARGEST = True

COPY_IMAGES = True

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_class_map():
    return {name: idx for idx, name in enumerate(FIXED_CLASS_NAMES)}


def build_predicate_map(records):
    if FIXED_PREDICATE_NAMES is not None:
        return {name: idx for idx, name in enumerate(FIXED_PREDICATE_NAMES)}

    pred_names = []
    seen = set()
    for rec in records:
        for rel in rec.get("relationships", []):
            pred = rel["predicate"].strip()
            if pred not in seen:
                seen.add(pred)
                pred_names.append(pred)
    return {name: idx for idx, name in enumerate(pred_names)}


def split_records(records, seed=42):
    records = list(records)
    random.Random(seed).shuffle(records)

    n = len(records)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train = records[:n_train]
    val = records[n_train:n_train + n_val]
    test = records[n_train + n_val:]
    return train, val, test


def ensure_dirs(root):
    for sub in ["train", "val", "test", "annotations"]:
        (root / sub).mkdir(parents=True, exist_ok=True)


def copy_or_link(src, dst):
    src = Path(src)
    dst = Path(dst)
    if dst.exists():
        return
    if COPY_IMAGES:
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src.resolve())
        except Exception:
            shutil.copy2(src, dst)


def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size


def load_mask(mask_path):
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        if np.all(mask[..., 0] == mask[..., 1]) and np.all(mask[..., 1] == mask[..., 2]):
            mask = mask[..., 0]
        else:
            raise ValueError(
                f"Mask at {mask_path} is multi-channel. "
                "Expected single-channel instance-id mask."
            )
    return mask


def get_instance_value(obj, obj_index_zero_based):
    if MASK_VALUE_MODE == "object_order":
        return obj_index_zero_based + 1

    if MASK_VALUE_MODE == "object_id_number":
        obj_id = obj["id"]
        digits = "".join(ch for ch in obj_id if ch.isdigit())
        if not digits:
            raise ValueError(f"Cannot parse numeric id from object id: {obj_id}")
        return int(digits)

    if MASK_VALUE_MODE == "mask_value_field":
        if "mask_value" not in obj:
            raise ValueError(f"Object {obj['id']} has no 'mask_value' field.")
        return int(obj["mask_value"])

    raise ValueError(f"Unknown MASK_VALUE_MODE: {MASK_VALUE_MODE}")


def binary_mask_to_polygons(binary_mask):
    binary_mask = (binary_mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        if APPROX_EPSILON_RATIO > 0:
            epsilon = APPROX_EPSILON_RATIO * cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, epsilon, True)

        if len(cnt) >= 3:
            valid.append(cnt)

    if KEEP_ONLY_LARGEST and valid:
        valid = [max(valid, key=cv2.contourArea)]

    polygons = []
    for cnt in valid:
        pts = cnt.reshape(-1, 2).astype(float).reshape(-1).tolist()
        if len(pts) >= 6:
            polygons.append(pts)
    return polygons


def mask_to_area(binary_mask):
    return int((binary_mask > 0).sum())


def bbox_xyxy_to_xywh(bbox):
    x1, y1, x2, y2 = bbox
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return [x1, y1, w, h]


def make_annotation_for_object(obj, obj_idx, image_id, ann_id, class_map, mask):
    # origin_id = obj["id"]
    cls_name = obj["class"].strip()
    if cls_name not in class_map:
        raise ValueError(f"Unknown class '{cls_name}'. Add it to FIXED_CLASS_NAMES.")

    category_id = class_map[cls_name]
    bbox_xywh = bbox_xyxy_to_xywh(obj["bbox"])

    if bbox_xywh[2] <= 0 or bbox_xywh[3] <= 0:
        return None

    ann = {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        # "origin_id": origin_id,
        "bbox": bbox_xywh,
        "area": int(bbox_xywh[2] * bbox_xywh[3]),
        "iscrowd": 0
    }

    if INCLUDE_SEGMENTATION and mask is not None:
        instance_value = get_instance_value(obj, obj_idx)
        binary = (mask == instance_value).astype(np.uint8)

        if binary.sum() > 0:
            polygons = binary_mask_to_polygons(binary)
            ann["segmentation"] = polygons if polygons else []
            ann["area"] = mask_to_area(binary)
        else:
            ann["segmentation"] = []

    return ann


def convert_split(records, split_name, class_map, predicate_map):
    images = []
    annotations = []
    rel_annotations = []

    image_id = 0
    ann_id = 0
    rel_id = 0

    missing_masks = []
    skipped_boxes = 0
    skipped_relations = 0

    split_dir = OUTPUT_ROOT / split_name

    for rec in records:
        src_img = Path(rec["image_path"])
        if not src_img.exists():
            print(f"[WARN] Missing image: {src_img}")
            continue

        # dst_img = split_dir / src_img.name
        # copy_or_link(src_img, dst_img)
        
        new_file_name = f"{image_id}{src_img.suffix}"
        dst_img = split_dir / new_file_name
        copy_or_link(src_img, dst_img)

        width, height = get_image_size(src_img)

        mask = None
        mask_path = rec.get("mask_path")
        if INCLUDE_SEGMENTATION and mask_path:
            mask_path = Path(mask_path)
            if mask_path.exists():
                mask = load_mask(mask_path)
                if mask.shape[0] != height or mask.shape[1] != width:
                    raise ValueError(
                        f"Mask/image size mismatch for {src_img}: "
                        f"image=({width}, {height}), mask=({mask.shape[1]}, {mask.shape[0]})"
                    )
            else:
                missing_masks.append(str(mask_path))

        images.append({
            "id": image_id,
            # "file_name": src_img.name,
            "file_name": new_file_name,
            "width": width,
            "height": height
        })

        obj_id_to_ann_id = {}

        for obj_idx, obj in enumerate(rec.get("objects", [])):
            ann = make_annotation_for_object(obj, obj_idx, image_id, ann_id, class_map, mask)
            if ann is None:
                skipped_boxes += 1
                continue

            annotations.append(ann)
            obj_id_to_ann_id[obj["id"]] = ann_id
            ann_id += 1

        for rel in rec.get("relationships", []):
            sub = rel["subject"]
            obj = rel["object"]
            pred = rel["predicate"].strip()

            if pred not in predicate_map:
                skipped_relations += 1
                continue

            if sub not in obj_id_to_ann_id or obj not in obj_id_to_ann_id:
                skipped_relations += 1
                continue

            rel_annotations.append({
                "id": rel_id,
                "subject_id": obj_id_to_ann_id[sub],
                "predicate_id": predicate_map[pred] + 1,
                "object_id": obj_id_to_ann_id[obj],
                "image_id": image_id
            })
            rel_id += 1

        image_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": idx, "name": CLASS_NAME_MAP[name], "supercategory": "none"}
            for name, idx in class_map.items()
        ],
        "rel_categories": [
            {"id": idx + 1, "name": name}
            for name, idx in predicate_map.items()
        ],
        "rel_annotations": rel_annotations
    }

    stats = {
        "images": len(images),
        "annotations": len(annotations),
        "relations": len(rel_annotations),
        "missing_mask_files": sorted(list(set(missing_masks))),
        "skipped_boxes": skipped_boxes,
        "skipped_relations": skipped_relations
    }

    return coco, stats


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    records = load_jsonl(INPUT_JSONL)

    class_map = build_class_map()
    predicate_map = build_predicate_map(records)

    ensure_dirs(OUTPUT_ROOT)

    train_records, val_records, test_records = split_records(records, seed=SEED)

    train_coco, train_stats = convert_split(train_records, "train", class_map, predicate_map)
    val_coco, val_stats = convert_split(val_records, "val", class_map, predicate_map)
    test_coco, test_stats = convert_split(test_records, "test", class_map, predicate_map)

    save_json(train_coco, OUTPUT_ROOT / "train" / "_annotations.coco.json")
    save_json(val_coco, OUTPUT_ROOT / "val" / "_annotations.coco.json")
    save_json(test_coco, OUTPUT_ROOT / "test" / "_annotations.coco.json")

    save_json(
        [{"id": idx, "name": name, "supercategory": "none"} for name, idx in class_map.items()],
        OUTPUT_ROOT / "annotations" / "object_classes.json"
    )
    save_json(
        [{"id": idx, "name": name} for name, idx in predicate_map.items()],
        OUTPUT_ROOT / "annotations" / "predicate_classes.json"
    )

    summary = {
        "num_images_total": len(records),
        "num_classes": len(class_map),
        "num_predicates": len(predicate_map),
        "class_map": class_map,
        "predicate_map": predicate_map,
        "splits": {
            "train": train_stats,
            "val": val_stats,
            "test": test_stats
        },
        "mask_value_mode": MASK_VALUE_MODE,
        "include_segmentation": INCLUDE_SEGMENTATION,
        "min_area": MIN_AREA,
        "approx_epsilon_ratio": APPROX_EPSILON_RATIO,
        "keep_only_largest": KEEP_ONLY_LARGEST
    }

    save_json(summary, OUTPUT_ROOT / "annotations" / "summary.json")

    print("Done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()