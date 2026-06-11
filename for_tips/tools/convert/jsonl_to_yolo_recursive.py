#!/usr/bin/env python3
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dataset_path_resolver import read_class_list, load_records_from_tree, unique_output_stem

# =========================
# User settings
# =========================
DATASET_ROOT = Path("datasets/custom")
JSONL_PATTERN = "manual_labeled.jsonl"
OUTPUT_ROOT = Path("datasets/custom/ultralytics_yolo")

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

FIXED_CLASS_NAMES = read_class_list(DATASET_ROOT / "object_classes.txt")
MASK_VALUE_MODE = "object_id_number"
# MASK_VALUE_MODE = "mask_value_field"
MIN_AREA = 10
APPROX_EPSILON_RATIO = 0.002
KEEP_ONLY_LARGEST = True
COPY_IMAGES = True

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6


def build_class_map():
    return {name: idx for idx, name in enumerate(FIXED_CLASS_NAMES)}


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
    for sub in ["images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"]:
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
            os.symlink(src.resolve(), dst)
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
            raise ValueError(f"Mask at {mask_path} is multi-channel. This script expects a single-channel instance-id mask.")
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


def binary_mask_to_contours(binary_mask):
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
    return valid


def contour_to_yolo_line(contour, class_id, width, height):
    contour = contour.reshape(-1, 2).astype(np.float32)
    if len(contour) < 3:
        return None
    contour[:, 0] /= width
    contour[:, 1] /= height
    contour[:, 0] = np.clip(contour[:, 0], 0.0, 1.0)
    contour[:, 1] = np.clip(contour[:, 1], 0.0, 1.0)
    flat = contour.reshape(-1).tolist()
    if len(flat) < 6:
        return None
    return str(class_id) + " " + " ".join(f"{v:.6f}" for v in flat)


def make_label_lines(record, class_map):
    image_path = Path(record["image_path"])
    mask_path = Path(record["mask_path"])
    width, height = get_image_size(image_path)
    mask = load_mask(mask_path)
    if mask.shape[0] != height or mask.shape[1] != width:
        raise ValueError(f"Mask/image size mismatch for {image_path}: image=({width}, {height}), mask=({mask.shape[1]}, {mask.shape[0]})")

    lines = []
    missing_instances = []
    for i, obj in enumerate(record.get("objects", [])):
        cls_name = obj["class"].strip()
        if cls_name not in class_map:
            raise ValueError(f"Unknown class '{cls_name}' in scene {record.get('scene_id', 'unknown')}. Please add it to object_classes.txt.")
        class_id = class_map[cls_name]
        instance_value = get_instance_value(obj, i)
        binary = (mask == instance_value).astype(np.uint8)
        if binary.sum() == 0:
            missing_instances.append((obj["id"], cls_name, instance_value))
            continue
        contours = binary_mask_to_contours(binary)
        if not contours:
            missing_instances.append((obj["id"], cls_name, instance_value))
            continue
        for cnt in contours:
            line = contour_to_yolo_line(cnt, class_id, width, height)
            if line is not None:
                lines.append(line)
    return lines, missing_instances


def write_yaml(root):
    yaml_text = [
        f"path: {root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    for idx, name in enumerate(FIXED_CLASS_NAMES):
        yaml_text.append(f"  {idx}: {name}")
    with open(root / "data.yaml", "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_text) + "\n")


def process_split(records, split_name, class_map):
    image_dir = OUTPUT_ROOT / "images" / split_name
    label_dir = OUTPUT_ROOT / "labels" / split_name
    stats = {"images": 0, "objects_total": 0, "label_rows_written": 0, "objects_missing_mask": 0}

    for rec in records:
        src_img = Path(rec["image_path"])
        src_mask = Path(rec.get("mask_path", ""))
        if not src_img.exists():
            print(f"[WARN] Missing image: {src_img}")
            continue
        if not src_mask.exists():
            print(f"[WARN] Missing mask: {src_mask}")
            continue

        stem = unique_output_stem(rec)
        dst_img = image_dir / f"{stem}{src_img.suffix.lower()}"
        copy_or_link(src_img, dst_img)

        label_path = label_dir / f"{stem}.txt"
        lines, missing = make_label_lines(rec, class_map)
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

        stats["images"] += 1
        stats["objects_total"] += len(rec.get("objects", []))
        stats["label_rows_written"] += len(lines)
        stats["objects_missing_mask"] += len(missing)
        if missing:
            # print(f"[WARN] {dst_img.name}: missing/empty masks for {missing}")
            print(
                "[WARN] missing/empty masks\n"
                f"  split      : {split_name}\n"
                f"  scene_id   : {rec.get('scene_id', 'N/A')}\n"
                f"  image_path : {src_img}\n"
                f"  mask_path  : {src_mask}\n"
                f"  output_img : {dst_img}\n"
                f"  label_path : {label_path}\n"
                f"  missing    : {missing}"
            )
    return stats


def main():
    records, discovery = load_records_from_tree(DATASET_ROOT, JSONL_PATTERN)
    class_map = build_class_map()
    ensure_dirs(OUTPUT_ROOT)
    train_records, val_records, test_records = split_records(records, seed=SEED)
    train_stats = process_split(train_records, "train", class_map)
    val_stats = process_split(val_records, "val", class_map)
    test_stats = process_split(test_records, "test", class_map)
    write_yaml(OUTPUT_ROOT)

    with open(OUTPUT_ROOT / "classes.json", "w", encoding="utf-8") as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)

    summary = {
        "num_images_total": len(records),
        "num_classes": len(class_map),
        "class_map": class_map,
        "splits": {"train": train_stats, "val": val_stats, "test": test_stats},
        "mask_value_mode": MASK_VALUE_MODE,
        "min_area": MIN_AREA,
        "approx_epsilon_ratio": APPROX_EPSILON_RATIO,
        "keep_only_largest": KEEP_ONLY_LARGEST,
        "discovery": discovery,
    }
    with open(OUTPUT_ROOT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
