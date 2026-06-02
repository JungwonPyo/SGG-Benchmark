#!/usr/bin/env python3
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def read_class_list(txt_path: str | Path) -> List[str]:
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"Class txt file not found: {txt_path}")

    items = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(line)

    if not items:
        raise ValueError(f"No valid class names found in: {txt_path}")
    return items


def load_jsonl(path: str | Path) -> List[dict]:
    path = Path(path)
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def find_jsonl_files(dataset_root: str | Path, pattern: str = "manual_labeled.jsonl") -> List[Path]:
    dataset_root = Path(dataset_root)
    return sorted(p for p in dataset_root.rglob(pattern) if p.is_file())


def _candidate_dirs_from_jsonl(jsonl_path: Path) -> Dict[str, List[Path]]:
    dataset_dir = jsonl_path.parent
    color_dir = dataset_dir.parent
    camera_dir = color_dir.parent
    segment_dir = camera_dir.parent
    bag_dir = segment_dir.parent

    image_candidates = [
        color_dir / "image_raw",
        color_dir / "images",
        dataset_dir / "image_raw",
        dataset_dir / "images",
        segment_dir / "camera" / "color" / "image_raw",
    ]
    mask_candidates = [
        color_dir / "masks",
        dataset_dir / "masks",
        segment_dir / "camera" / "color" / "masks",
    ]
    depth_candidates = [
        camera_dir / "depth" / "image_raw",
        camera_dir / "depth" / "image_viewer",
    ]
    return {
        "dataset_dir": [dataset_dir],
        "color_dir": [color_dir],
        "camera_dir": [camera_dir],
        "segment_dir": [segment_dir],
        "bag_dir": [bag_dir],
        "image_dirs": image_candidates,
        "mask_dirs": mask_candidates,
        "depth_dirs": depth_candidates,
    }


def _dedupe(seq: List[Path]) -> List[Path]:
    seen = set()
    out = []
    for p in seq:
        rp = str(p)
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def resolve_existing_path(raw_path: Optional[str], candidates: List[Path]) -> Optional[Path]:
    candidate_names = []
    if raw_path:
        raw = Path(raw_path)
        candidate_names.append(raw.name)
        if raw.suffix:
            candidate_names.append(raw.stem)

    candidate_names = [x for i, x in enumerate(candidate_names) if x and x not in candidate_names[:i]]

    for c in candidates:
        if c and c.exists():
            return c.resolve()

    return None


def resolve_image_and_mask(record: dict, jsonl_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    ctx = _candidate_dirs_from_jsonl(jsonl_path)

    raw_img = record.get("image_path")
    raw_mask = record.get("mask_path")

    image_name = Path(raw_img).name if raw_img else None
    mask_name = Path(raw_mask).name if raw_mask else None

    image_candidates = []
    mask_candidates = []

    if raw_img:
        image_candidates.append(Path(raw_img))
    if raw_mask:
        mask_candidates.append(Path(raw_mask))

    for d in ctx["image_dirs"]:
        if image_name:
            image_candidates.append(d / image_name)
    for d in ctx["mask_dirs"]:
        if mask_name:
            mask_candidates.append(d / mask_name)

    scene_id = str(record.get("scene_id", ""))
    stem = None
    if image_name:
        stem = Path(image_name).stem
    elif scene_id:
        stem = Path(scene_id).stem

    if stem:
        for d in ctx["image_dirs"]:
            image_candidates.append(d / f"{stem}.png")
            image_candidates.append(d / f"{stem}.jpg")
            image_candidates.append(d / f"{stem}.jpeg")
        for d in ctx["mask_dirs"]:
            mask_candidates.append(d / f"{stem}_mask.png")
            mask_candidates.append(d / f"{stem}.png")

    image_path = resolve_existing_path(raw_img, _dedupe(image_candidates))
    mask_path = resolve_existing_path(raw_mask, _dedupe(mask_candidates))
    return image_path, mask_path


def normalize_record(record: dict, jsonl_path: Path) -> Tuple[dict, Dict[str, str]]:
    image_path, mask_path = resolve_image_and_mask(record, jsonl_path)
    new_record = dict(record)
    meta = {
        "jsonl_path": str(jsonl_path.resolve()),
        "jsonl_name": jsonl_path.name,
        "bag_name": jsonl_path.parents[3].name if len(jsonl_path.parents) >= 4 else jsonl_path.parent.name,
        "segment_name": jsonl_path.parents[2].name if len(jsonl_path.parents) >= 3 else "segment_unknown",
    }
    if image_path is not None:
        new_record["image_path"] = str(image_path)
    if mask_path is not None:
        new_record["mask_path"] = str(mask_path)
    new_record["__meta__"] = meta
    return new_record, meta


def load_records_from_tree(dataset_root: str | Path, jsonl_pattern: str = "manual_labeled.jsonl") -> Tuple[List[dict], Dict[str, object]]:
    dataset_root = Path(dataset_root)
    jsonl_files = find_jsonl_files(dataset_root, jsonl_pattern)
    if not jsonl_files:
        raise FileNotFoundError(f"No jsonl files matching '{jsonl_pattern}' found under: {dataset_root}")

    records: List[dict] = []
    missing_images = []
    missing_masks = []
    jsonl_summaries = []

    for jp in jsonl_files:
        local = load_jsonl(jp)
        count_before = len(records)
        for rec in local:
            norm, meta = normalize_record(rec, jp)
            if not Path(norm.get("image_path", "__missing__")).exists():
                missing_images.append({
                    "jsonl": str(jp),
                    "scene_id": rec.get("scene_id"),
                    "image_path": rec.get("image_path"),
                })
                continue
            if rec.get("mask_path") and not Path(norm.get("mask_path", "__missing__")).exists():
                missing_masks.append({
                    "jsonl": str(jp),
                    "scene_id": rec.get("scene_id"),
                    "mask_path": rec.get("mask_path"),
                })
            records.append(norm)
        jsonl_summaries.append({
            "jsonl": str(jp.resolve()),
            "records_loaded": len(records) - count_before,
        })

    summary = {
        "dataset_root": str(dataset_root.resolve()),
        "jsonl_pattern": jsonl_pattern,
        "jsonl_files_found": len(jsonl_files),
        "jsonl_files": [str(p.resolve()) for p in jsonl_files],
        "records_loaded": len(records),
        "missing_images_count": len(missing_images),
        "missing_masks_count": len(missing_masks),
        "missing_images_preview": missing_images[:20],
        "missing_masks_preview": missing_masks[:20],
        "jsonl_summaries": jsonl_summaries,
    }
    return records, summary


def unique_output_stem(record: dict) -> str:
    img = Path(record["image_path"])
    meta = record.get("__meta__", {})
    bag = meta.get("bag_name", "bag")
    seg = meta.get("segment_name", "segment")
    scene_id = str(record.get("scene_id", img.stem))
    digest = hashlib.md5(str(img).encode("utf-8")).hexdigest()[:8]
    safe_scene = scene_id.replace("/", "_").replace(" ", "_")
    return f"{bag}__{seg}__{safe_scene}__{img.stem}__{digest}"
