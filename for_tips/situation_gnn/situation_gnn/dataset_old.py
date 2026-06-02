from __future__ import annotations
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch_geometric.data import Data


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

    if len(items) == 0:
        raise ValueError(f"No valid class names found in: {txt_path}")
    return items


def load_records(path: str | Path) -> List[dict]:
    path = Path(path)
    if path.is_file():
        if path.suffix == ".jsonl":
            records = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records
        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return obj if isinstance(obj, list) else [obj]
        else:
            raise ValueError(f"Unsupported file type: {path}")
    elif path.is_dir():
        records = []
        for p in sorted(path.glob("*.json")):
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            records.extend(obj if isinstance(obj, list) else [obj])
        return records
    else:
        raise FileNotFoundError(path)


def _sort_situation_name(x: str):
    if x.startswith("S") and x[1:].isdigit():
        return (0, int(x[1:]))
    return (1, x)


def make_maps(
    records: List[dict],
    task: str = "multiclass",
    obj_classes: List[str] | None = None,
    rel_classes: List[str] | None = None,
) -> Dict[str, dict]:
    if obj_classes is None or rel_classes is None:
        raise ValueError("obj_classes and rel_classes must be provided from txt files.")

    situation_values = sorted(
        {r["situation"] for r in records},
        key=_sort_situation_name
    )

    if task == "binary":
        sit_list = ["S0", "meaningful"]
    elif task == "meaningful_multiclass":
        sit_list = sorted([s for s in situation_values if s != "S0"], key=_sort_situation_name)
    else:
        sit_list = situation_values

    obj_list = ["__UNK__"] + list(obj_classes)
    rel_list = ["__UNK__"] + list(rel_classes)

    maps = {
        "task": task,
        "obj_list": obj_list,
        "rel_list": rel_list,
        "sit_list": sit_list,
        "obj2idx": {k: i for i, k in enumerate(obj_list)},
        "rel2idx": {k: i for i, k in enumerate(rel_list)},
        "sit2idx": {k: i for i, k in enumerate(sit_list)},
    }
    return maps


def _get_image_size(record: dict) -> Tuple[int, int]:
    p = record.get("image_path", "")
    if p and Path(p).exists():
        try:
            with Image.open(p) as im:
                return im.width, im.height
        except Exception:
            pass
    return 1280, 720


def _bbox_norm(bbox, width, height):
    x1, y1, x2, y2 = bbox
    bw = max(float(x2 - x1), 1.0)
    bh = max(float(y2 - y1), 1.0)
    cx = (float(x1) + float(x2)) * 0.5
    cy = (float(y1) + float(y2)) * 0.5
    return [
        cx / width,
        cy / height,
        bw / width,
        bh / height,
        (bw * bh) / (width * height),
        bw / bh,
    ]


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _edge_geom(sub_box, obj_box, width, height, reverse_flag: float):
    sx1, sy1, sx2, sy2 = sub_box
    ox1, oy1, ox2, oy2 = obj_box

    scx = (sx1 + sx2) * 0.5 / width
    scy = (sy1 + sy2) * 0.5 / height
    ocx = (ox1 + ox2) * 0.5 / width
    ocy = (oy1 + oy2) * 0.5 / height

    sw = max(float(sx2 - sx1), 1.0) / width
    sh = max(float(sy2 - sy1), 1.0) / height
    ow = max(float(ox2 - ox1), 1.0) / width
    oh = max(float(oy2 - oy1), 1.0) / height

    dx = scx - ocx
    dy = scy - ocy
    dw = sw - ow
    dh = sh - oh
    dist = math.sqrt(dx * dx + dy * dy)
    iou = _iou(sub_box, obj_box)

    return [dx, dy, dw, dh, dist, iou, reverse_flag]


def _label_for_record(record: dict, maps, task: str):
    sit = record["situation"]
    if task == "binary":
        return maps["sit2idx"]["S0"] if sit == "S0" else maps["sit2idx"]["meaningful"]
    elif task == "meaningful_multiclass":
        return maps["sit2idx"][sit]
    return maps["sit2idx"][sit]


def record_to_data(
    record: dict,
    maps: Dict[str, dict],
    task: str = "multiclass",
    sample_idx: int = 0,
) -> Data:
    width, height = _get_image_size(record)
    objects = record.get("objects", [])
    relations = record.get("relationships", [])

    id_to_idx = {}
    x_cat = []
    x_num = []

    for i, obj in enumerate(objects):
        id_to_idx[obj["id"]] = i
        x_cat.append(maps["obj2idx"].get(obj["class"], 0))
        x_num.append(_bbox_norm(obj["bbox"], width, height))

    if len(x_cat) == 0:
        x_cat = [0]
        x_num = [[0, 0, 0, 0, 0, 0]]

    edge_src, edge_dst, edge_type, edge_num = [], [], [], []

    for rel in relations:
        sid = rel["subject"]
        oid = rel["object"]
        if sid not in id_to_idx or oid not in id_to_idx:
            continue

        s = id_to_idx[sid]
        o = id_to_idx[oid]
        rel_idx = maps["rel2idx"].get(rel["predicate"], 0)
        sb = objects[s]["bbox"]
        ob = objects[o]["bbox"]

        edge_src.append(s)
        edge_dst.append(o)
        edge_type.append(rel_idx)
        edge_num.append(_edge_geom(sb, ob, width, height, 0.0))

        edge_src.append(o)
        edge_dst.append(s)
        edge_type.append(rel_idx)
        edge_num.append(_edge_geom(ob, sb, width, height, 1.0))

    if len(edge_src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)
        edge_num = torch.zeros((0, 7), dtype=torch.float)
    else:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        edge_num = torch.tensor(edge_num, dtype=torch.float)

    y = torch.tensor([_label_for_record(record, maps, task)], dtype=torch.long)

    return Data(
        x_cat=torch.tensor(x_cat, dtype=torch.long),
        x_num=torch.tensor(x_num, dtype=torch.float),
        edge_index=edge_index,
        edge_type=edge_type,
        edge_num=edge_num,
        y=y,
        sample_idx=torch.tensor([sample_idx], dtype=torch.long),
        num_nodes=len(x_cat),
    )


def build_graphs(records: List[dict], maps: Dict[str, dict], task: str = "multiclass") -> List[Data]:
    return [record_to_data(r, maps, task=task, sample_idx=i) for i, r in enumerate(records)]


def class_weights_from_graphs(graphs: List[Data], num_classes: int) -> torch.Tensor:
    labels = [int(g.y.item()) for g in graphs]
    counts = Counter(labels)
    weights = torch.ones(num_classes, dtype=torch.float)
    total = sum(counts.values())
    for c in range(num_classes):
        weights[c] = total / max(1, counts.get(c, 0)) / num_classes
    return weights