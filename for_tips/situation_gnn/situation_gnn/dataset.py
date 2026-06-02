from __future__ import annotations

import copy
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
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


def _normalize_record_schema(record: dict) -> dict:
    r = copy.deepcopy(record)

    if "generated_graph" in r:
        g = r.get("generated_graph", {})
        r["objects"] = g.get("objects", [])
        r["relationships"] = g.get("relationships", [])
        if "gt_label" in r and r["gt_label"] is not None:
            r["situation"] = r["gt_label"]
        else:
            r["situation"] = r.get("situation", "S0")

    r.setdefault("objects", [])
    r.setdefault("relationships", [])
    r.setdefault("situation", "S0")
    r.setdefault("scene_id", "")
    r.setdefault("image_path", "")
    return r


def load_records(path: str | Path) -> List[dict]:
    path = Path(path)

    def _read_jsonl(p: Path) -> List[dict]:
        out = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(_normalize_record_schema(json.loads(line)))
        return out

    def _read_json(p: Path) -> List[dict]:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        rows = obj if isinstance(obj, list) else [obj]
        return [_normalize_record_schema(x) for x in rows]

    if path.is_file():
        if path.suffix == ".jsonl":
            return _read_jsonl(path)
        if path.suffix == ".json":
            return _read_json(path)
        raise ValueError(f"Unsupported file type: {path}")

    if path.is_dir():
        records = []
        for p in sorted(path.rglob("*.jsonl")):
            records.extend(_read_jsonl(p))
        for p in sorted(path.rglob("*.json")):
            records.extend(_read_json(p))
        return records

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
    use_inverse_relations: bool = True,
) -> Dict[str, Any]:
    if obj_classes is None or rel_classes is None:
        raise ValueError("obj_classes and rel_classes must be provided from txt files.")

    situation_values = sorted({r["situation"] for r in records}, key=_sort_situation_name)

    if task == "binary":
        sit_list = ["S0", "meaningful"]
    elif task == "meaningful_multiclass":
        sit_list = sorted([s for s in situation_values if s != "S0"], key=_sort_situation_name)
    else:
        sit_list = situation_values

    obj_list = ["__UNK__"] + list(obj_classes)
    rel_forward = ["__UNK__"] + list(rel_classes)
    if use_inverse_relations:
        rel_reverse = [f"{r}__REV" for r in rel_classes]
        rel_list = rel_forward + rel_reverse
    else:
        rel_list = rel_forward

    maps = {
        "task": task,
        "obj_list": obj_list,
        "rel_list": rel_list,
        "sit_list": sit_list,
        "obj2idx": {k: i for i, k in enumerate(obj_list)},
        "rel2idx": {k: i for i, k in enumerate(rel_list)},
        "sit2idx": {k: i for i, k in enumerate(sit_list)},
        "use_inverse_relations": use_inverse_relations,
        "num_plain_rel_classes": len(rel_classes),
        "num_forward_rel_entries": len(rel_forward),
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


def _safe_score(v: Any, default: float = 1.0) -> float:
    try:
        x = float(v)
    except Exception:
        x = default
    if math.isnan(x) or math.isinf(x):
        x = default
    return max(0.0, min(1.0, x))


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
    if task == "meaningful_multiclass":
        return maps["sit2idx"][sit]
    return maps["sit2idx"][sit]


def record_key(record: dict, fallback_idx: int = 0) -> str:
    if record.get("scene_id"):
        return str(record["scene_id"])
    if record.get("image_path"):
        return str(record["image_path"])
    return f"sample_{fallback_idx}"


def infer_group_id(record: dict, fallback_idx: int = 0, mode: str = "segment") -> str:
    bag = str(record.get("bag_name", "") or record.get("__bag_name__", "")).strip()
    seg = str(record.get("segment_name", "") or record.get("__segment_name__", "")).strip()
    src = str(record.get("source_jsonl", "")).strip()
    img = str(record.get("image_path", "")).strip()

    if mode == "frame":
        return record_key(record, fallback_idx)

    if mode == "bag" and bag:
        return bag

    if mode in ("segment", "bag_segment"):
        if bag and seg:
            return f"{bag}/{seg}"
        if seg:
            return seg

    if src:
        p = Path(src)
        parts = p.parts
        if "dataset" in parts:
            try:
                i = parts.index("dataset")
                if i >= 2:
                    if mode == "bag_segment":
                        return "/".join(parts[max(0, i - 2):i])
                    return parts[i - 1]
            except Exception:
                pass

    if img:
        p = Path(img)
        if len(p.parts) >= 3:
            return str(p.parent)

    return record_key(record, fallback_idx)


def _relation_to_idx(predicate: str, maps: Dict[str, Any], reverse: bool) -> int:
    base_idx = maps["rel2idx"].get(predicate, 0)
    if not reverse or not maps.get("use_inverse_relations", False):
        return base_idx
    if base_idx <= 0:
        return 0
    return base_idx + maps["num_plain_rel_classes"]


@dataclass
class GraphAugmentConfig:
    enabled: bool = False
    drop_node_prob: float = 0.0
    drop_edge_prob: float = 0.0
    mask_obj_prob: float = 0.0
    mask_rel_prob: float = 0.0
    bbox_jitter_frac: float = 0.0
    score_noise_std: float = 0.0
    add_fp_edge_prob: float = 0.0
    max_fp_edges: int = 2


def _jitter_bbox(bbox, width: int, height: int, frac: float):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)

    dx1 = random.gauss(0.0, frac * bw)
    dy1 = random.gauss(0.0, frac * bh)
    dx2 = random.gauss(0.0, frac * bw)
    dy2 = random.gauss(0.0, frac * bh)

    nx1 = max(0.0, min(width - 1.0, x1 + dx1))
    ny1 = max(0.0, min(height - 1.0, y1 + dy1))
    nx2 = max(0.0, min(width - 1.0, x2 + dx2))
    ny2 = max(0.0, min(height - 1.0, y2 + dy2))

    if nx2 <= nx1:
        nx2 = min(width - 1.0, nx1 + 1.0)
    if ny2 <= ny1:
        ny2 = min(height - 1.0, ny1 + 1.0)

    return [nx1, ny1, nx2, ny2]


def _apply_graph_augmentation(
    record: dict,
    width: int,
    height: int,
    augment_cfg: Optional[GraphAugmentConfig],
) -> dict:
    if augment_cfg is None or not augment_cfg.enabled:
        return copy.deepcopy(record)

    r = copy.deepcopy(record)
    objects = r.get("objects", [])
    relations = r.get("relationships", [])

    for obj in objects:
        obj["score"] = _safe_score(obj.get("score", 1.0))
        if augment_cfg.score_noise_std > 0:
            obj["score"] = _safe_score(obj["score"] + random.gauss(0.0, augment_cfg.score_noise_std))
        if augment_cfg.bbox_jitter_frac > 0 and "bbox" in obj:
            obj["bbox"] = _jitter_bbox(obj["bbox"], width, height, augment_cfg.bbox_jitter_frac)
        if augment_cfg.mask_obj_prob > 0 and random.random() < augment_cfg.mask_obj_prob:
            obj["class"] = "__UNK__"

    kept_objects = []
    for obj in objects:
        if len(objects) - len(kept_objects) <= 1:
            kept_objects.append(obj)
            continue
        if augment_cfg.drop_node_prob > 0 and random.random() < augment_cfg.drop_node_prob:
            continue
        kept_objects.append(obj)

    kept_ids = {o["id"] for o in kept_objects}
    filtered_relations = []
    for rel in relations:
        if rel.get("subject") not in kept_ids or rel.get("object") not in kept_ids:
            continue
        rel = copy.deepcopy(rel)
        rel["score"] = _safe_score(rel.get("score", 1.0))
        if augment_cfg.score_noise_std > 0:
            rel["score"] = _safe_score(rel["score"] + random.gauss(0.0, augment_cfg.score_noise_std))
        if augment_cfg.drop_edge_prob > 0 and random.random() < augment_cfg.drop_edge_prob:
            continue
        if augment_cfg.mask_rel_prob > 0 and random.random() < augment_cfg.mask_rel_prob:
            rel["predicate"] = "__UNK__"
        filtered_relations.append(rel)

    if augment_cfg.add_fp_edge_prob > 0 and len(kept_objects) >= 2:
        existing = {(rel["subject"], rel["object"], rel.get("predicate", "__UNK__")) for rel in filtered_relations}
        fp_budget = min(augment_cfg.max_fp_edges, len(kept_objects) * (len(kept_objects) - 1))
        for _ in range(fp_budget):
            if random.random() >= augment_cfg.add_fp_edge_prob:
                continue
            a, b = random.sample(kept_objects, 2)
            key = (a["id"], b["id"], "__UNK__")
            if key in existing:
                continue
            filtered_relations.append(
                {
                    "subject": a["id"],
                    "predicate": "__UNK__",
                    "object": b["id"],
                    "score": 0.05,
                }
            )
            existing.add(key)

    r["objects"] = kept_objects
    r["relationships"] = filtered_relations
    return r


def record_to_data(
    record: dict,
    maps: Dict[str, Any],
    task: str = "multiclass",
    sample_idx: int = 0,
    augment_cfg: Optional[GraphAugmentConfig] = None,
    training: bool = False,
) -> Data:
    width, height = _get_image_size(record)
    if training:
        record = _apply_graph_augmentation(record, width, height, augment_cfg)

    objects = record.get("objects", [])
    relations = record.get("relationships", [])

    id_to_idx = {}
    x_cat = []
    x_num = []

    for i, obj in enumerate(objects):
        id_to_idx[obj["id"]] = i
        obj_cls = obj.get("class", "__UNK__")
        obj_score = _safe_score(obj.get("score", 1.0))
        x_cat.append(maps["obj2idx"].get(obj_cls, 0))
        x_num.append(_bbox_norm(obj["bbox"], width, height) + [obj_score])

    if len(x_cat) == 0:
        x_cat = [0]
        x_num = [[0, 0, 0, 0, 0, 0, 0]]

    edge_src, edge_dst, edge_type, edge_num = [], [], [], []

    for rel in relations:
        sid = rel.get("subject")
        oid = rel.get("object")
        if sid not in id_to_idx or oid not in id_to_idx:
            continue

        s = id_to_idx[sid]
        o = id_to_idx[oid]
        pred = rel.get("predicate", "__UNK__")
        rel_score = _safe_score(rel.get("score", 1.0))
        sb = objects[s]["bbox"]
        ob = objects[o]["bbox"]

        edge_src.append(s)
        edge_dst.append(o)
        edge_type.append(_relation_to_idx(pred, maps, reverse=False))
        edge_num.append(_edge_geom(sb, ob, width, height, 0.0) + [rel_score])

        edge_src.append(o)
        edge_dst.append(s)
        edge_type.append(_relation_to_idx(pred, maps, reverse=True))
        edge_num.append(_edge_geom(ob, sb, width, height, 1.0) + [rel_score])

    if len(edge_src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)
        edge_num = torch.zeros((0, 8), dtype=torch.float)
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


def build_graphs(
    records: List[dict],
    maps: Dict[str, Any],
    task: str = "multiclass",
    augment_cfg: Optional[GraphAugmentConfig] = None,
    training: bool = False,
) -> List[Data]:
    return [
        record_to_data(
            r,
            maps,
            task=task,
            sample_idx=i,
            augment_cfg=augment_cfg,
            training=training,
        )
        for i, r in enumerate(records)
    ]


def class_weights_from_graphs(graphs: List[Data], num_classes: int) -> torch.Tensor:
    labels = [int(g.y.item()) for g in graphs]
    counts = Counter(labels)
    weights = torch.ones(num_classes, dtype=torch.float)
    total = sum(counts.values())
    for c in range(num_classes):
        weights[c] = total / max(1, counts.get(c, 0)) / num_classes
    return weights


class GraphRecordDataset(Dataset):
    def __init__(
        self,
        records: List[dict],
        maps: Dict[str, Any],
        task: str = "multiclass",
        training: bool = False,
        augment_cfg: Optional[GraphAugmentConfig] = None,
    ):
        self.records = records
        self.maps = maps
        self.task = task
        self.training = training
        self.augment_cfg = augment_cfg

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return record_to_data(
            self.records[idx],
            self.maps,
            task=self.task,
            sample_idx=idx,
            augment_cfg=self.augment_cfg,
            training=self.training,
        )