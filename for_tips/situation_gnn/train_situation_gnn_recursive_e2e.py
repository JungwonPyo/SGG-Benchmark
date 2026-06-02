from __future__ import annotations

import sys
import cv2
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from situation_gnn.dataset import (
    GraphAugmentConfig,
    GraphRecordDataset,
    build_graphs,
    class_weights_from_graphs,
    infer_group_id,
    load_records,
    make_maps,
    read_class_list,
    record_key,
)
from situation_gnn.model import SceneSituationGNN

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SGG_ROOT = ROOT.parent
if str(SGG_ROOT) not in sys.path:
    sys.path.insert(0, str(SGG_ROOT))

from demo.onnx_model import SGG_ONNX_Model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices_frame(labels, seed=42, test_ratio=0.15, val_ratio=0.15):
    idx = np.arange(len(labels))
    counts = Counter(labels)
    stratify = labels if len(counts) > 1 and min(counts.values()) >= 2 else None

    train_idx, temp_idx = train_test_split(
        idx,
        test_size=test_ratio + val_ratio,
        random_state=seed,
        stratify=stratify,
    )

    temp_labels = [labels[i] for i in temp_idx]
    counts2 = Counter(temp_labels)
    stratify2 = temp_labels if len(counts2) > 1 and min(counts2.values()) >= 2 else None

    rel_test = test_ratio / (test_ratio + val_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=rel_test,
        random_state=seed,
        stratify=stratify2,
    )
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def split_indices_grouped(records, labels, seed=42, test_ratio=0.15, val_ratio=0.15, group_mode="segment"):
    group_to_indices = defaultdict(list)
    for i, r in enumerate(records):
        gid = infer_group_id(r, fallback_idx=i, mode=group_mode)
        group_to_indices[gid].append(i)

    groups = sorted(group_to_indices.keys())
    if len(groups) < 3:
        return split_indices_frame(labels, seed=seed, test_ratio=test_ratio, val_ratio=val_ratio)

    group_labels = []
    for g in groups:
        ys = [labels[i] for i in group_to_indices[g]]
        group_labels.append(Counter(ys).most_common(1)[0][0])

    counts = Counter(group_labels)
    stratify = group_labels if len(counts) > 1 and min(counts.values()) >= 2 else None

    train_groups, temp_groups = train_test_split(
        groups,
        test_size=test_ratio + val_ratio,
        random_state=seed,
        stratify=stratify,
    )

    temp_group_labels = [group_labels[groups.index(g)] for g in temp_groups]
    counts2 = Counter(temp_group_labels)
    stratify2 = temp_group_labels if len(counts2) > 1 and min(counts2.values()) >= 2 else None

    rel_test = test_ratio / (test_ratio + val_ratio)
    val_groups, test_groups = train_test_split(
        temp_groups,
        test_size=rel_test,
        random_state=seed,
        stratify=stratify2,
    )

    train_idx = [i for g in train_groups for i in group_to_indices[g]]
    val_idx = [i for g in val_groups for i in group_to_indices[g]]
    test_idx = [i for g in test_groups for i in group_to_indices[g]]
    return train_idx, val_idx, test_idx


def build_pred_record_map(pred_records: List[dict]) -> Dict[str, dict]:
    out = {}
    for i, r in enumerate(pred_records):
        out[record_key(r, i)] = r
    return out


def match_records_by_key(base_records: List[dict], pred_map: Dict[str, dict], fallback_to_base: bool = True) -> Tuple[List[dict], int]:
    matched = []
    hit = 0
    for i, r in enumerate(base_records):
        k = record_key(r, i)
        if k in pred_map:
            matched.append(pred_map[k])
            hit += 1
        elif fallback_to_base:
            matched.append(r)
    return matched, hit


def evaluate(model, loader, device, criterion):
    model.eval()
    all_y, all_pred = [], []
    total_loss = 0.0
    n_graphs = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = criterion(logits, batch.y.view(-1))
            pred = logits.argmax(dim=-1)

            total_loss += loss.item() * batch.num_graphs
            n_graphs += batch.num_graphs
            all_y.extend(batch.y.view(-1).cpu().tolist())
            all_pred.extend(pred.cpu().tolist())

    acc = accuracy_score(all_y, all_pred) if all_y else 0.0
    macro_f1 = f1_score(all_y, all_pred, average="macro", zero_division=0) if all_y else 0.0
    return total_loss / max(1, n_graphs), acc, macro_f1, all_y, all_pred

def lookup_label(vocab, idx: int, prefix: str) -> str:
    if vocab is None:
        return f"{prefix}_{idx}"

    if isinstance(vocab, dict):
        return str(vocab.get(idx, vocab.get(str(idx), f"{prefix}_{idx}")))

    if isinstance(vocab, (list, tuple)):
        if 0 <= idx < len(vocab):
            return str(vocab[idx])

    return f"{prefix}_{idx}"


def resolve_image_path(record: Dict[str, Any], image_root: str = "") -> Path:
    raw = str(record.get("image_path", "")).strip()
    candidates = []

    if raw:
        p = Path(raw)
        if p.exists():
            return p
        if image_root:
            candidates.append(Path(image_root) / raw)
            candidates.append(Path(image_root) / Path(raw).name)

    scene_id = str(record.get("scene_id", "")).strip()
    if image_root:
        root = Path(image_root)
        if scene_id:
            for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                candidates.append(root / f"{scene_id}{ext}")
                candidates.append(root / "image_raw" / f"{scene_id}{ext}")

    for c in candidates:
        if c.exists():
            return c

    if image_root and scene_id:
        root = Path(image_root)
        for ext in ["png", "jpg", "jpeg", "bmp"]:
            hits = list(root.rglob(f"{scene_id}.{ext}"))
            if hits:
                return hits[0]

    raise FileNotFoundError(
        f"Could not resolve image for scene_id={record.get('scene_id', '')}, "
        f"image_path={record.get('image_path', '')}"
    )


class OnnxPredGraphBuilder:
    def __init__(
        self,
        sgg_onnx: str,
        sgg_config: str,
        image_root: str,
        onnx_provider: str = "CUDAExecutionProvider",
        sgg_cpu: bool = False,
        detections_per_img: int = 100,
        rel_conf: float = 0.1,
        box_conf: float = 0.5,
    ):
        provider = "CPUExecutionProvider" if sgg_cpu else onnx_provider
        self.image_root = image_root
        self.model = SGG_ONNX_Model(
            sgg_config if sgg_config else None,
            sgg_onnx,
            provider=provider,
            dcs=detections_per_img,
            tracking=False,
            rel_conf=rel_conf,
            box_conf=box_conf,
            show_fps=False,
        )

    def sgg_arrays_to_record(self, bboxes: np.ndarray, rels: np.ndarray, clean_record: Dict[str, Any]) -> Dict[str, Any]:
        objects = []
        relationships = []

        obj_vocab = self.model.stats["obj_classes"]
        rel_vocab = self.model.stats["rel_classes"]

        for i, b in enumerate(bboxes):
            x1, y1, x2, y2 = [int(v) for v in b[:4]]
            score = float(b[4]) if len(b) > 4 else 0.0
            cls_id = int(b[5]) if len(b) > 5 else -1
            cls_name = lookup_label(obj_vocab, cls_id, "obj")

            objects.append({
                "id": f"O{i}",
                "class": cls_name,
                "bbox": [x1, y1, x2, y2],
                "score": score,
            })

        if rels is not None and len(rels) > 0:
            for r in rels:
                subj_idx = int(r[0])
                obj_idx = int(r[1])
                rel_id = int(r[2])
                rel_score = float(r[3]) if len(r) > 3 else 0.0

                if subj_idx < 0 or obj_idx < 0 or subj_idx >= len(objects) or obj_idx >= len(objects):
                    continue

                rel_name = lookup_label(rel_vocab, rel_id, "rel")
                relationships.append({
                    "subject": f"O{subj_idx}",
                    "predicate": rel_name,
                    "object": f"O{obj_idx}",
                    "score": rel_score,
                })

        pred_record = dict(clean_record)
        pred_record["objects"] = objects
        pred_record["relationships"] = relationships
        pred_record["situation"] = clean_record["situation"]
        return pred_record

    def predict_one(self, clean_record: Dict[str, Any]) -> Dict[str, Any]:
        jsonl_path = None
        if "__meta__" in clean_record and clean_record["__meta__"].get("jsonl_path"):
            jsonl_path = Path(clean_record["__meta__"]["jsonl_path"])

        img_path = None
        if jsonl_path is not None:
            img_path = resolve_image_path_from_jsonl(clean_record, jsonl_path, self.image_root)

        if img_path is None:
            raise FileNotFoundError(
                f"Could not resolve image for scene_id={clean_record.get('scene_id', '')}, "
                f"image_path={clean_record.get('image_path', '')}"
            )

        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        bboxes, rels = self.model.predict(image, visu_type="raw")
        return self.sgg_arrays_to_record(bboxes, rels, clean_record)

    def build_many(self, records: List[dict], desc: str = "ONNX SGG") -> Tuple[List[dict], int]:
        out = []
        hit = 0
        for r in tqdm(records, desc=desc):
            try:
                out.append(self.predict_one(r))
                hit += 1
            except Exception as e:
                print(f"[WARN] ONNX prediction failed for scene_id={r.get('scene_id', '')}: {e}")
        return out, hit

def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_jsonl_files(root: Path, pattern: str = "manual_labeled.jsonl") -> List[Path]:
    return sorted(root.rglob(pattern))


def resolve_image_path_from_jsonl(record: dict, jsonl_path: Path, image_root: str = "") -> Path | None:
    raw = str(record.get("image_path", "")).strip()
    if raw:
        p = Path(raw)
        if p.exists():
            return p

    image_name = Path(raw).name if raw else ""
    dataset_dir = jsonl_path.parent
    color_dir = dataset_dir.parent

    candidates = []

    if image_name:
        candidates.extend([
            color_dir / "image_raw" / image_name,
            color_dir / image_name,
            dataset_dir / image_name,
        ])

    scene_id = str(record.get("scene_id", "")).strip()
    if scene_id:
        candidates.extend([
            color_dir / "image_raw" / f"{scene_id}.png",
            color_dir / "image_raw" / f"{scene_id}.jpg",
            color_dir / "image_raw" / f"{scene_id}.jpeg",
            color_dir / f"{scene_id}.png",
            color_dir / f"{scene_id}.jpg",
            color_dir / f"{scene_id}.jpeg",
        ])

    for c in candidates:
        if c.exists():
            return c

    if image_root:
        root = Path(image_root)

        if image_name:
            hits = list(root.rglob(image_name))
            if hits:
                return hits[0]

        if scene_id:
            for ext in ("png", "jpg", "jpeg", "bmp"):
                hits = list(root.rglob(f"{scene_id}.{ext}"))
                if hits:
                    return hits[0]

    return None


def load_records_from_tree_with_meta(root: str, jsonl_pattern: str = "manual_labeled.jsonl") -> List[dict]:
    all_records = []
    for jsonl_path in find_jsonl_files(Path(root), jsonl_pattern):
        rows = load_jsonl(jsonl_path)
        for idx, row in enumerate(rows):
            row["__meta__"] = {
                "jsonl_path": str(jsonl_path),
                "row_index": idx,
            }
            all_records.append(row)
    return all_records

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True, help="Clean graph labels.jsonl/json or directory")
    parser.add_argument("--pred-graphs", type=str, default="", help="Optional predicted-graph jsonl/json from e2e output")
    parser.add_argument("--outdir", type=str, default="./runs/situation_gnn")

    parser.add_argument(
        "--task",
        type=str,
        default="multiclass",
        choices=["multiclass", "binary", "meaningful_multiclass"],
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--obj-classes", type=str, required=True, help="Path to object class txt")
    parser.add_argument("--rel-classes", type=str, required=True, help="Path to relation class txt")

    parser.add_argument("--split-mode", type=str, default="group", choices=["frame", "group"])
    parser.add_argument("--group-mode", type=str, default="segment", choices=["frame", "bag", "segment", "bag_segment"])
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--val-ratio", type=float, default=0.15)

    parser.add_argument("--use-inverse-relations", action="store_true")
    parser.add_argument("--mix-clean-train", action="store_true")
    parser.add_argument("--eval-on-pred-graphs", action="store_true")

    parser.add_argument("--aug-enabled", action="store_true")
    parser.add_argument("--drop-node-prob", type=float, default=0.05)
    parser.add_argument("--drop-edge-prob", type=float, default=0.15)
    parser.add_argument("--mask-obj-prob", type=float, default=0.10)
    parser.add_argument("--mask-rel-prob", type=float, default=0.15)
    parser.add_argument("--bbox-jitter-frac", type=float, default=0.03)
    parser.add_argument("--score-noise-std", type=float, default=0.05)
    parser.add_argument("--add-fp-edge-prob", type=float, default=0.10)
    parser.add_argument("--max-fp-edges", type=int, default=2)
    
    # For SGG
    parser.add_argument("--pred-source", type=str, default="file", choices=["file", "onnx"])
    parser.add_argument("--sgg-onnx", type=str, default="", help="Path to SGG ONNX model")
    parser.add_argument("--sgg-config", type=str, default="", help="SGG config.yml; needed if ONNX has no embedded class metadata")
    parser.add_argument("--image-root", type=str, default="", help="Root directory for resolving training images")
    parser.add_argument("--onnx-provider", type=str, default="CUDAExecutionProvider")
    parser.add_argument("--sgg-cpu", action="store_true")
    parser.add_argument("--box-conf", type=float, default=0.5)
    parser.add_argument("--rel-conf", type=float, default=0.1)
    parser.add_argument("--detections-per-img", type=int, default=100)

    args = parser.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data)
    if data_path.is_dir():
        clean_records = load_records_from_tree_with_meta(args.data, "manual_labeled.jsonl")
    else:
        clean_records = load_records(args.data)
        for i, r in enumerate(clean_records):
            r["__meta__"] = {
                "jsonl_path": str(data_path),
                "row_index": i,
            }
            
    if args.task == "meaningful_multiclass":
        clean_records = [r for r in clean_records if r["situation"] != "S0"]

    pred_records = []
    pred_map = {}
    use_onnx_pred = args.pred_source == "onnx"

    if (not use_onnx_pred) and args.pred_graphs:
        pred_records = load_records(args.pred_graphs)
        if args.task == "meaningful_multiclass":
            pred_records = [r for r in pred_records if r["situation"] != "S0"]
        pred_map = build_pred_record_map(pred_records)

    obj_classes = read_class_list(args.obj_classes)
    rel_classes = read_class_list(args.rel_classes)

    maps = make_maps(
        clean_records + pred_records,
        task=args.task,
        obj_classes=obj_classes,
        rel_classes=rel_classes,
        use_inverse_relations=args.use_inverse_relations,
    )

    label_targets = []
    for r in clean_records:
        if args.task == "binary":
            label_targets.append(0 if r["situation"] == "S0" else 1)
        else:
            label_targets.append(maps["sit2idx"][r["situation"]])

    if args.split_mode == "group":
        train_idx, val_idx, test_idx = split_indices_grouped(
            clean_records,
            label_targets,
            seed=args.seed,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
            group_mode=args.group_mode,
        )
    else:
        train_idx, val_idx, test_idx = split_indices_frame(
            label_targets,
            seed=args.seed,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
        )

    train_clean = [clean_records[i] for i in train_idx]
    val_clean = [clean_records[i] for i in val_idx]
    test_clean = [clean_records[i] for i in test_idx]

    pred_train_hits = pred_val_hits = pred_test_hits = 0
    train_pred = []
    val_pred = []
    test_pred = []

    if use_onnx_pred:
        if not args.sgg_onnx:
            raise ValueError("--sgg-onnx is required when --pred-source onnx")
        if not args.image_root:
            print("[WARN] --image-root is empty; image resolution will rely only on record['image_path'].")

        onnx_builder = OnnxPredGraphBuilder(
            sgg_onnx=args.sgg_onnx,
            sgg_config=args.sgg_config,
            image_root=args.image_root,
            onnx_provider=args.onnx_provider,
            sgg_cpu=args.sgg_cpu,
            detections_per_img=args.detections_per_img,
            rel_conf=args.rel_conf,
            box_conf=args.box_conf,
        )

        train_pred, pred_train_hits = onnx_builder.build_many(train_clean, desc="ONNX train_pred")
        val_pred, pred_val_hits = onnx_builder.build_many(val_clean, desc="ONNX val_pred")
        test_pred, pred_test_hits = onnx_builder.build_many(test_clean, desc="ONNX test_pred")

    elif pred_map:
        train_pred, pred_train_hits = match_records_by_key(train_clean, pred_map, fallback_to_base=False)
        val_pred, pred_val_hits = match_records_by_key(val_clean, pred_map, fallback_to_base=False)
        test_pred, pred_test_hits = match_records_by_key(test_clean, pred_map, fallback_to_base=False)

    if args.mix_clean_train:
        if len(train_pred) > 0:
            train_records = train_clean + train_pred
        else:
            train_records = train_clean
    else:
        if len(train_pred) > 0:
            train_records = train_pred
        else:
            train_records = train_clean

    if args.eval_on_pred_graphs and len(val_pred) > 0:
        val_records = val_pred
    else:
        val_records = val_clean

    if args.eval_on_pred_graphs and len(test_pred) > 0:
        test_records = test_pred
    else:
        test_records = test_clean

    augment_cfg = GraphAugmentConfig(
        enabled=args.aug_enabled,
        drop_node_prob=args.drop_node_prob,
        drop_edge_prob=args.drop_edge_prob,
        mask_obj_prob=args.mask_obj_prob,
        mask_rel_prob=args.mask_rel_prob,
        bbox_jitter_frac=args.bbox_jitter_frac,
        score_noise_std=args.score_noise_std,
        add_fp_edge_prob=args.add_fp_edge_prob,
        max_fp_edges=args.max_fp_edges,
    )

    train_dataset = GraphRecordDataset(
        train_records,
        maps,
        task=args.task,
        training=True,
        augment_cfg=augment_cfg,
    )

    val_graphs = build_graphs(val_records, maps, task=args.task, training=False)
    test_graphs = build_graphs(test_records, maps, task=args.task, training=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SceneSituationGNN(
        num_obj_classes=len(maps["obj_list"]),
        num_rel_classes=len(maps["rel_list"]),
        num_situation_classes=len(maps["sit_list"]),
        node_num_dim=7,
        edge_num_dim=8,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    weight_graphs = build_graphs(train_records, maps, task=args.task, training=False)
    class_weights = class_weights_from_graphs(weight_graphs, len(maps["sit_list"])).to(device)

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=8,
        verbose=True,
    )

    best_val_f1 = -1.0
    best_path = outdir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_graphs = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}")
        for batch in pbar:
            batch = batch.to(device)
            logits = model(batch)
            loss = criterion(logits, batch.y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs
            pbar.set_postfix(loss=f"{total_loss / max(1, total_graphs):.4f}")

        train_loss = total_loss / max(1, total_graphs)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_f1)

        print(
            f"[{epoch:03d}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_macro_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "maps": maps,
                    "args": vars(args),
                },
                best_path,
            )

    print(f"\nBest model saved to: {best_path}")

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, device, criterion)
    print("\n=== TEST ===")
    print(f"test_loss={test_loss:.4f}")
    print(f"test_acc={test_acc:.4f}")
    print(f"test_macro_f1={test_f1:.4f}")

    target_names = ckpt["maps"]["sit_list"]
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:")
    print(cm)

    with open(outdir / "label_maps.json", "w", encoding="utf-8") as f:
        json.dump(ckpt["maps"], f, ensure_ascii=False, indent=2)

    split_info = {
        "split_mode": args.split_mode,
        "group_mode": args.group_mode,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "num_clean_records": len(clean_records),
        "num_pred_records": len(pred_records),
        "num_train_records_used": len(train_records),
        "num_val_records_used": len(val_records),
        "num_test_records_used": len(test_records),
        "pred_train_hits": pred_train_hits,
        "pred_val_hits": pred_val_hits,
        "pred_test_hits": pred_test_hits,
    }
    with open(outdir / "splits.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    metrics = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "classification_report": classification_report(
            y_true, y_pred, target_names=target_names, digits=4, output_dict=True, zero_division=0
        ),
        "confusion_matrix": cm.tolist(),
    }
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()