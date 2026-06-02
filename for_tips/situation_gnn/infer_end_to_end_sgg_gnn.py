#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SGG_ROOT = ROOT.parent
if str(SGG_ROOT) not in sys.path:
    sys.path.insert(0, str(SGG_ROOT))

from demo.demo_model import SGG_Model
from demo.onnx_model import SGG_ONNX_Model
from for_tips.situation_gnn.situation_gnn.dataset import build_graphs
from for_tips.situation_gnn.situation_gnn.model import SceneSituationGNN


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_jsonl_files(root: Path, pattern: str) -> List[Path]:
    return sorted(root.rglob(pattern))


def normalize_record_schema(record: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(record)

    if "generated_graph" in row and isinstance(row["generated_graph"], dict):
        g = row["generated_graph"]
        row["objects"] = g.get("objects", row.get("objects", []))
        row["relationships"] = g.get("relationships", row.get("relationships", []))

    if "situation" not in row and "gt_label" in row:
        row["situation"] = row["gt_label"]

    row.setdefault("objects", [])
    row.setdefault("relationships", [])
    row.setdefault("situation", "S0")
    row.setdefault("scene_id", "")
    return row


def resolve_image_path(record: Dict[str, Any], jsonl_path: Path) -> Path | None:
    raw = str(record.get("image_path", "")).strip()
    if raw:
        p = Path(raw)
        if p.exists():
            return p

    image_name = Path(raw).name if raw else ""
    dataset_dir = jsonl_path.parent
    color_dir = dataset_dir.parent

    candidates: List[Path] = []

    if image_name:
        candidates.extend(
            [
                color_dir / "image_raw" / image_name,
                color_dir / image_name,
                dataset_dir / image_name,
            ]
        )

    scene_id = str(record.get("scene_id", "")).strip()
    if scene_id:
        candidates.extend(
            [
                color_dir / "image_raw" / f"{scene_id}.png",
                color_dir / "image_raw" / f"{scene_id}.jpg",
                color_dir / "image_raw" / f"{scene_id}.jpeg",
                color_dir / f"{scene_id}.png",
                color_dir / f"{scene_id}.jpg",
                color_dir / f"{scene_id}.jpeg",
            ]
        )

    for c in candidates:
        if c.exists():
            return c

    image_raw_dir = color_dir / "image_raw"
    if image_raw_dir.exists() and image_name:
        matches = list(image_raw_dir.rglob(image_name))
        if matches:
            return matches[0]

    return None


def extract_dataset_meta(jsonl_path: Path) -> Dict[str, str]:
    parts = list(jsonl_path.parts)
    bag_name = ""
    segment_name = ""
    sensor_name = ""

    if "dataset" in parts:
        try:
            dataset_idx = parts.index("dataset")
            if dataset_idx >= 4:
                bag_name = parts[dataset_idx - 4]
                segment_name = parts[dataset_idx - 3]
                sensor_name = "/".join(parts[dataset_idx - 2:dataset_idx])
            elif dataset_idx >= 2:
                bag_name = parts[dataset_idx - 2]
                segment_name = parts[dataset_idx - 1]
        except Exception:
            pass

    return {
        "bag_name": bag_name,
        "segment_name": segment_name,
        "sensor_name": sensor_name,
    }


def load_records_from_tree(root: Path, jsonl_pattern: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    jsonl_files = find_jsonl_files(root, jsonl_pattern)
    all_records: List[Dict[str, Any]] = []
    missing_images: List[Dict[str, Any]] = []

    for jsonl_path in jsonl_files:
        rows = load_jsonl(jsonl_path)
        path_meta = extract_dataset_meta(jsonl_path)

        for idx, row in enumerate(rows):
            row = normalize_record_schema(row)
            img_path = resolve_image_path(row, jsonl_path)

            row["__meta__"] = {
                "jsonl_path": str(jsonl_path),
                "bag_name": path_meta["bag_name"],
                "segment_name": path_meta["segment_name"],
                "sensor_name": path_meta["sensor_name"],
                "row_index": idx,
                "resolved_image_path": str(img_path) if img_path else "",
            }

            if img_path is None:
                missing_images.append(
                    {
                        "jsonl_path": str(jsonl_path),
                        "row_index": idx,
                        "scene_id": row.get("scene_id", ""),
                        "raw_image_path": row.get("image_path", ""),
                    }
                )
                continue

            all_records.append(row)

    discovery = {
        "dataset_root": str(root.resolve()),
        "jsonl_pattern": jsonl_pattern,
        "jsonl_files_found": len(jsonl_files),
        "jsonl_files": [str(p) for p in jsonl_files],
        "records_loaded": len(all_records),
        "missing_images_count": len(missing_images),
        "missing_images_preview": missing_images[:50],
    }
    return all_records, discovery


def infer_model_meta_from_checkpoint(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    state = ckpt["model_state"]
    args = ckpt.get("args", {})
    saved_meta = ckpt.get("model_meta", {})

    obj_emb_dim = int(saved_meta.get("obj_emb_dim", state["obj_emb.weight"].shape[1]))
    rel_emb_dim = int(saved_meta.get("rel_emb_dim", state["rel_emb.weight"].shape[1]))
    hidden_dim = int(saved_meta.get("hidden_dim", state["node_proj.weight"].shape[0]))

    node_proj_in = int(state["node_proj.weight"].shape[1])
    edge_proj_in = int(state["edge_proj.weight"].shape[1])

    node_num_dim = int(saved_meta.get("node_num_dim", node_proj_in - obj_emb_dim))
    edge_num_dim = int(saved_meta.get("edge_num_dim", edge_proj_in - rel_emb_dim))

    conv_ids = sorted(
        {
            int(k.split(".")[1])
            for k in state.keys()
            if k.startswith("convs.") and k.split(".")[1].isdigit()
        }
    )
    num_layers = int(saved_meta.get("num_layers", args.get("num_layers", len(conv_ids))))
    dropout = float(saved_meta.get("dropout", args.get("dropout", 0.0)))

    return {
        "obj_emb_dim": obj_emb_dim,
        "rel_emb_dim": rel_emb_dim,
        "hidden_dim": hidden_dim,
        "node_num_dim": node_num_dim,
        "edge_num_dim": edge_num_dim,
        "num_layers": num_layers,
        "dropout": dropout,
    }


def build_gnn_model_from_checkpoint(ckpt: Dict[str, Any], device: str):
    maps = ckpt["maps"]
    meta = infer_model_meta_from_checkpoint(ckpt)

    model = SceneSituationGNN(
        num_obj_classes=len(maps["obj_list"]),
        num_rel_classes=len(maps["rel_list"]),
        num_situation_classes=len(maps["sit_list"]),
        node_num_dim=meta["node_num_dim"],
        edge_num_dim=meta["edge_num_dim"],
        obj_emb_dim=meta["obj_emb_dim"],
        rel_emb_dim=meta["rel_emb_dim"],
        hidden_dim=meta["hidden_dim"],
        num_layers=meta["num_layers"],
        dropout=meta["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, meta


def validate_graph_dims(graphs, meta: Dict[str, Any], source_name: str):
    if not graphs:
        return

    node_dims = sorted({int(g.x_num.size(-1)) for g in graphs if hasattr(g, "x_num") and g.x_num.dim() == 2})
    edge_dims = sorted({int(g.edge_num.size(-1)) for g in graphs if hasattr(g, "edge_num") and g.edge_num.dim() == 2})

    actual_node_dim = node_dims[0] if node_dims else 0
    actual_edge_dim = edge_dims[0] if edge_dims else 0

    if actual_node_dim != meta["node_num_dim"] or actual_edge_dim != meta["edge_num_dim"]:
        raise ValueError(
            f"Graph/checkpoint feature mismatch for {source_name}: "
            f"checkpoint expects node_num_dim={meta['node_num_dim']}, edge_num_dim={meta['edge_num_dim']}, "
            f"but built graphs have node_num_dim={actual_node_dim}, edge_num_dim={actual_edge_dim}. "
            f"This usually means the inference dataset.py feature schema does not match the checkpoint."
        )


class EndToEndSGGGNNEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

        self.is_onnx = args.sgg_weights.lower().endswith(".onnx")
        if self.is_onnx:
            provider = "CPUExecutionProvider" if args.cpu else args.onnx_provider
            self.sgg_model = SGG_ONNX_Model(
                args.sgg_config,
                args.sgg_weights,
                provider=provider,
                dcs=args.detections_per_img,
                tracking=args.enable_tracking,
                rel_conf=args.rel_conf,
                box_conf=args.box_conf,
                show_fps=False,
            )
        else:
            self.sgg_model = SGG_Model(
                args.sgg_config,
                args.sgg_weights,
                dcs=args.detections_per_img,
                tracking=args.enable_tracking,
                rel_conf=args.rel_conf,
                box_conf=args.box_conf,
                show_fps=False,
            )

        self.gnn_ckpt = torch.load(args.gnn_checkpoint, map_location=self.device, weights_only=False)
        self.maps = self.gnn_ckpt["maps"]
        self.task = self.maps["task"]
        self.idx2sit = {i: s for i, s in enumerate(self.maps["sit_list"])}
        self.sit2idx = {s: i for i, s in enumerate(self.maps["sit_list"])}
        self.gnn_model, self.gnn_meta = build_gnn_model_from_checkpoint(self.gnn_ckpt, self.device)

    def lookup_label(self, vocab, idx: int, prefix: str) -> str:
        if vocab is None:
            return f"{prefix}_{idx}"

        if isinstance(vocab, dict):
            return str(vocab.get(idx, vocab.get(str(idx), f"{prefix}_{idx}")))

        if isinstance(vocab, (list, tuple)):
            if 0 <= idx < len(vocab):
                return str(vocab[idx])
            return f"{prefix}_{idx}"

        return f"{prefix}_{idx}"

    def predict_graph(self, image: np.ndarray) -> Dict[str, Any]:
        if self.is_onnx:
            bboxes, rels = self.sgg_model.predict(image, visu_type="raw")
        else:
            self.sgg_model.model.roi_heads.eval()
            self.sgg_model.model.backbone.eval()

            img_list, _ = self.sgg_model._pre_processing(image)
            img_list.image_sizes = [(image.shape[0], image.shape[1])]
            img_list = img_list.to(self.sgg_model.device)

            with torch.no_grad():
                predictions = self.sgg_model.model(img_list, None, return_attention=False)

            bboxes, rels = self.sgg_model._post_process2(
                predictions[0],
                orig_size=image.shape[:2],
                box_thres=self.args.box_conf,
                rel_threshold=self.args.rel_conf,
            )

        bboxes = bboxes.cpu().numpy() if hasattr(bboxes, "cpu") else np.asarray(bboxes)
        rels = rels.cpu().numpy() if hasattr(rels, "cpu") else np.asarray(rels)
        return {"bboxes": bboxes, "rels": rels}

    def sgg_arrays_to_record(self, bboxes: np.ndarray, rels: np.ndarray, gt_record: Dict[str, Any]) -> Dict[str, Any]:
        objects: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []

        obj_vocab = self.sgg_model.stats["obj_classes"]
        rel_vocab = self.sgg_model.stats["rel_classes"]

        for i, b in enumerate(bboxes):
            x1, y1, x2, y2 = [int(v) for v in b[:4]]
            score = float(b[4]) if len(b) > 4 else 0.0
            cls_id = int(b[5]) if len(b) > 5 else -1
            cls_name = self.lookup_label(obj_vocab, cls_id, "obj")

            objects.append(
                {
                    "id": f"O{i}",
                    "class": cls_name,
                    "bbox": [x1, y1, x2, y2],
                    "score": score,
                }
            )

        if rels is not None and len(rels) > 0:
            for r in rels:
                subj_idx = int(r[0])
                obj_idx = int(r[1])
                rel_id = int(r[2])
                rel_score = float(r[3]) if len(r) > 3 else 0.0

                if subj_idx < 0 or obj_idx < 0 or subj_idx >= len(objects) or obj_idx >= len(objects):
                    continue

                rel_name = self.lookup_label(rel_vocab, rel_id, "rel")
                relationships.append(
                    {
                        "subject": f"O{subj_idx}",
                        "predicate": rel_name,
                        "object": f"O{obj_idx}",
                        "score": rel_score,
                    }
                )

        return {
            "scene_id": gt_record.get("scene_id", Path(gt_record["__meta__"]["resolved_image_path"]).stem),
            "situation": gt_record.get("situation", gt_record.get("gt_label", "S0")),
            "objects": objects,
            "relationships": relationships,
        }

    def predict_situation(self, record: Dict[str, Any]) -> Dict[str, Any]:
        graphs = build_graphs([record], self.maps, task=self.task)
        validate_graph_dims(graphs, self.gnn_meta, record.get("scene_id", "single_record"))

        loader = DataLoader(graphs, batch_size=1, shuffle=False)

        with torch.no_grad():
            batch = next(iter(loader)).to(self.device)
            logits = self.gnn_model(batch)
            probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
            pred_idx = int(np.argmax(probs))

        return {
            "pred_index": pred_idx,
            "pred_label": self.idx2sit[pred_idx],
            "probs": {self.idx2sit[i]: float(probs[i]) for i in range(len(probs))},
        }

    def draw_overlay(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        rels: np.ndarray,
        pred: Dict[str, Any],
        gt_label: str,
    ) -> np.ndarray:
        image = self.sgg_model.draw_full_graph(image, bboxes, rels)

        topk = sorted(pred["probs"].items(), key=lambda x: x[1], reverse=True)[:3]
        title = f"GT: {gt_label} | Pred: {pred['pred_label']}"
        prob_text = " | ".join([f"{k}:{v:.2f}" for k, v in topk])

        panel_w = min(image.shape[1] - 20, 1100)
        cv2.rectangle(image, (10, 10), (10 + panel_w, 90), (20, 20, 20), -1)
        cv2.putText(image, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, prob_text, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        return image


def build_argparser():
    p = argparse.ArgumentParser(description="Offline end-to-end SGG -> GNN evaluator")
    p.add_argument("--input", type=str, required=True, help="Dataset root or one jsonl file")
    p.add_argument("--jsonl-pattern", type=str, default="manual_labeled.jsonl")
    p.add_argument("--sgg-config", type=str, required=True)
    p.add_argument("--sgg-weights", type=str, required=True)
    p.add_argument("--gnn-checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default="./e2e_predictions.jsonl")
    p.add_argument("--metrics-output", type=str, default="")
    p.add_argument("--save-vis-dir", type=str, default="")
    p.add_argument("--box-conf", type=float, default=0.5)
    p.add_argument("--rel-conf", type=float, default=0.1)
    p.add_argument("--detections-per-img", type=int, default=100)
    p.add_argument("--enable-tracking", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--onnx-provider", type=str, default="CUDAExecutionProvider")
    return p


def main():
    args = build_argparser().parse_args()

    for path_arg in [args.sgg_config, args.sgg_weights, args.gnn_checkpoint]:
        if not Path(path_arg).exists():
            raise FileNotFoundError(f"Required file not found: {path_arg}")

    input_path = Path(args.input)
    if input_path.is_file():
        raw_records = load_jsonl(input_path)
        path_meta = extract_dataset_meta(input_path)

        records = []
        for i, r in enumerate(raw_records):
            r = normalize_record_schema(r)
            img_path = resolve_image_path(r, input_path)
            r["__meta__"] = {
                "jsonl_path": str(input_path.resolve()),
                "bag_name": path_meta["bag_name"],
                "segment_name": path_meta["segment_name"],
                "sensor_name": path_meta["sensor_name"],
                "row_index": i,
                "resolved_image_path": str(img_path) if img_path else "",
            }
            if img_path is not None:
                records.append(r)

        discovery = {
            "mode": "single_file",
            "input": str(input_path.resolve()),
            "records_loaded": len(records),
        }
    else:
        records, discovery = load_records_from_tree(input_path, args.jsonl_pattern)

    if not records:
        raise ValueError("No valid records found.")

    evaluator = EndToEndSGGGNNEvaluator(args)

    outputs: List[Dict[str, Any]] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    save_vis_dir = Path(args.save_vis_dir) if args.save_vis_dir else None
    if save_vis_dir:
        save_vis_dir.mkdir(parents=True, exist_ok=True)

    for idx, gt_record in enumerate(records):
        image_path = gt_record["__meta__"]["resolved_image_path"]
        image = cv2.imread(image_path)
        if image is None:
            outputs.append(
                {
                    "scene_id": gt_record.get("scene_id", f"sample_{idx}"),
                    "image_path": image_path,
                    "source_jsonl": gt_record["__meta__"]["jsonl_path"],
                    "bag_name": gt_record["__meta__"]["bag_name"],
                    "segment_name": gt_record["__meta__"]["segment_name"],
                    "sensor_name": gt_record["__meta__"]["sensor_name"],
                    "gt_label": gt_record.get("situation", gt_record.get("gt_label")),
                    "error": "Failed to read image with cv2.imread",
                }
            )
            continue

        try:
            sgg = evaluator.predict_graph(image)
            pred_record = evaluator.sgg_arrays_to_record(sgg["bboxes"], sgg["rels"], gt_record)
            pred = evaluator.predict_situation(pred_record)
        except Exception as e:
            outputs.append(
                {
                    "scene_id": gt_record.get("scene_id", f"sample_{idx}"),
                    "image_path": image_path,
                    "source_jsonl": gt_record["__meta__"]["jsonl_path"],
                    "bag_name": gt_record["__meta__"]["bag_name"],
                    "segment_name": gt_record["__meta__"]["segment_name"],
                    "sensor_name": gt_record["__meta__"]["sensor_name"],
                    "gt_label": gt_record.get("situation", gt_record.get("gt_label")),
                    "error": str(e),
                }
            )
            continue

        gt_label = gt_record.get("situation", gt_record.get("gt_label"))
        gt_index = evaluator.sit2idx.get(gt_label)

        if gt_index is not None:
            y_true.append(gt_index)
            y_pred.append(pred["pred_index"])

        row = {
            "scene_id": gt_record.get("scene_id", f"sample_{idx}"),
            "image_path": image_path,
            "source_jsonl": gt_record["__meta__"]["jsonl_path"],
            "bag_name": gt_record["__meta__"]["bag_name"],
            "segment_name": gt_record["__meta__"]["segment_name"],
            "sensor_name": gt_record["__meta__"]["sensor_name"],
            "gt_label": gt_label,
            "gt_index": gt_index,
            "pred_label": pred["pred_label"],
            "pred_index": pred["pred_index"],
            "correct": None if gt_index is None else bool(gt_index == pred["pred_index"]),
            "num_objects": len(pred_record["objects"]),
            "num_relations": len(pred_record["relationships"]),
            "probs": pred["probs"],
            "generated_graph": {
                "objects": pred_record["objects"],
                "relationships": pred_record["relationships"],
            },
        }
        outputs.append(row)

        if save_vis_dir:
            vis = evaluator.draw_overlay(image.copy(), sgg["bboxes"], sgg["rels"], pred, gt_label or "NA")
            safe_scene_id = str(row["scene_id"]).replace("/", "_")
            stem = f"{idx:06d}_{safe_scene_id}"
            cv2.imwrite(str(save_vis_dir / f"{stem}.jpg"), vis)

        if (idx + 1) % 50 == 0:
            print(f"[{idx + 1}/{len(records)}] processed")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics: Dict[str, Any] = {
        "num_records": len(records),
        "num_predictions": len([o for o in outputs if "pred_label" in o]),
        "num_records_with_ground_truth": len(y_true),
        "model_meta": evaluator.gnn_meta,
        "discovery": discovery,
    }

    if y_true:
        labels_present = sorted(set(y_true) | set(y_pred))
        target_names = [evaluator.idx2sit[i] for i in labels_present]
        metrics.update(
            {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
                "confusion_matrix_labels": target_names,
                "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels_present).tolist(),
                "classification_report": classification_report(
                    y_true,
                    y_pred,
                    labels=labels_present,
                    target_names=target_names,
                    digits=4,
                    output_dict=True,
                    zero_division=0,
                ),
            }
        )
    else:
        metrics["warning"] = "No valid ground-truth labels found for accuracy computation."

    metrics_path = Path(args.metrics_output) if args.metrics_output else out_path.with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    discovery_path = out_path.with_suffix(".discovery.json")
    with open(discovery_path, "w", encoding="utf-8") as f:
        json.dump(discovery, f, ensure_ascii=False, indent=2)

    print(f"Saved predictions to: {out_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved discovery info to: {discovery_path}")
    if "accuracy" in metrics:
        print(f"accuracy={metrics['accuracy']:.4f}")
        print(f"macro_f1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()