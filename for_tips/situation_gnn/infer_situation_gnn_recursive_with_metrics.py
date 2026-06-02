#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch_geometric.loader import DataLoader

from situation_gnn.dataset import build_graphs
from situation_gnn.model import SceneSituationGNN


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

            all_records.append(row)

    discovery = {
        "dataset_root": str(root.resolve()),
        "jsonl_pattern": jsonl_pattern,
        "jsonl_files_found": len(jsonl_files),
        "jsonl_files": [str(p) for p in jsonl_files],
        "records_loaded": len(all_records),
        "missing_images_count": len(missing_images),
        "missing_images_preview": missing_images[:50],
        "mode": "recursive_directory",
    }
    return all_records, discovery


def load_records_flexible(data_path: str | Path, jsonl_pattern: str = "manual_labeled.jsonl"):
    data_path = Path(data_path)

    if data_path.is_file():
        rows = [normalize_record_schema(r) for r in load_jsonl(data_path)]
        path_meta = extract_dataset_meta(data_path)
        for i, row in enumerate(rows):
            img_path = resolve_image_path(row, data_path)
            row["__meta__"] = {
                "jsonl_path": str(data_path.resolve()),
                "bag_name": path_meta["bag_name"],
                "segment_name": path_meta["segment_name"],
                "sensor_name": path_meta["sensor_name"],
                "row_index": i,
                "resolved_image_path": str(img_path) if img_path else "",
            }
        return rows, {
            "mode": "single_file",
            "input": str(data_path.resolve()),
            "records_loaded": len(rows),
        }

    rows, discovery = load_records_from_tree(data_path, jsonl_pattern)
    return rows, discovery


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="jsonl file or dataset root directory")
    parser.add_argument(
        "--jsonl-pattern",
        type=str,
        default="manual_labeled.jsonl",
        help="Recursive filename match when --input is a directory",
    )
    parser.add_argument("--output", type=str, default="./predictions.jsonl")
    parser.add_argument("--metrics-output", type=str, default="", help="Optional metrics json path")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    maps = ckpt["maps"]
    task = maps["task"]

    records, discovery = load_records_flexible(args.input, args.jsonl_pattern)
    if not records:
        raise ValueError("No valid records were loaded for inference.")

    graphs = build_graphs(records, maps, task=task)
    model, model_meta = build_gnn_model_from_checkpoint(ckpt, device)
    validate_graph_dims(graphs, model_meta, str(args.input))

    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)

    idx2sit = {i: s for i, s in enumerate(maps["sit_list"])}
    sit2idx = {s: i for i, s in enumerate(maps["sit_list"])}

    outputs = []
    y_true = []
    y_pred = []
    ptr = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1)

            probs_cpu = probs.cpu().tolist()
            pred_cpu = pred.cpu().tolist()

            for i in range(len(pred_cpu)):
                rec = records[ptr]
                meta = rec.get("__meta__", {})
                gt_label = rec.get("situation", rec.get("gt_label"))
                gt_index = sit2idx.get(gt_label)

                if gt_index is not None:
                    y_true.append(gt_index)
                    y_pred.append(pred_cpu[i])

                outputs.append(
                    {
                        "scene_id": rec.get("scene_id", f"sample_{ptr}"),
                        "gt_label": gt_label,
                        "gt_index": gt_index,
                        "pred_label": idx2sit[pred_cpu[i]],
                        "pred_index": pred_cpu[i],
                        "correct": None if gt_index is None else bool(gt_index == pred_cpu[i]),
                        "probs": {idx2sit[j]: float(probs_cpu[i][j]) for j in range(len(probs_cpu[i]))},
                        "source_jsonl": meta.get("jsonl_path", ""),
                        "bag_name": meta.get("bag_name", ""),
                        "segment_name": meta.get("segment_name", ""),
                        "sensor_name": meta.get("sensor_name", ""),
                    }
                )
                ptr += 1

    metrics = {
        "num_records": len(records),
        "num_predictions": len(outputs),
        "num_records_with_ground_truth": len(y_true),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "input": str(Path(args.input).resolve()),
        "task": task,
        "model_meta": model_meta,
        "discovery": discovery,
    }

    if y_true:
        labels_present = sorted(set(y_true) | set(y_pred))
        target_names = [idx2sit[i] for i in labels_present]
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
        metrics["warning"] = "No valid ground-truth situation labels found in input records, so accuracy could not be computed."

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

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
    else:
        print(metrics["warning"])


if __name__ == "__main__":
    main()