from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from situation_gnn.dataset import build_graphs, load_records
from situation_gnn.model import SceneSituationGNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="jsonl/json file or directory")
    parser.add_argument("--output", type=str, default="./predictions.jsonl")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    maps = ckpt["maps"]
    task = maps["task"]

    records = load_records(args.input)
    graphs = build_graphs(records, maps, task=task)
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)

    model = SceneSituationGNN(
        num_obj_classes=len(maps["obj_list"]),
        num_rel_classes=len(maps["rel_list"]),
        num_situation_classes=len(maps["sit_list"]),
        hidden_dim=ckpt["args"]["hidden_dim"],
        num_layers=ckpt["args"]["num_layers"],
        dropout=ckpt["args"]["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    idx2sit = {i: s for i, s in enumerate(maps["sit_list"])}
    outputs = []

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
                outputs.append({
                    "scene_id": rec.get("scene_id", f"sample_{ptr}"),
                    "pred_label": idx2sit[pred_cpu[i]],
                    "pred_index": pred_cpu[i],
                    "probs": {idx2sit[j]: float(probs_cpu[i][j]) for j in range(len(probs_cpu[i]))},
                })
                ptr += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()