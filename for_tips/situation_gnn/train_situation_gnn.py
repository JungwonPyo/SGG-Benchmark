from __future__ import annotations
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sgg_benchmark.utils import parser
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from situation_gnn.dataset import (
    build_graphs,
    class_weights_from_graphs,
    load_records,
    make_maps,
    read_class_list,
)
from situation_gnn.model import SceneSituationGNN


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(labels, seed=42, test_ratio=0.15, val_ratio=0.15):
    idx = np.arange(len(labels))
    counts = Counter(labels)
    stratify = labels if min(counts.values()) >= 2 else None

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


def evaluate(model, loader, device):
    model.eval()
    all_y, all_pred = [], []
    total_loss = 0.0
    n_graphs = 0
    criterion = torch.nn.CrossEntropyLoss()

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

    acc = accuracy_score(all_y, all_pred)
    macro_f1 = f1_score(all_y, all_pred, average="macro")
    return total_loss / max(1, n_graphs), acc, macro_f1, all_y, all_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to labels.jsonl or a directory of json files")
    parser.add_argument("--outdir", type=str, default="./runs/situation_gnn")
    parser.add_argument(
        "--task",
        type=str,
        default="multiclass",
        choices=["multiclass", "binary", "meaningful_multiclass"]
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--obj-classes", type=str, required=True, help="Path to object class txt")
    parser.add_argument("--rel-classes", type=str, required=True, help="Path to relation class txt")
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.data)

    if args.task == "meaningful_multiclass":
        records = [r for r in records if r["situation"] != "S0"]

    obj_classes = read_class_list(args.obj_classes)
    rel_classes = read_class_list(args.rel_classes)

    maps = make_maps(
        records,
        task=args.task,
        obj_classes=obj_classes,
        rel_classes=rel_classes,
    )
    
    graphs = build_graphs(records, maps, task=args.task)
    labels = [int(g.y.item()) for g in graphs]

    train_idx, val_idx, test_idx = split_indices(labels, seed=args.seed)
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SceneSituationGNN(
        num_obj_classes=len(maps["obj_list"]),
        num_rel_classes=len(maps["rel_list"]),
        num_situation_classes=len(maps["sit_list"]),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    class_weights = class_weights_from_graphs(train_graphs, len(maps["sit_list"])).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs
            pbar.set_postfix(loss=f"{total_loss / max(1, total_graphs):.4f}")

        train_loss = total_loss / max(1, total_graphs)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device)

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

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, device)
    print("\n=== TEST ===")
    print(f"test_loss={test_loss:.4f}")
    print(f"test_acc={test_acc:.4f}")
    print(f"test_macro_f1={test_f1:.4f}")

    target_names = ckpt["maps"]["sit_list"]
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:")
    print(cm)

    with open(outdir / "label_maps.json", "w", encoding="utf-8") as f:
        json.dump(ckpt["maps"], f, ensure_ascii=False, indent=2)

    split_info = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }
    with open(outdir / "splits.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()