# graph_net/train.py
import os, torch, argparse
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Data, Batch
from graph_net.dataset.scene_graph_dataset import SceneGraphDataset
from graph_net.models.gcn_classifier import SceneGCN
from torch.utils.tensorboard import SummaryWriter

SITU_NAMES   = ["S1:손진입","S2:접근점유","S3:궤적간섭","S4:인간접촉","S5:배치점유"]
PATHMOD_NAMES= ["stop","detour","retarget","wait","normal"]

def collate(batch):
    graphs = []
    labels, pmods = [], []
    for b in batch:
        g = Data(
            x          = b["x"],
            edge_index = b["edge_index"],
            edge_attr  = b["edge_attr"]
        )
        graphs.append(g)
        labels.append(b["label"])
        pmods.append(b["path_mod"])
    batched = Batch.from_data_list(graphs)
    return batched, torch.stack(labels), torch.stack(pmods)

def train(args):
    ds = SceneGraphDataset(args.jsonl)
    n_val   = max(1, int(len(ds)*0.2))
    n_train = len(ds) - n_val
    tr, va  = random_split(ds, [n_train, n_val])
    tr_loader = DataLoader(tr, batch_size=args.bs, shuffle=True,  collate_fn=collate)
    va_loader = DataLoader(va, batch_size=args.bs, shuffle=False, collate_fn=collate)

    model = SceneGCN(num_classes=5).to(args.device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit  = nn.CrossEntropyLoss()
    writer= SummaryWriter(f"runs/{args.exp}")
    best_acc = 0.0

    for ep in range(1, args.epochs+1):
        model.train()
        total_loss = 0
        for batch_g, labels, pmods in tr_loader:
            batch_g = batch_g.to(args.device)
            labels  = labels.to(args.device)
            pmods   = pmods.to(args.device)
            opt.zero_grad()
            situ_out, path_out = model(batch_g.x, batch_g.edge_index,
                                       batch_g.edge_attr, batch_g.batch)
            loss = crit(situ_out, labels) + 0.5*crit(path_out, pmods)
            loss.backward(); opt.step()
            total_loss += loss.item()
        sch.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_g, labels, pmods in va_loader:
                batch_g = batch_g.to(args.device)
                labels  = labels.to(args.device)
                situ_out, _ = model(batch_g.x, batch_g.edge_index,
                                    batch_g.edge_attr, batch_g.batch)
                correct += (situ_out.argmax(-1) == labels).sum().item()
                total   += len(labels)
        acc = correct / total if total > 0 else 0

        writer.add_scalar("Loss/train", total_loss/len(tr_loader), ep)
        writer.add_scalar("Acc/val", acc, ep)
        print(f"[{ep:03d}/{args.epochs}] loss={total_loss/len(tr_loader):.4f} val_acc={acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            os.makedirs(args.ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{args.ckpt_dir}/best_model.pth")
            print(f"  ✅ Best model saved (acc={acc:.3f})")

    writer.close()
    print(f"\n🎉 Training complete. Best val acc: {best_acc:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl",    default="data/jsonl/manual_labeled.jsonl")
    ap.add_argument("--epochs",   type=int,   default=50)
    ap.add_argument("--bs",       type=int,   default=16)
    ap.add_argument("--lr",       type=float, default=1e-3)
    ap.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt_dir", default="graph_net/checkpoints")
    ap.add_argument("--exp",      default="robot_gcn_exp1")
    args = ap.parse_args()
    train(args)
