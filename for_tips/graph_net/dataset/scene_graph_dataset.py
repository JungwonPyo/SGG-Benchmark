# graph_net/dataset/scene_graph_dataset.py
import json, torch
import numpy as np
from torch.utils.data import Dataset

CLASSES  = ["부품 박스","플라스틱 트레이","공정 부품","드라이버","작업자 손",
            "조립 지그","폐기 박스","렌치","케이블 묶음","보호 고글"]
RELATIONS= ["on","inside","next_to","above","touching","blocking","near","beside"]
SITU2ID  = {"S1":0, "S2":1, "S3":2, "S4":3, "S5":4}
CLS2ID   = {c:i for i,c in enumerate(CLASSES)}
REL2ID   = {r:i for i,r in enumerate(RELATIONS)}
PATHMOD2ID = {"stop":0,"detour":1,"retarget":2,"wait":3,"normal":4}

class SceneGraphDataset(Dataset):
    def __init__(self, jsonl_path, max_nodes=10, max_edges=20):
        self.samples   = []
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if line.strip(): self.samples.append(json.loads(line))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s      = self.samples[idx]
        objs   = s["objects"]
        rels   = s.get("relationships", [])
        id2idx = {o["id"]:i for i,o in enumerate(objs)}

        # Node features: [class_onehot(10) + bbox_norm(4)] = 14-dim
        node_feats = np.zeros((self.max_nodes, 14), dtype=np.float32)
        for i, obj in enumerate(objs[:self.max_nodes]):
            oh = np.zeros(10); oh[CLS2ID.get(obj["class"],0)] = 1
            b  = np.array(obj["bbox"], dtype=np.float32) / 640.0
            node_feats[i] = np.concatenate([oh, b])

        # Edge index + edge features: relation onehot(8)
        src_list, dst_list = [], []
        edge_feats = np.zeros((self.max_edges, 8), dtype=np.float32)
        for ei, rel in enumerate(rels[:self.max_edges]):
            si = id2idx.get(rel["subject"], -1)
            di = id2idx.get(rel["object"],  -1)
            if si < 0 or di < 0: continue
            src_list.append(si); dst_list.append(di)
            edge_feats[ei, REL2ID.get(rel["predicate"], 0)] = 1

        if len(src_list) == 0: src_list,dst_list = [0],[0]
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        label = SITU2ID.get(s.get("situation","S1"), 0)
        path_mod = PATHMOD2ID.get(s.get("path_modification","normal"), 4)

        return {
            "x":          torch.tensor(node_feats, dtype=torch.float32),
            "edge_index": edge_index,
            "edge_attr":  torch.tensor(edge_feats[:len(src_list)], dtype=torch.float32),
            "label":      torch.tensor(label, dtype=torch.long),
            "path_mod":   torch.tensor(path_mod, dtype=torch.long)
        }
