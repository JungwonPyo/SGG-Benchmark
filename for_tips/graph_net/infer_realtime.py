# graph_net/infer_realtime.py
"""
REACT++ 출력 scene graph → GCN → 실시간 상황 판단
Usage: python graph_net/infer_realtime.py
"""
import torch
import numpy as np
from torch_geometric.data import Data, Batch
from graph_net.models.gcn_classifier import SceneGCN

SITU_NAMES   = {0:"S1:손진입→STOP", 1:"S2:접근점유→DETOUR",
                2:"S3:궤적간섭→RETARGET", 3:"S4:인간접촉→WAIT", 4:"S5:배치점유→NORMAL"}
PATHMOD_NAMES= {0:"STOP", 1:"DETOUR", 2:"RETARGET", 3:"WAIT", 4:"NORMAL"}
CLASSES      = ["부품 박스","플라스틱 트레이","공정 부품","드라이버","작업자 손",
                "조립 지그","폐기 박스","렌치","케이블 묶음","보호 고글"]
RELATIONS    = ["on","inside","next_to","above","touching","blocking","near","beside"]
CLS2ID       = {c:i for i,c in enumerate(CLASSES)}
REL2ID       = {r:i for i,r in enumerate(RELATIONS)}

class RealtimeClassifier:
    def __init__(self, ckpt_path, device="cpu"):
        self.device = device
        self.model  = SceneGCN(num_classes=5).to(device)
        self.model.load_state_dict(
            torch.load(ckpt_path, map_location=device))
        self.model.eval()

    def scene_to_graph(self, objects, relations):
        """REACT++ 출력 → PyG Data"""
        id2idx = {o["id"]:i for i,o in enumerate(objects)}

        # Node features
        node_feats = []
        for obj in objects:
            oh = np.zeros(10); oh[CLS2ID.get(obj["class"],0)] = 1
            b  = np.array(obj["bbox"], dtype=np.float32) / 640.0
            node_feats.append(np.concatenate([oh, b]))
        x = torch.tensor(np.array(node_feats), dtype=torch.float32)

        # Edge index + edge features
        src_list, dst_list, edge_feats = [], [], []
        for rel in relations:
            si = id2idx.get(rel["subject"], -1)
            di = id2idx.get(rel["object"],  -1)
            if si<0 or di<0: continue
            src_list.append(si); dst_list.append(di)
            ef = np.zeros(8); ef[REL2ID.get(rel["predicate"],0)] = 1
            edge_feats.append(ef)

        if len(src_list) == 0:
            src_list,dst_list = [0],[0]
            edge_feats        = [np.zeros(8)]

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr  = torch.tensor(np.array(edge_feats), dtype=torch.float32)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    @torch.no_grad()
    def predict(self, objects, relations):
        graph = self.scene_to_graph(objects, relations)
        batch = Batch.from_data_list([graph]).to(self.device)
        situ_logit, path_logit = self.model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        situ_id     = situ_logit.argmax(-1).item()
        path_id     = path_logit.argmax(-1).item()
        situ_conf   = torch.softmax(situ_logit, -1).max().item()
        return {
            "situation":  SITU_NAMES[situ_id],
            "path_mod":   PATHMOD_NAMES[path_id],
            "confidence": round(situ_conf, 3)
        }

# ── Demo ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    clf = RealtimeClassifier("graph_net/checkpoints/best_model.pth")

    # 가상 REACT++ 출력 (S2 상황)
    objects = [
        {"id":"O1","class":"부품 박스","bbox":[140,190,260,320]},
        {"id":"O6","class":"조립 지그","bbox":[150,200,250,300]},
        {"id":"O3","class":"공정 부품","bbox":[100,210,140,260]},
    ]
    relations = [
        {"subject":"O1","predicate":"blocking","object":"O6"},
        {"subject":"O3","predicate":"next_to", "object":"O1"},
    ]
    result = clf.predict(objects, relations)
    print("=== 실시간 상황 판단 ===")
    print(f"  상황:    {result['situation']}")
    print(f"  경로 수정: {result['path_mod']}")
    print(f"  신뢰도:  {result['confidence']}")
