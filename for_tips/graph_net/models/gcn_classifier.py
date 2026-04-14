# graph_net/models/gcn_classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class SceneGCN(nn.Module):
    """
    REACT++ Scene Graph → 상황 분류 (S1~S5)
    Architecture: GATConv x2 → global pool → MLP
    """
    def __init__(self, node_dim=14, edge_dim=8, hidden=64, num_classes=5):
        super().__init__()
        self.edge_proj = nn.Linear(edge_dim, node_dim)

        # GAT layers (attention-based → 관계 가중치 반영)
        self.gat1 = GATConv(node_dim, hidden, heads=4, dropout=0.3, concat=True)
        self.gat2 = GATConv(hidden*4, hidden, heads=1, dropout=0.3, concat=False)

        # GCN residual
        self.gcn  = GCNConv(node_dim, hidden)

        # Fusion
        self.fusion = nn.Linear(hidden * 2, hidden)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
        # Path modification head (stop/detour/wait/retarget/normal)
        self.path_head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # GAT branch
        g1 = F.elu(self.gat1(x, edge_index))
        g1 = F.dropout(g1, p=0.3, training=self.training)
        g2 = F.elu(self.gat2(g1, edge_index))

        # GCN branch (residual)
        gc = F.relu(self.gcn(x, edge_index))

        # Fusion
        fused = torch.cat([g2, gc], dim=-1)
        fused = F.relu(self.fusion(fused))

        # Global pool
        pooled = global_mean_pool(fused, batch)

        situ     = self.classifier(pooled)
        path_mod = self.path_head(pooled)
        return situ, path_mod
