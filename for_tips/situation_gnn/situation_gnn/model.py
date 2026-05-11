from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool


class SceneSituationGNN(nn.Module):
    def __init__(
        self,
        num_obj_classes: int,
        num_rel_classes: int,
        num_situation_classes: int,
        node_num_dim: int = 6,
        edge_num_dim: int = 7,
        obj_emb_dim: int = 64,
        rel_emb_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.obj_emb = nn.Embedding(num_obj_classes, obj_emb_dim)
        self.rel_emb = nn.Embedding(num_rel_classes, rel_emb_dim)

        self.node_proj = nn.Linear(obj_emb_dim + node_num_dim, hidden_dim)
        self.edge_proj = nn.Linear(rel_emb_dim + edge_num_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_situation_classes),
        )

    def forward(self, data):
        x = torch.cat([self.obj_emb(data.x_cat), data.x_num], dim=-1)
        x = self.node_proj(x)

        if data.edge_type.numel() > 0:
            e = torch.cat([self.rel_emb(data.edge_type), data.edge_num], dim=-1)
            e = self.edge_proj(e)
        else:
            e = torch.zeros((0, x.size(-1)), device=x.device, dtype=x.dtype)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, data.edge_index, e)
            x = norm(x)
            x = F.relu(x)

        g_mean = global_mean_pool(x, data.batch)
        g_max = global_max_pool(x, data.batch)
        g = torch.cat([g_mean, g_max], dim=-1)
        logits = self.head(g)
        return logits