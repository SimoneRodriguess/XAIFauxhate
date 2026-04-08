import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse


def compute_khop_edge_index(edge_index, num_nodes, k):
    A = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    Ak = A.clone()
    for _ in range(k - 1):
        Ak = torch.bmm(Ak.unsqueeze(0), A.unsqueeze(0)).squeeze(0)
    Ak = (Ak > 0).float()
    edge_index_k, _ = dense_to_sparse(Ak)
    return edge_index_k


class MHMOGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, orders=(1, 2, 3), heads=2, dropout=0.3):
        super().__init__()
        self.orders  = orders
        self.heads   = heads
        self.dropout = dropout
        head_dim     = max(out_dim // (len(orders) * heads), 1)
        self.convs   = nn.ModuleList([
            GATConv(in_dim, head_dim, heads=heads, dropout=dropout, concat=True)
            for _ in orders
        ])
        self.self_lin  = nn.Linear(in_dim, out_dim)
        concat_dim     = len(orders) * heads * head_dim + out_dim
        self.proj      = nn.Linear(concat_dim, out_dim)
        self.norm      = nn.LayerNorm(out_dim)

    def forward(self, x, edge_indices):
        order_outs = [conv(x, edge_indices[o]) for o, conv in zip(self.orders, self.convs)]
        self_out   = self.self_lin(x)
        h = torch.cat(order_outs + [self_out], dim=-1)
        return self.norm(self.proj(h))


class GNNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_targets, n_intent_clusters,
                 n_implication_clusters, num_nodes, edge_index_1hop,
                 orders=(1, 2, 3), gat_heads=2, dropout=0.3):
        super().__init__()
        self.orders    = list(orders)
        self.num_nodes = num_nodes
        self.dropout_p = dropout

        self.register_buffer("edge_index_1", edge_index_1hop)
        for k in orders:
            if k == 1:
                continue
            self.register_buffer(f"edge_index_{k}",
                compute_khop_edge_index(edge_index_1hop, num_nodes, k))

        self.layer1 = MHMOGATLayer(in_dim,     hidden_dim, orders, gat_heads, dropout)
        self.layer2 = MHMOGATLayer(hidden_dim, hidden_dim, orders, gat_heads, dropout)

        self.target_head      = nn.Linear(hidden_dim, n_targets)
        self.intent_head      = nn.Linear(hidden_dim, n_intent_clusters)
        self.implication_head = nn.Linear(
            hidden_dim + n_targets + n_intent_clusters, n_implication_clusters)

    def _edge_indices(self):
        return {k: getattr(self, f"edge_index_{k}") for k in self.orders}

    def forward(self, x, edge_index=None):
        ei = self._edge_indices()
        h  = F.dropout(F.relu(self.layer1(x, ei)), p=self.dropout_p, training=self.training)
        h  = F.relu(self.layer2(h, ei))
        tl = self.target_head(h)
        il = self.intent_head(h)
        tp = torch.sigmoid(tl)
        ip = F.softmax(il, dim=-1)
        ml = self.implication_head(torch.cat([h, tp, ip], dim=-1))
        return tl, il, ml
