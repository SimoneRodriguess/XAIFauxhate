import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GNNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_targets, n_intent_clusters, n_implication_clusters):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        self.target_head = nn.Linear(hidden_dim, n_targets)
        self.intent_head = nn.Linear(hidden_dim, n_intent_clusters)
        self.implication_head = nn.Linear(hidden_dim + n_targets + n_intent_clusters, n_implication_clusters)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.3, training=self.training)
        h = F.relu(self.conv2(h, edge_index))

        target_logits = self.target_head(h)
        intent_logits = self.intent_head(h)

        target_probs = torch.sigmoid(target_logits)
        intent_probs = F.softmax(intent_logits, dim=-1)
        implication_input = torch.cat([h, target_probs, intent_probs], dim=-1)
        implication_logits = self.implication_head(implication_input)

        return target_logits, intent_logits, implication_logits
