import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

class GNNClassifier(nn.Module):
    def __init__(self, hidden_dim, n_targets, n_intent, n_impl):
        super().__init__()
        # lazy init (-1) handles mixed input dims across node types
        self.conv1 = HeteroConv({
            ('comment', 'sim', 'comment'):       SAGEConv(-1, hidden_dim),
            ('topic', 'about_rev', 'comment'):   SAGEConv(-1, hidden_dim),
            ('claim', 'targets_rev', 'comment'): SAGEConv(-1, hidden_dim),
        }, aggr='sum')
        self.conv2 = HeteroConv({
            ('comment', 'sim', 'comment'):       SAGEConv(-1, hidden_dim),
            ('topic', 'about_rev', 'comment'):   SAGEConv(-1, hidden_dim),
            ('claim', 'targets_rev', 'comment'): SAGEConv(-1, hidden_dim),
        }, aggr='sum')
        self.target_head      = nn.Linear(hidden_dim, n_targets)
        self.intent_head      = nn.Linear(hidden_dim, n_intent)
        self.implication_head = nn.Linear(hidden_dim + n_targets + n_intent, n_impl)

    def forward(self, x_dict, edge_index_dict):
        x0 = x_dict.copy()  # topic/claim are static providers, preserve originals

        out = self.conv1(x_dict, edge_index_dict)
        x_dict = {**x0, **out}   # overwrite only comment
        x_dict['comment'] = F.relu(F.dropout(x_dict['comment'], p=0.3, training=self.training))

        out = self.conv2(x_dict, edge_index_dict)
        x_dict = {**x0, **out}   # again overwrite only comment
        x_dict['comment'] = F.relu(x_dict['comment'])

        h = x_dict['comment']
        target_logits = self.target_head(h)
        intent_logits = self.intent_head(h)
        target_probs  = torch.sigmoid(target_logits)
        intent_probs  = F.softmax(intent_logits, dim=-1)
        impl_logits   = self.implication_head(
            torch.cat([h, target_probs, intent_probs], dim=-1)
        )
        return target_logits, intent_logits, impl_logits
