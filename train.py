import torch
import torch.nn.functional as F
from model import GNNClassifier
import json

GRAPH_PATH     = '/home/imone/hatemirage/graph.pt'
TAXONOMY_PATH  = '/home/imone/hatemirage/taxonomy.json'
MODEL_SAVE_PATH= '/home/imone/hatemirage/model.pt'
HIDDEN_DIM = 256
EPOCHS = 200
LR = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

with open(TAXONOMY_PATH) as f:
    taxonomy = json.load(f)

n_targets    = len(taxonomy['target_entities'])
n_intent     = taxonomy['n_intent_clusters']
n_implication= taxonomy['n_implication_clusters']

data = torch.load(GRAPH_PATH, weights_only=False).to(device)

n = data['comment'].x.shape[0]
idx = torch.randperm(n)
train_idx = idx[:80]
test_idx  = idx[80:]

model = GNNClassifier(
    hidden_dim=HIDDEN_DIM,
    n_targets=n_targets,
    n_intent=n_intent,
    n_impl=n_implication,
).to(device)

# initialize lazy SAGEConv weights with one forward pass before optimizer
with torch.no_grad():
    model(data.x_dict, data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def train():
    model.train()
    optimizer.zero_grad()
    t_logits, i_logits, imp_logits = model(data.x_dict, data.edge_index_dict)
    loss_t   = F.binary_cross_entropy_with_logits(t_logits[train_idx],   data['comment'].target_y[train_idx])
    loss_i   = F.cross_entropy(i_logits[train_idx],   data['comment'].intent_y[train_idx])
    loss_imp = F.cross_entropy(imp_logits[train_idx], data['comment'].implication_y[train_idx])
    loss = loss_t + loss_i + loss_imp
    loss.backward()
    optimizer.step()
    return loss_t.item(), loss_i.item(), loss_imp.item()

@torch.no_grad()
def test():
    model.eval()
    t_logits, i_logits, imp_logits = model(data.x_dict, data.edge_index_dict)
    t_acc   = ((torch.sigmoid(t_logits[test_idx]) > 0.5).float() == data['comment'].target_y[test_idx]).float().mean().item()
    i_acc   = (i_logits[test_idx].argmax(-1) == data['comment'].intent_y[test_idx]).float().mean().item()
    imp_acc = (imp_logits[test_idx].argmax(-1) == data['comment'].implication_y[test_idx]).float().mean().item()
    return t_acc, i_acc, imp_acc

for epoch in range(1, EPOCHS + 1):
    l_t, l_i, l_imp = train()
    if epoch % 20 == 0:
        t_acc, i_acc, imp_acc = test()
        print(f"Epoch {epoch:03d} | Loss T:{l_t:.3f} I:{l_i:.3f} Imp:{l_imp:.3f} | Acc T:{t_acc:.3f} I:{i_acc:.3f} Imp:{imp_acc:.3f}")

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
