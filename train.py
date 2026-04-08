import torch
import torch.nn.functional as F
from model import GNNClassifier
import json

GRAPH_PATH      = '/home/imone/hatemirage/graph.pt'
TAXONOMY_PATH   = '/home/imone/hatemirage/taxonomy.json'
MODEL_SAVE_PATH = '/home/imone/hatemirage/model.pt'
HIDDEN_DIM = 256
EPOCHS     = 200
LR         = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

with open(TAXONOMY_PATH) as f:
    taxonomy = json.load(f)

n_targets     = len(taxonomy['target_entities'])
n_intent      = taxonomy['n_intent_clusters']
n_implication = taxonomy['n_implication_clusters']

data = torch.load(GRAPH_PATH, weights_only=False)
data = data.to(device)

n         = data.x.shape[0]
idx       = torch.randperm(n)
train_idx = idx[:80]
test_idx  = idx[80:]

model = GNNClassifier(
    in_dim=data.x.shape[1],
    hidden_dim=HIDDEN_DIM,
    n_targets=n_targets,
    n_intent_clusters=n_intent,
    n_implication_clusters=n_implication,
    num_nodes=data.x.shape[0],
    edge_index_1hop=data.edge_index,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def train():
    model.train()
    optimizer.zero_grad()
    tl, il, ml = model(data.x, data.edge_index)
    loss = (F.binary_cross_entropy_with_logits(tl[train_idx], data.target_y[train_idx]) +
            F.cross_entropy(il[train_idx], data.intent_y[train_idx]) +
            F.cross_entropy(ml[train_idx], data.implication_y[train_idx]))
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test():
    model.eval()
    tl, il, ml = model(data.x, data.edge_index)
    t_acc   = ((torch.sigmoid(tl[test_idx]) > 0.5).float() == data.target_y[test_idx]).float().mean().item()
    i_acc   = (il[test_idx].argmax(-1) == data.intent_y[test_idx]).float().mean().item()
    imp_acc = (ml[test_idx].argmax(-1) == data.implication_y[test_idx]).float().mean().item()
    return t_acc, i_acc, imp_acc

for epoch in range(1, EPOCHS + 1):
    loss = train()
    if epoch % 20 == 0:
        t_acc, i_acc, imp_acc = test()
        print(f"Epoch {epoch:03d} | Loss {loss:.3f} | "
              f"Acc T:{t_acc:.3f} I:{i_acc:.3f} Imp:{imp_acc:.3f}")

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\nModel saved to {MODEL_SAVE_PATH}")
