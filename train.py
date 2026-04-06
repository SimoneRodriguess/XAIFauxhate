import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from model import GNNClassifier
import json

GRAPH_PATH = '/home/imone/hatemirage/graph.pt'
TAXONOMY_PATH = '/home/imone/hatemirage/taxonomy.json'
MODEL_SAVE_PATH = '/home/imone/hatemirage/model.pt'
HIDDEN_DIM = 256
EPOCHS = 200
LR = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

with open(TAXONOMY_PATH) as f:
    taxonomy = json.load(f)

n_targets = len(taxonomy['target_entities'])
n_intent = taxonomy['n_intent_clusters']
n_implication = taxonomy['n_implication_clusters']

data = torch.load(GRAPH_PATH, weights_only=False)
data = data.to(device)

n = data.x.shape[0]
idx = torch.randperm(n)
train_idx = idx[:80]
test_idx = idx[80:]

model = GNNClassifier(
    in_dim=data.x.shape[1],
    hidden_dim=HIDDEN_DIM,
    n_targets=n_targets,
    n_intent_clusters=n_intent,
    n_implication_clusters=n_implication,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def train():
    model.train()
    optimizer.zero_grad()
    target_logits, intent_logits, implication_logits = model(data.x, data.edge_index)
    loss_target = F.binary_cross_entropy_with_logits(
        target_logits[train_idx], data.target_y[train_idx])
    loss_intent = F.cross_entropy(
        intent_logits[train_idx], data.intent_y[train_idx])
    loss_implication = F.cross_entropy(
        implication_logits[train_idx], data.implication_y[train_idx])
    loss = loss_target + loss_intent + loss_implication
    loss.backward()
    optimizer.step()
    return loss_target.item(), loss_intent.item(), loss_implication.item()

@torch.no_grad()
def test():
    model.eval()
    target_logits, intent_logits, implication_logits = model(data.x, data.edge_index)

    target_preds = (torch.sigmoid(target_logits[test_idx]) > 0.5).float()
    target_acc = (target_preds == data.target_y[test_idx]).float().mean().item()

    intent_preds = intent_logits[test_idx].argmax(dim=-1)
    intent_acc = (intent_preds == data.intent_y[test_idx]).float().mean().item()

    impl_preds = implication_logits[test_idx].argmax(dim=-1)
    impl_acc = (impl_preds == data.implication_y[test_idx]).float().mean().item()

    return target_acc, intent_acc, impl_acc

for epoch in range(1, EPOCHS + 1):
    l_t, l_i, l_imp = train()
    if epoch % 20 == 0:
        t_acc, i_acc, imp_acc = test()
        print(f"Epoch {epoch:03d} | "
              f"Loss T:{l_t:.3f} I:{l_i:.3f} Imp:{l_imp:.3f} | "
              f"Acc T:{t_acc:.3f} I:{i_acc:.3f} Imp:{imp_acc:.3f}")

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\nModel saved to {MODEL_SAVE_PATH}")
