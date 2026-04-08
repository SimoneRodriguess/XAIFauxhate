import torch
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from model import GNNClassifier, compute_khop_edge_index

TAXONOMY_PATH = '/home/imone/hatemirage/taxonomy.json'
GRAPH_PATH    = '/home/imone/hatemirage/graph.pt'
MODEL_PATH    = '/home/imone/hatemirage/model.pt'
MODEL_NAME    = 'sentence-transformers/all-mpnet-base-v2'
HIDDEN_DIM    = 256
SIM_THRESHOLD = 0.7
TOP_K         = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(TAXONOMY_PATH) as f:
    taxonomy = json.load(f)

target_entities  = taxonomy['target_entities']
intent_reps      = taxonomy['intent_cluster_representatives']
implication_reps = taxonomy['implication_cluster_representatives']
n_targets        = len(target_entities)
n_intent         = taxonomy['n_intent_clusters']
n_implication    = taxonomy['n_implication_clusters']

encoder = SentenceTransformer(MODEL_NAME)
data    = torch.load(GRAPH_PATH, weights_only=False)

model = GNNClassifier(
    in_dim=data.x.shape[1],
    hidden_dim=HIDDEN_DIM,
    n_targets=n_targets,
    n_intent_clusters=n_intent,
    n_implication_clusters=n_implication,
    num_nodes=data.x.shape[0],
    edge_index_1hop=data.edge_index,
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device).eval()
data  = data.to(device)

def infer(comment):
    emb        = encoder.encode([comment])
    emb_tensor = torch.tensor(emb, dtype=torch.float).to(device)

    existing    = data.x.cpu().numpy()
    norms_exist = np.linalg.norm(existing, axis=1, keepdims=True)
    norms_new   = np.linalg.norm(emb,      axis=1, keepdims=True)
    sims        = (existing / (norms_exist + 1e-8)) @ (emb / (norms_new + 1e-8)).T
    sims        = sims.flatten()
    top_k_idx   = np.argsort(sims)[::-1][:TOP_K]
    neighbors   = [j for j in top_k_idx if sims[j] >= SIM_THRESHOLD]

    new_x        = torch.cat([data.x, emb_tensor], dim=0)
    new_node_idx = data.x.shape[0]
    new_edge_index = (torch.cat([data.edge_index,
        torch.tensor([[new_node_idx]*len(neighbors), neighbors],
                     dtype=torch.long).to(device)], dim=1)
        if neighbors else data.edge_index)

    num_nodes_new    = new_x.shape[0]
    original_buffers = {}
    for k in model.orders:
        buf = f"edge_index_{k}"
        original_buffers[buf] = getattr(model, buf).clone()
        setattr(model, buf,
            new_edge_index if k == 1
            else compute_khop_edge_index(new_edge_index, num_nodes_new, k).to(device))

    with torch.no_grad():
        tl, il, ml = model(new_x, new_edge_index)

    for buf, val in original_buffers.items():
        setattr(model, buf, val)

    tp  = torch.sigmoid(tl[new_node_idx])
    targets = [target_entities[i] for i, p in enumerate(tp) if p > 0.4]
    print(f"\nComment:     {comment}")
    print(f"Target:      {', '.join(targets) if targets else 'None detected'}")
    print(f"Intent:      {intent_reps[str(il[new_node_idx].argmax().item())]}")
    print(f"Implication: {implication_reps[str(ml[new_node_idx].argmax().item())]}")
    print(f"Neighbors:   {len(neighbors)}")

if __name__ == '__main__':
    infer("Tablighi Jamaat is responsible for spreading COVID all over India.")
    infer("China created this virus in a lab to destroy the world economy.")
    infer("These immigrants are taking over our country and our jobs.")
