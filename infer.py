import torch
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from model import GNNClassifier

TAXONOMY_PATH = '/home/imone/hatemirage/taxonomy.json'
GRAPH_PATH = '/home/imone/hatemirage/graph.pt'
MODEL_PATH = '/home/imone/hatemirage/model.pt'
MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
HIDDEN_DIM = 256
SIM_THRESHOLD = 0.7
TOP_K = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(TAXONOMY_PATH) as f:
    taxonomy = json.load(f)

target_entities = taxonomy['target_entities']
intent_reps = taxonomy['intent_cluster_representatives']
implication_reps = taxonomy['implication_cluster_representatives']
n_targets = len(target_entities)
n_intent = taxonomy['n_intent_clusters']
n_implication = taxonomy['n_implication_clusters']

encoder = SentenceTransformer(MODEL_NAME)
data = torch.load(GRAPH_PATH, weights_only=False)

model = GNNClassifier(
    in_dim=data.x.shape[1],
    hidden_dim=HIDDEN_DIM,
    n_targets=n_targets,
    n_intent_clusters=n_intent,
    n_implication_clusters=n_implication,
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
data = data.to(device)

def infer(comment):
    emb = encoder.encode([comment])
    emb_tensor = torch.tensor(emb, dtype=torch.float).to(device)

    existing = data.x.cpu().numpy()
    norms_exist = np.linalg.norm(existing, axis=1, keepdims=True)
    norms_new = np.linalg.norm(emb, axis=1, keepdims=True)
    sims = (existing / (norms_exist + 1e-8)) @ (emb / (norms_new + 1e-8)).T
    sims = sims.flatten()

    top_k_idx = np.argsort(sims)[::-1][:TOP_K]
    neighbors = [j for j in top_k_idx if sims[j] >= SIM_THRESHOLD]

    new_x = torch.cat([data.x, emb_tensor], dim=0)
    new_node_idx = data.x.shape[0]

    new_edges_src = [new_node_idx] * len(neighbors)
    new_edges_dst = neighbors
    if neighbors:
        extra_edges = torch.tensor([new_edges_src, new_edges_dst], dtype=torch.long).to(device)
        new_edge_index = torch.cat([data.edge_index, extra_edges], dim=1)
    else:
        new_edge_index = data.edge_index

    with torch.no_grad():
        target_logits, intent_logits, implication_logits = model(new_x, new_edge_index)

    target_probs = torch.sigmoid(target_logits[new_node_idx])
    predicted_targets = [target_entities[i] for i, p in enumerate(target_probs) if p > 0.4]

    intent_cluster = intent_logits[new_node_idx].argmax().item()
    implication_cluster = implication_logits[new_node_idx].argmax().item()

    print(f"\nComment: {comment}")
    print(f"Target:      {', '.join(predicted_targets) if predicted_targets else 'None detected'}")
    print(f"Intent:      {intent_reps[str(intent_cluster)]}")
    print(f"Implication: {implication_reps[str(implication_cluster)]}")
    print(f"Neighbors used: {len(neighbors)}")

infer("Tablighi Jamaat is responsible for spreading COVID all over India.")
infer("China created this virus in a lab to destroy the world economy.")
infer("These immigrants are taking over our country and our jobs.")
