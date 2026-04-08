import torch
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from model import GNNClassifier

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
data    = torch.load(GRAPH_PATH, weights_only=False).to(device)

model = GNNClassifier(hidden_dim=HIDDEN_DIM, n_targets=n_targets,
                      n_intent=n_intent, n_impl=n_implication).to(device)
with torch.no_grad():
    model(data.x_dict, data.edge_index_dict)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def _build_hetero_inference_graph(emb, emb_tensor):
    new_idx = data['comment'].x.shape[0]

    # sim edges
    existing = data['comment'].x.cpu().numpy()
    norms_e  = np.linalg.norm(existing, axis=1, keepdims=True)
    norms_n  = np.linalg.norm(emb, axis=1, keepdims=True)
    sims     = ((existing / (norms_e + 1e-8)) @ (emb / (norms_n + 1e-8)).T).flatten()
    top_k    = np.argsort(sims)[::-1][:TOP_K]
    neighbors = [int(j) for j in top_k if sims[j] >= SIM_THRESHOLD]

    sim_base = data['comment', 'sim', 'comment'].edge_index
    if neighbors:
        extra = torch.tensor([[new_idx]*len(neighbors), neighbors],
                             dtype=torch.long, device=device)
        new_sim = torch.cat([sim_base, extra], dim=1)
    else:
        new_sim = sim_base

    # about_rev edges: nearest centroid lookup
    topic_np = data['topic'].x.cpu().numpy()  # [16, 768]
    intent_dists = np.linalg.norm(topic_np[:8] - emb[0], axis=1)
    impl_dists   = np.linalg.norm(topic_np[8:] - emb[0], axis=1)
    intent_t = int(np.argmin(intent_dists))
    impl_t   = int(np.argmin(impl_dists)) + 8
    about_base = data['topic', 'about_rev', 'comment'].edge_index
    extra_about = torch.tensor([[intent_t, impl_t], [new_idx, new_idx]],
                               dtype=torch.long, device=device)
    new_about = torch.cat([about_base, extra_about], dim=1)

    new_x_dict = {
        'comment': torch.cat([data['comment'].x, emb_tensor], dim=0),
        'topic':   data['topic'].x,
        'claim':   data['claim'].x,
    }
    new_edge_dict = {
        ('comment', 'sim', 'comment'):       new_sim,
        ('topic', 'about_rev', 'comment'):   new_about,
        ('claim', 'targets_rev', 'comment'): data['claim', 'targets_rev', 'comment'].edge_index,
    }
    return new_x_dict, new_edge_dict, new_idx, len(neighbors)

def infer(comment):
    emb        = encoder.encode([comment])
    emb_tensor = torch.tensor(emb, dtype=torch.float, device=device)

    new_x_dict, new_edge_dict, new_idx, n_neighbors = \
        _build_hetero_inference_graph(emb, emb_tensor)

    with torch.no_grad():
        t_logits, i_logits, imp_logits = model(new_x_dict, new_edge_dict)

    target_probs = torch.sigmoid(t_logits[new_idx])
    predicted_targets   = [target_entities[i] for i, p in enumerate(target_probs) if p > 0.4]
    intent_cluster      = i_logits[new_idx].argmax().item()
    implication_cluster = imp_logits[new_idx].argmax().item()

    print(f"\nComment:     {comment}")
    print(f"Target:      {', '.join(predicted_targets) if predicted_targets else 'None detected'}")
    print(f"Intent:      {intent_reps[str(intent_cluster)]}")
    print(f"Implication: {implication_reps[str(implication_cluster)]}")
    print(f"Neighbors used: {n_neighbors}")
    return {
        'target_text':      ', '.join(predicted_targets),
        'intent_text':      intent_reps[str(intent_cluster)],
        'implication_text': implication_reps[str(implication_cluster)],
    }

if __name__ == '__main__':
    infer("Tablighi Jamaat is responsible for spreading COVID all over India.")
    infer("China created this virus in a lab to destroy the world economy.")
    infer("These immigrants are taking over our country and our jobs.")
