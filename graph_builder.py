import pandas as pd
import json
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer

DATA_PATH = '/home/imone/hatemirage/HateMirage Sample Data.xlsx'
TAXONOMY_PATH = '/home/imone/hatemirage/taxonomy.json'
LABELS_PATH = '/home/imone/hatemirage/labels.json'
OUTPUT_PATH = '/home/imone/hatemirage/graph.pt'
MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
SIM_THRESHOLD = 0.7
TOP_K = 5

df = pd.read_excel(DATA_PATH)
with open(TAXONOMY_PATH) as f:
    taxonomy = json.load(f)
with open(LABELS_PATH) as f:
    labels = json.load(f)

sbert = SentenceTransformer(MODEL_NAME)

# --- comment nodes ---
print("Encoding comments...")
comment_emb = sbert.encode(df['Comments'].fillna('').tolist(), show_progress_bar=True)

# --- topic nodes: 8 intent centroids + 8 implication centroids = [16, 768] ---
intent_centroids = np.array(taxonomy['intent_centroids'])       # [8, 768]
impl_centroids = np.array(taxonomy['implication_centroids'])    # [8, 768]
topic_emb = np.vstack([intent_centroids, impl_centroids])       # [16, 768]

# --- claim nodes: SBERT of entity name strings = [34, 768] ---
entity_names = taxonomy['target_entities']
claim_emb = sbert.encode(entity_names, show_progress_bar=True)

# --- (comment, sim, comment) edges ---
norms = np.linalg.norm(comment_emb, axis=1, keepdims=True)
normalized = comment_emb / (norms + 1e-8)
sim_matrix = normalized @ normalized.T

src, dst = [], []
for i in range(len(df)):
    sims = sim_matrix[i].copy()
    sims[i] = -1
    top_k = np.argsort(sims)[::-1][:TOP_K]
    for j in top_k:
        if sims[j] >= SIM_THRESHOLD:
            src.append(i); dst.append(j)
sim_edges = torch.tensor([src, dst], dtype=torch.long)

# --- (topic, about_rev, comment) edges ---
# intent cluster id → topic node index (0-7)
# implication cluster id → topic node index (8-15)
intent_ids = labels['intent_cluster_ids']
impl_ids = labels['implication_cluster_ids']
about_src, about_dst = [], []
for i in range(len(df)):
    about_src.append(intent_ids[i])     # topic node
    about_dst.append(i)                 # comment node
    about_src.append(impl_ids[i] + 8)  # topic node (offset)
    about_dst.append(i)
about_edges = torch.tensor([about_src, about_dst], dtype=torch.long)

# --- (claim, targets_rev, comment) edges ---
target_labels = np.array(labels['target_labels'])  # [N, 34]
claim_src, claim_dst = [], []
for i in range(len(df)):
    for j in range(len(entity_names)):
        if target_labels[i][j] == 1:
            claim_src.append(j)   # claim node
            claim_dst.append(i)   # comment node
claim_edges = torch.tensor([claim_src, claim_dst], dtype=torch.long)

# --- build HeteroData ---
data = HeteroData()
data['comment'].x = torch.tensor(comment_emb, dtype=torch.float)
data['topic'].x   = torch.tensor(topic_emb, dtype=torch.float)
data['claim'].x   = torch.tensor(claim_emb, dtype=torch.float)

data['comment', 'sim', 'comment'].edge_index       = sim_edges
data['topic', 'about_rev', 'comment'].edge_index   = about_edges
data['claim', 'targets_rev', 'comment'].edge_index = claim_edges

data['comment'].target_y      = torch.tensor(target_labels, dtype=torch.float)
data['comment'].intent_y      = torch.tensor(intent_ids, dtype=torch.long)
data['comment'].implication_y = torch.tensor(impl_ids, dtype=torch.long)

print(f"comment nodes:   {data['comment'].x.shape}")
print(f"topic nodes:     {data['topic'].x.shape}")
print(f"claim nodes:     {data['claim'].x.shape}")
print(f"sim edges:       {sim_edges.shape[1]}")
print(f"about_rev edges: {about_edges.shape[1]}")
print(f"targets_rev edges: {claim_edges.shape[1]}")
torch.save(data, OUTPUT_PATH)
print(f"Saved to {OUTPUT_PATH}")
