import pandas as pd
import json
import numpy as np
import torch
from torch_geometric.data import Data
from hinglish_encoder import HingBERTEncoder

DATA_PATH = '/home/imone/hatemirage/HateMirage Sample Data.xlsx'
TAXONOMY_PATH = '/home/imone/hatemirage/taxonomy.json'
LABELS_PATH = '/home/imone/hatemirage/labels.json'
OUTPUT_PATH = '/home/imone/hatemirage/graph.pt'
FINETUNED_PATH = '/home/imone/hatemirage/hinglish_bert_finetuned'

SIM_THRESHOLD = 0.7
TOP_K = 5

df = pd.read_excel(DATA_PATH)

with open(TAXONOMY_PATH) as f:
    taxonomy = json.load(f)
with open(LABELS_PATH) as f:
    labels = json.load(f)

model = HingBERTEncoder(model_name=FINETUNED_PATH)

print("Encoding comments...")
comment_embeddings = model.encode(df['Comments'].fillna('').tolist(), show_progress_bar=True)

x = torch.tensor(comment_embeddings, dtype=torch.float)

print("Building edges...")
norms = np.linalg.norm(comment_embeddings, axis=1, keepdims=True)
normalized = comment_embeddings / (norms + 1e-8)
sim_matrix = normalized @ normalized.T

edge_src, edge_dst = [], []
n = len(df)
for i in range(n):
    sims = sim_matrix[i].copy()
    sims[i] = -1
    top_k_idx = np.argsort(sims)[::-1][:TOP_K]
    for j in top_k_idx:
        if sims[j] >= SIM_THRESHOLD:
            edge_src.append(i)
            edge_dst.append(j)

edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
print(f"Edges created: {edge_index.shape[1]}")

target_labels = torch.tensor(labels['target_labels'], dtype=torch.float)
intent_labels = torch.tensor(labels['intent_cluster_ids'], dtype=torch.long)
implication_labels = torch.tensor(labels['implication_cluster_ids'], dtype=torch.long)

data = Data(
    x=x,
    edge_index=edge_index,
    target_y=target_labels,
    intent_y=intent_labels,
    implication_y=implication_labels,
)

print(f"Node features: {data.x.shape}")
print(f"Edge index: {data.edge_index.shape}")
print(f"Target labels: {data.target_y.shape}")
print(f"Intent labels: {data.intent_y.shape}")
print(f"Implication labels: {data.implication_y.shape}")

torch.save(data, OUTPUT_PATH)
print(f"Saved graph to {OUTPUT_PATH}")
