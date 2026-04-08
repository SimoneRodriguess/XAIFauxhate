import pandas as pd
import json
import numpy as np
from hinglish_encoder import HingBERTEncoder
from sklearn.cluster import KMeans
from collections import Counter

DATA_PATH = '/home/imone/hatemirage/HateMirage Sample Data.xlsx'
OUTPUT_DIR = '/home/imone/hatemirage/'
N_INTENT_CLUSTERS = 8
N_IMPLICATION_CLUSTERS = 8
MODEL_NAME = '/home/imone/hatemirage/hinglish_bert_finetuned'

NORMALIZE = {
    'Tablighi': 'Tablighi Jamaat',
    'Tablighi Saad': 'Tablighi Jamaat',
    'Jamaat': 'Tablighi Jamaat',
    'Congregation': 'Tablighi Jamaat',
    'CCP (Chinese Communist Party)': 'CCP',
    'Arvind Kejriwal': 'Kejriwal',
    'Muslim League': 'Muslims',
    'Islamists': 'Islam',
    'Love_jihad': 'Muslims',
    'Muslim_girls': 'Muslims',
    "Farmers' protest": 'Farmers',
    'Delhi government': 'Government',
    'DChaurasia2312': None,
    'RajatSharmaLive': None,
}

df = pd.read_excel(DATA_PATH)
print(f"Loaded {len(df)} rows")

all_targets = []
for t in df['Target'].dropna():
    entities = [e.strip() for e in t.split(',')]
    for e in entities:
        normalized = NORMALIZE.get(e, e)
        if normalized is not None:
            all_targets.append(normalized)

target_counts = Counter(all_targets)
target_entities = sorted(target_counts.keys())
target_to_idx = {t: i for i, t in enumerate(target_entities)}

print(f"Unique target entities after normalization: {len(target_entities)}")
print(f"Entities: {target_entities}")

target_labels = []
for t in df['Target'].fillna(''):
    raw = [e.strip() for e in t.split(',')]
    normalized = [NORMALIZE.get(e, e) for e in raw]
    normalized = [e for e in normalized if e is not None and e in target_to_idx]
    vec = [0] * len(target_entities)
    for e in normalized:
        vec[target_to_idx[e]] = 1
    target_labels.append(vec)

model = HingBERTEncoder(model_name=MODEL_NAME)

intent_texts = df['Intent'].fillna('').tolist()
implication_texts = df['Implication'].fillna('').tolist()

print("Encoding intent texts...")
intent_embeddings = model.encode(intent_texts, show_progress_bar=True)
print("Encoding implication texts...")
implication_embeddings = model.encode(implication_texts, show_progress_bar=True)

intent_kmeans = KMeans(n_clusters=N_INTENT_CLUSTERS, random_state=42, n_init=10)
intent_cluster_ids = intent_kmeans.fit_predict(intent_embeddings)

implication_kmeans = KMeans(n_clusters=N_IMPLICATION_CLUSTERS, random_state=42, n_init=10)
implication_cluster_ids = implication_kmeans.fit_predict(implication_embeddings)

def get_representatives(texts, embeddings, cluster_ids, n_clusters):
    representatives = {}
    for c in range(n_clusters):
        indices = [i for i, cid in enumerate(cluster_ids) if cid == c]
        cluster_embeddings = embeddings[indices]
        cluster_texts = [texts[i] for i in indices]
        centroid = cluster_embeddings.mean(axis=0)
        dists = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        representatives[c] = cluster_texts[np.argmin(dists)]
    return representatives

intent_reps = get_representatives(intent_texts, intent_embeddings, intent_cluster_ids, N_INTENT_CLUSTERS)
implication_reps = get_representatives(implication_texts, implication_embeddings, implication_cluster_ids, N_IMPLICATION_CLUSTERS)

print("\nIntent cluster representatives:")
for k, v in intent_reps.items():
    print(f"  [{k}] {v}")

print("\nImplication cluster representatives:")
for k, v in implication_reps.items():
    print(f"  [{k}] {v}")

taxonomy = {
    'target_entities': target_entities,
    'target_to_idx': target_to_idx,
    'n_intent_clusters': N_INTENT_CLUSTERS,
    'n_implication_clusters': N_IMPLICATION_CLUSTERS,
    'intent_cluster_representatives': {str(k): v for k, v in intent_reps.items()},
    'implication_cluster_representatives': {str(k): v for k, v in implication_reps.items()},
    'normalize_map': {k: v for k, v in NORMALIZE.items() if v is not None},
    'drop_entities': [k for k, v in NORMALIZE.items() if v is None],
}

labels = {
    'target_labels': target_labels,
    'intent_cluster_ids': intent_cluster_ids.tolist(),
    'implication_cluster_ids': implication_cluster_ids.tolist(),
}

with open(OUTPUT_DIR + 'taxonomy.json', 'w') as f:
    json.dump(taxonomy, f, indent=2)

with open(OUTPUT_DIR + 'labels.json', 'w') as f:
    json.dump(labels, f, indent=2)

print(f"\nFinal entity count: {len(target_entities)}")
print("Saved taxonomy.json and labels.json")
