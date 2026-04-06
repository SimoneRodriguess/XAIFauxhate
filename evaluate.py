import torch
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
from model import GNNClassifier

DATA_PATH = '/home/imone/hatemirage/HateMirage Sample Data.xlsx'
TAXONOMY_PATH = '/home/imone/hatemirage/taxonomy.json'
GRAPH_PATH = '/home/imone/hatemirage/graph.pt'
MODEL_PATH = '/home/imone/hatemirage/model.pt'
SBERT_MODEL = 'sentence-transformers/all-mpnet-base-v2'
HIDDEN_DIM = 256
TARGET_THRESHOLD = 0.4
SIM_THRESHOLD = 0.7
TOP_K = 5

device = torch.device('cuda')

with open(TAXONOMY_PATH) as f:
    taxonomy = json.load(f)

target_entities = taxonomy['target_entities']
intent_reps = taxonomy['intent_cluster_representatives']
implication_reps = taxonomy['implication_cluster_representatives']
n_targets = len(target_entities)
n_intent = taxonomy['n_intent_clusters']
n_implication = taxonomy['n_implication_clusters']

encoder = SentenceTransformer(SBERT_MODEL)
data = torch.load(GRAPH_PATH, weights_only=False).to(device)
gnn = GNNClassifier(data.x.shape[1], HIDDEN_DIM, n_targets, n_intent, n_implication)
gnn.load_state_dict(torch.load(MODEL_PATH, map_location=device))
gnn = gnn.to(device).eval()

df = pd.read_excel(DATA_PATH)
torch.manual_seed(42)
idx = torch.randperm(len(df))
test_idx = idx[80:].tolist()
test_df = df.iloc[test_idx].reset_index(drop=True)

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def gnn_predict(comment):
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

    if neighbors:
        extra_edges = torch.tensor(
            [[new_node_idx] * len(neighbors), neighbors],
            dtype=torch.long).to(device)
        new_edge_index = torch.cat([data.edge_index, extra_edges], dim=1)
    else:
        new_edge_index = data.edge_index

    with torch.no_grad():
        target_logits, intent_logits, implication_logits = gnn(new_x, new_edge_index)

    target_probs = torch.sigmoid(target_logits[new_node_idx])
    predicted_targets = [target_entities[i] for i, p in enumerate(target_probs) if p > TARGET_THRESHOLD]
    intent_cluster = intent_logits[new_node_idx].argmax().item()
    implication_cluster = implication_logits[new_node_idx].argmax().item()

    return {
        'target_text': ', '.join(predicted_targets) if predicted_targets else '',
        'intent_text': intent_reps[str(intent_cluster)],
        'implication_text': implication_reps[str(implication_cluster)],
    }

target_sbert, target_rouge = [], []
intent_sbert, intent_rouge = [], []
impl_sbert, impl_rouge = [], []

for _, row in test_df.iterrows():
    comment = str(row['Comments'])
    gold_target = str(row['Target'])
    gold_intent = str(row['Intent'])
    gold_impl = str(row['Implication'])

    pred = gnn_predict(comment)

    # SBERT similarities
    def sbert_sim(a, b):
        if not a or not b:
            return 0.0
        ea = encoder.encode(a, convert_to_tensor=True)
        eb = encoder.encode(b, convert_to_tensor=True)
        return util.cos_sim(ea, eb).item()

    # ROUGE-L F1
    def rouge_l(pred_text, gold_text):
        if not pred_text or not gold_text:
            return 0.0
        return scorer.score(gold_text, pred_text)['rougeL'].fmeasure

    target_sbert.append(sbert_sim(pred['target_text'], gold_target))
    target_rouge.append(rouge_l(pred['target_text'], gold_target))

    intent_sbert.append(sbert_sim(pred['intent_text'], gold_intent))
    intent_rouge.append(rouge_l(pred['intent_text'], gold_intent))

    impl_sbert.append(sbert_sim(pred['implication_text'], gold_impl))
    impl_rouge.append(rouge_l(pred['implication_text'], gold_impl))

print("\n=== Evaluation vs GPT-4 annotations (n=20) ===")
print(f"{'':15} {'SBERT':>8} {'ROUGE-L':>8}")
print(f"{'Target':15} {np.mean(target_sbert)*100:>7.2f}% {np.mean(target_rouge)*100:>7.2f}%")
print(f"{'Intent':15} {np.mean(intent_sbert)*100:>7.2f}% {np.mean(intent_rouge)*100:>7.2f}%")
print(f"{'Implication':15} {np.mean(impl_sbert)*100:>7.2f}% {np.mean(impl_rouge)*100:>7.2f}%")
print("\n(Paper's best zero-shot: Target SBERT=65.55%, ROUGE=50.36%)")
print("(Paper's best RAG:       Target SBERT=63.65%, ROUGE=47.81%)")
