import torch
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
from model import GNNClassifier
from infer import _build_hetero_inference_graph, model, encoder, data, \
                  target_entities, intent_reps, implication_reps

DATA_PATH     = '/home/imone/hatemirage/HateMirage Sample Data.xlsx'
TAXONOMY_PATH = '/home/imone/hatemirage/taxonomy.json'
TARGET_THRESHOLD = 0.4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_excel(DATA_PATH)
torch.manual_seed(42)
idx      = torch.randperm(len(df))
test_idx = idx[80:].tolist()
test_df  = df.iloc[test_idx].reset_index(drop=True)

rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def sbert_sim(a, b):
    if not a or not b:
        return 0.0
    return util.cos_sim(encoder.encode(a, convert_to_tensor=True),
                        encoder.encode(b, convert_to_tensor=True)).item()

def rouge_l(pred, gold):
    if not pred or not gold:
        return 0.0
    return rouge.score(gold, pred)['rougeL'].fmeasure

target_sbert, target_rouge = [], []
intent_sbert,  intent_rouge  = [], []
impl_sbert,    impl_rouge    = [], []

for _, row in test_df.iterrows():
    comment   = str(row['Comments'])
    gold_t    = str(row['Target'])
    gold_i    = str(row['Intent'])
    gold_imp  = str(row['Implication'])

    emb        = encoder.encode([comment])
    emb_tensor = torch.tensor(emb, dtype=torch.float, device=device)
    new_x_dict, new_edge_dict, new_idx, _ = _build_hetero_inference_graph(emb, emb_tensor)

    with torch.no_grad():
        t_logits, i_logits, imp_logits = model(new_x_dict, new_edge_dict)

    target_probs = torch.sigmoid(t_logits[new_idx])
    pred_targets = [target_entities[i] for i, p in enumerate(target_probs) if p > TARGET_THRESHOLD]
    pred_t   = ', '.join(pred_targets)
    pred_i   = intent_reps[str(i_logits[new_idx].argmax().item())]
    pred_imp = implication_reps[str(imp_logits[new_idx].argmax().item())]

    target_sbert.append(sbert_sim(pred_t,   gold_t))
    target_rouge.append(rouge_l(pred_t,     gold_t))
    intent_sbert.append(sbert_sim(pred_i,   gold_i))
    intent_rouge.append(rouge_l(pred_i,     gold_i))
    impl_sbert.append(sbert_sim(pred_imp,   gold_imp))
    impl_rouge.append(rouge_l(pred_imp,     gold_imp))

print("\n=== Evaluation vs GPT-4 annotations (n=20) ===")
print(f"{'':15} {'SBERT':>8} {'ROUGE-L':>8}")
print(f"{'Target':15} {np.mean(target_sbert)*100:>7.2f}% {np.mean(target_rouge)*100:>7.2f}%")
print(f"{'Intent':15} {np.mean(intent_sbert)*100:>7.2f}% {np.mean(intent_rouge)*100:>7.2f}%")
print(f"{'Implication':15} {np.mean(impl_sbert)*100:>7.2f}% {np.mean(impl_rouge)*100:>7.2f}%")
print("\n(Paper best zero-shot: Target SBERT=65.55%, ROUGE=50.36%)")
print("(Paper best RAG:       Target SBERT=63.65%, ROUGE=47.81%)")
