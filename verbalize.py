import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from model import GNNClassifier

TAXONOMY_PATH = '/home/imone/hatemirage/taxonomy.json'
GRAPH_PATH = '/home/imone/hatemirage/graph.pt'
MODEL_PATH = '/home/imone/hatemirage/model.pt'
SBERT_MODEL = 'sentence-transformers/all-mpnet-base-v2'
PHI3_MODEL = 'microsoft/Phi-3-mini-4k-instruct'
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

print("Loading GNN...")
encoder = SentenceTransformer(SBERT_MODEL)
data = torch.load(GRAPH_PATH, weights_only=False).to(device)
gnn = GNNClassifier(data.x.shape[1], HIDDEN_DIM, n_targets, n_intent, n_implication)
gnn.load_state_dict(torch.load(MODEL_PATH, map_location=device))
gnn = gnn.to(device).eval()

print("Loading Phi-3 in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
)
tokenizer = AutoTokenizer.from_pretrained(PHI3_MODEL)
phi3 = AutoModelForCausalLM.from_pretrained(
    PHI3_MODEL,
    quantization_config=bnb_config,
    device_map='cuda',
    trust_remote_code=True,
)
print("Phi-3 loaded")

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
        'targets': predicted_targets if predicted_targets else ['unspecified group'],
        'intent_label': intent_reps[str(intent_cluster)],
        'implication_label': implication_reps[str(implication_cluster)],
    }

def verbalize(comment, gnn_output):
    targets = ', '.join(gnn_output['targets'])
    intent_label = gnn_output['intent_label']
    implication_label = gnn_output['implication_label']

    prompt = f"""<|user|>
You are an expert in hate speech analysis. Given a faux hate comment and its structured analysis, write concise one-sentence explanations for Intent and Implication.

Comment: "{comment}"
Target: {targets}
Intent category: {intent_label}
Implication category: {implication_label}

Write exactly:
Intent: [one sentence describing the author's motive, mentioning the target]
Implication: [one sentence describing the societal impact, mentioning the target]
<|end|>
<|assistant|>
"""

    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = phi3.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated.strip()

def run(comment):
    gnn_output = gnn_predict(comment)
    text = verbalize(comment, gnn_output)
    print(f"\nComment:     {comment}")
    print(f"Target:      {', '.join(gnn_output['targets'])}")
    print(f"GNN Intent:  {gnn_output['intent_label']}")
    print(f"GNN Impl:    {gnn_output['implication_label']}")
    print(f"Phi-3 output:\n{text}")

run("Tablighi Jamaat is responsible for spreading COVID all over India.")
run("China created this virus in a lab to destroy the world economy.")
run("These immigrants are taking over our country and our jobs.")
