import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from model import GNNClassifier, compute_khop_edge_index

TAXONOMY_PATH = '/home/imone/hatemirage/taxonomy.json'
GRAPH_PATH    = '/home/imone/hatemirage/graph.pt'
MODEL_PATH    = '/home/imone/hatemirage/model.pt'
SBERT_MODEL   = 'sentence-transformers/all-mpnet-base-v2'
PHI3_MODEL    = 'microsoft/Phi-3-mini-4k-instruct'
HIDDEN_DIM    = 256
TARGET_THRESHOLD = 0.4
SIM_THRESHOLD = 0.7
TOP_K         = 5

device = torch.device('cuda')

with open(TAXONOMY_PATH) as f:
    taxonomy = json.load(f)

target_entities  = taxonomy['target_entities']
intent_reps      = taxonomy['intent_cluster_representatives']
implication_reps = taxonomy['implication_cluster_representatives']
n_targets        = len(target_entities)
n_intent         = taxonomy['n_intent_clusters']
n_implication    = taxonomy['n_implication_clusters']

print("Loading GNN...")
encoder = SentenceTransformer(SBERT_MODEL)
data    = torch.load(GRAPH_PATH, weights_only=False)

gnn = GNNClassifier(
    in_dim=data.x.shape[1],
    hidden_dim=HIDDEN_DIM,
    n_targets=n_targets,
    n_intent_clusters=n_intent,
    n_implication_clusters=n_implication,
    num_nodes=data.x.shape[0],
    edge_index_1hop=data.edge_index,
)
gnn.load_state_dict(torch.load(MODEL_PATH, map_location=device))
gnn  = gnn.to(device).eval()
data = data.to(device)

print("Loading Phi-3 in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
)
tokenizer = AutoTokenizer.from_pretrained(PHI3_MODEL)
phi3      = AutoModelForCausalLM.from_pretrained(
    PHI3_MODEL, quantization_config=bnb_config,
    device_map='cuda', trust_remote_code=True)
print("Phi-3 loaded")

def gnn_predict(comment):
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
    for k in gnn.orders:
        buf = f"edge_index_{k}"
        original_buffers[buf] = getattr(gnn, buf).clone()
        setattr(gnn, buf,
            new_edge_index if k == 1
            else compute_khop_edge_index(new_edge_index, num_nodes_new, k).to(device))

    with torch.no_grad():
        tl, il, ml = gnn(new_x, new_edge_index)

    for buf, val in original_buffers.items():
        setattr(gnn, buf, val)

    tp      = torch.sigmoid(tl[new_node_idx])
    targets = [target_entities[i] for i, p in enumerate(tp) if p > TARGET_THRESHOLD]
    return {
        'targets':          targets if targets else ['unspecified group'],
        'intent_label':     intent_reps[str(il[new_node_idx].argmax().item())],
        'implication_label': implication_reps[str(ml[new_node_idx].argmax().item())],
    }

def verbalize(comment, gnn_output):
    prompt = f"""<|user|>
You are an expert in hate speech analysis. Given a faux hate comment and its structured analysis, write concise one-sentence explanations for Intent and Implication.

Comment: "{comment}"
Target: {', '.join(gnn_output['targets'])}
Intent category: {gnn_output['intent_label']}
Implication category: {gnn_output['implication_label']}

Write exactly:
Intent: [one sentence describing the author's motive, mentioning the target]
Implication: [one sentence describing the societal impact, mentioning the target]
<|end|>
<|assistant|>
"""
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = phi3.generate(
            **inputs, max_new_tokens=100, do_sample=False,
            temperature=1.0, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

def run(comment):
    out  = gnn_predict(comment)
    text = verbalize(comment, out)
    print(f"\nComment:     {comment}")
    print(f"Target:      {', '.join(out['targets'])}")
    print(f"GNN Intent:  {out['intent_label']}")
    print(f"GNN Impl:    {out['implication_label']}")
    print(f"Phi-3:\n{text}")

run("Tablighi Jamaat is responsible for spreading COVID all over India.")
run("China created this virus in a lab to destroy the world economy.")
run("These immigrants are taking over our country and our jobs.")
