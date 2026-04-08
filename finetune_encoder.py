import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import json
from hinglish_encoder import HingBERTEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = '/home/imone/hatemirage/HateMirage Sample Data.xlsx'
LABELS_PATH = '/home/imone/hatemirage/labels.json'
SAVE_PATH = '/home/imone/hatemirage/hinglish_bert_finetuned'
EPOCHS = 5
BATCH_SIZE = 16
LR = 2e-5
TEMPERATURE = 0.07

print(f"Using device: {DEVICE}")

df = pd.read_excel(DATA_PATH)
with open(LABELS_PATH) as f:
    labels = json.load(f)

comments = df['Comments'].fillna('').tolist()
intent_cluster_ids = labels['intent_cluster_ids']

class CommentDataset(Dataset):
    def __init__(self, texts, cluster_ids, tokenizer):
        self.texts = texts
        self.labels = cluster_ids
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True,
            max_length=128, padding='max_length', return_tensors='pt'
        )
        return {k: v.squeeze(0) for k, v in enc.items()}, self.labels[idx]

def supcon_loss(embeddings, labels, temperature=TEMPERATURE):
    embeddings = F.normalize(embeddings, dim=1)
    sim = torch.mm(embeddings, embeddings.T) / temperature
    labels = torch.tensor(labels).to(embeddings.device)
    mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    mask.fill_diagonal_(0)
    exp_sim = torch.exp(sim)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    loss = -(mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
    return loss.mean()

encoder = HingBERTEncoder(device=DEVICE)
dataset = CommentDataset(comments, intent_cluster_ids, encoder.tokenizer)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
optimizer = torch.optim.AdamW(encoder.bert.parameters(), lr=LR)

print(f"Training on {len(dataset)} samples for {EPOCHS} epochs...")
encoder.bert.train()

for epoch in range(EPOCHS):
    total_loss = 0
    for batch, cluster_labels in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        embeddings = encoder(batch['input_ids'], batch['attention_mask'])
        loss = supcon_loss(embeddings, cluster_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg:.4f}")

encoder.bert.save_pretrained(SAVE_PATH)
encoder.tokenizer.save_pretrained(SAVE_PATH)
print(f"Saved fine-tuned model to {SAVE_PATH}")
