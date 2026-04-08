import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class HingBERTEncoder(nn.Module):
    def __init__(self, model_name="l3cube-pune/hinglish-bert", device=None):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.device = device
        self.bert.to(device)

    def mean_pool(self, token_embeddings, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)

    def encode(self, texts, batch_size=32, show_progress_bar=False):
        """Drop-in replacement for SentenceTransformer.encode() — returns numpy array"""
        self.bert.eval()
        all_embeddings = []
        if show_progress_bar:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(texts), batch_size), desc="Encoding")
        else:
            iterator = range(0, len(texts), batch_size)
        for i in iterator:
            batch = texts[i:i+batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                output = self.bert(**encoded)
            embeddings = self.mean_pool(output.last_hidden_state, encoded["attention_mask"])
            all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0).numpy()

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.mean_pool(output.last_hidden_state, attention_mask)
