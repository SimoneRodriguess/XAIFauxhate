# HateMirage: Explainable Faux Hate Detection

HateMirage is an explainable NLP system for detecting **Faux Hate**: a subtle form of hate speech where harmful intent is embedded in misleading or fabricated narratives. It follows the HateMirage framework by decomposing each comment into three dimensions: **Target**, **Intent**, and **Implication**.

---

## 🧠 Pipeline

**Input:** comment `c`

1. **Encoding**  
   SBERT (`all-mpnet-base-v2`) → embedding `h_c`

2. **Retrieval**  
   - Similar comments → neighbor nodes (with labels)  
   - Claim documents → claim nodes  
   - Target word → topic node  

3. **Graph + GNN**  
   - Heterogeneous graph (comment, topic, claim)  
   - 2-layer GNN → enriched representation `h_c_enriched`

4. **Cascaded Prediction (Key Idea)**  
   - Target ← `h_c_enriched`  
   - Intent ← `h_c_enriched + Target`  
   - Implication ← `h_c_enriched + Target + Intent`  

→ GPT-4 generates Target, Intent, and Implication from the same comment in isolation, three separate prompts with zero conditioning between them. It uses independent generation, whereas this **cascade ensures Implication is grounded (knows who + why)**.

---

## 🏗 Code Structure

- `graph_builder.py` → builds graph from embeddings  
- `model.py` → GNN + 3 heads  
- `train.py` → training  
- `infer.py` → inference pipeline  
- `evaluate.py` → SBERT + ROUGE evaluation  
- `taxonomy.py` → label clustering / normalization  

---

## 📊 Results (Current)

- Trained on: **80 samples**  
- Tested on: **20 samples**  
- Data: GPT-generated annotations  

- Target SBERT: **72.40%**  
- Intent SBERT: **70.18%**  
- Implication SBERT: **70.83%**

---

## ⚠️ Note

- Trained on synthetic (GPT-generated) data  
- Real evaluation pending on **human-annotated dataset**

---

## 💡 Core Idea

Structured reasoning first (Target → Intent → Implication),  
instead of generating explanations independently → **more grounded, explainable outputs**.
