Key Changes in This Branch (HingBERT Encoder)
1) Encoder Replacement

Replaced SBERT (all-mpnet-base-v2) with HingBERT (l3cube-pune/hinglish-bert).

Motivation:

Better handling of code-mixed Hinglish text
Improved semantic understanding in low-resource linguistic settings
2️) Updated Encoding Pipeline

Before (SBERT):
Comment → SBERT → embedding

Now (HingBERT):
Comment → HingBERT → token embeddings → mean pooling → embedding

-> Results (HingBERT Branch)
             HingBERT scores   ROUGE-L  

Target       86.27% 34.83%
Intent       89.41% 33.72%
Implication  85.93% 25.46%
