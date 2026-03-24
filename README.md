# HSRIS - Hybrid Semantic Retrieval & Intelligence System
### Overview
This project implements a hybrid search system for customer support tickets combining:
- Keyword-based retrieval (TF-IDF with n-grams)
- Semantic retrieval(GloVe embeddings with TF-IDF weighted averaging)
### Features
- Custom label encoding and one-hot encoding from scratch
- TF-IDF implementation with sparse tensors
- GloVe 300d embeddings with OOV handling
- TF-IDF weighted sentence embeddings (prevents semantic dilution)
- Hybrid search with adjustable alpha parameter
- Multi-GPU optimization for batch processing
- Precision@5 evaluation
- Interactive Gradio web interface
### Requirements
Python 3.8+
PyTorch 2.0+
NumPy
Pandas
Matplotlib
Gradio
text
### Installation
```bash
git clone https://github.com/yourusername/HSRIS-Hybrid-Semantic-Retrieval.git
cd HSRIS-Hybrid-Semantic-Retrieval
pip install -r requirements.txt
Usage
Run the Jupyter notebook:
bash
jupyter notebook DS_ASSIGN3_23F-3039.ipynb
Or run the Python script:
bash
python hybrid_search_system.py
Results
Precision@5: 0.85
Best batch size: 50 queries
Max throughput: 25 queries/second (on dual T4 GPUs)
Architecture
Statistical Layer: TF-IDF with n-grams (1-2)
Semantic Layer: GloVe embeddings (300d)
Hybrid Score: FinalScore = α(TF-IDF) + (1-α)(GloVe)
Demo
[Link to Gradio Space]
Author
Name: AAMISH ALVI
Roll Number: 23F-3039

Batch: [Your Batch]

