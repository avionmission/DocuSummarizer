# 📄 DocuSummarizer

DocuSummarizer is a domain-specific summarization service for **finance documents**.  
It fine-tunes **Flan-T5-small** on the **[EDGAR-CORPUS](https://huggingface.co/datasets/lmqg/edgar-corpus)** dataset and serves summaries via a **FastAPI** microservice.

# Quick Start

## 1. Installation
```bash
git clone https://github.com/avionmission/DocuSummarizer
cd DocuSummarizer
pip install -r requirements.txt
```

## 2, Training (small subset for demo)
```bash
python src/train.py
```

## 3. Run API
```bash
uvicorn src.inference_server:app --reload --port 8000
```

## 4. Test
Using curl:
```bash
curl -X POST "http://127.0.0.1:8000/summarize" \
-H "Content-Type: application/json" \
-d '{"text":"The company reported a 12% increase in Q4 revenue driven by cloud services."}'
```

# Structure
```bash
client/                  # React client to test the inference server
src/
 ├─ train.py             # Fine-tune on EDGAR-CORPUS
 ├─ inference_server.py  # FastAPI microservice
outputs/                 # Saved models
requirements.txt
README.md
```

# 🛠 Tech Stack
- Python, FastAPI
- Hugging Face Transformers, Datasets
- Flan-T5-small + LoRA fine-tuning
- EDGAR-CORPUS (finance dataset)



