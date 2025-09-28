# 📄 DocuSummarizer

Temporarily trained on: https://huggingface.co/datasets/knkarthick/dialogsum

DocuSummarizer is a domain-specific summarization service for **finance documents**.  
It fine-tunes **Flan-T5-small** on the **[EDGAR-CORPUS](https://huggingface.co/datasets/lmqg/edgar-corpus)** dataset and serves summaries via a **FastAPI** microservice.

# Quick Start

## 1. Installation
```bash
git clone https://github.com/avionmission/DocuSummarizer
cd DocuSummarizer
pip install -r requirements.txt
```

## 2. Training (small subset for demo)
```bash
python src/train.py
```

## 3. Run API
```bash
uvicorn src.server:app --reload --port 8000
```

## 4. Test
Using curl:
```bash
curl -X POST "http://127.0.0.1:8000/summarize" \
-H "Content-Type: application/json" \
-d '{"text": "#Person1#: Hi, Mr. Smith. I'\''m Doctor Hawkins. Why are you here today?\n#Person2#: I found it would be a good idea to get a check-up.\n#Person1#: Yes, well, you haven'\''t had one for 5 years. You should have one every year.\n#Person2#: I know. I figure as long as there is nothing wrong, why go see the doctor?\n#Person1#: Well, the best way to avoid serious illnesses is to find out about them early.\n#Person2#: Ok, thanks doctor."}'
```

# Structure
```bash
client/                  # React client to test the inference server
src/
 ├─ train.py             # Fine-tune on EDGAR-CORPUS
 ├─ server.py            # FastAPI microservice
outputs/                 # Saved models
requirements.txt
README.md
```

# 🛠 Tech Stack
- Python, FastAPI
- Hugging Face Transformers, Datasets
- Flan-T5-small + LoRA fine-tuning
- EDGAR-CORPUS (finance dataset)



