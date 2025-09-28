import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

## 1. Setup

app =  FastAPI(
    title="Inference API",
    description="An API for running summarization model inference",
    version="1.0"
)

# Define the device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "./saved_model"

# Load the model and tokenizer from the saved directory at startup
# This is done once to avoid reloading them for every request.
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval() # Set model to evaluation mode
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    tokenizer = None
    model = None


# --- 2. Pydantic Models for Request and Response ---

class InferenceRequest(BaseModel):
    text: str
    max_new_tokens: int = 200

class InferenceResponse(BaseModel):
    summary: str


# --- 3. API Endpoint ---

@app.post("/summarize", response_model=InferenceResponse)
def summarize(request: InferenceRequest):
    """
    Takes a string of text and returns a generated summary.
    """
    if not model or not tokenizer:
        return {"error": "Model not loaded. Please check server logs."}

    # Prepare the prompt
    prompt = f"""
Summarize the following conversation.

{request.text}

Summary: """

    # Tokenize the input and move it to the correct device
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    
    # Generate the output
    with torch.no_grad(): # Disable gradient calculation for inference
        outputs = model.generate(inputs, max_new_tokens=request.max_new_tokens)
    
    # Decode the output
    summary_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return InferenceResponse(summary=summary_text)

@app.get("/")
def read_root():
    return {"status": "API is running."}