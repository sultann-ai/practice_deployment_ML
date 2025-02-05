from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

app = FastAPI()

# Enable CORS so that frontend can access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can restrict this for security)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Hugging Face Sentiment Analysis Model
classifier = pipeline("sentiment-analysis")

@app.post("/analyze/")
async def analyze_text(text: str = Form(...)):
    result = classifier(text)[0]
    return {"sentence": text, "label": result["label"], "confidence": result["score"]}
