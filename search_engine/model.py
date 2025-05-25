import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import pipeline
from sentence_transformers import SentenceTransformer

device = "cpu"

#Converts sentences into numerical vector embeddings
#uses all-MiniLM-L6-v2 model which is small, fast, and effective for semantic similarity
embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

#Sentiment Classifier
sentiment_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)

#convert the input text into a sentence embedding
def embed_text(text):
    return embedder.encode(text)
#this predicts sentiment of the input
def predict_sentiment(text):
    try:
        return sentiment_classifier(text[:512])[0]['label']
    except:
        return "NEUTRAL"