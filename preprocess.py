import pandas as pd
import numpy as np
from tqdm import tqdm
import faiss
import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline

df = pd.read_csv("data/hotels_reviews_cleaned.csv")
df['Cleaned Review Text'] = df['Cleaned Review Text'].fillna("").astype(str)

# Running sentence transformer for semantic analysis
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
def get_sentiment(text):
    try:
        return classifier(text[:512])[0]['label']
    except:
        return "NEUTRAL"

tqdm.pandas()
df['Predicted Sentiment'] = df['Cleaned Review Text'].progress_apply(get_sentiment)

# generating embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')
df['embeddings'] = df['Cleaned Review Text'].progress_apply(lambda x: embedder.encode(x))

df.to_csv("data/hotels_reviews_with_sentiment.csv", index=False)

# Building and saving the FAISS index
embedding_matrix = np.vstack(df['embeddings'].values).astype('float32')
index = faiss.IndexFlatL2(embedding_matrix.shape[1])
index.add(embedding_matrix)

os.makedirs("faiss_index", exist_ok=True)
faiss.write_index(index, "faiss_index/index.bin")

print("Preprocessing complete. Sentiment CSV and FAISS index saved.")
