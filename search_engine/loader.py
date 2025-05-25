import pandas as pd
import numpy as np

#Convert embedding into a numpy array
def parse_embedding(text):
    try:
        return np.fromstring(text.strip("[]"), sep=" ")
    except:
        return np.array([])

def load_data(path="data/hotels_reviews_with_sentiment.csv"):
    df = pd.read_csv(path)
    df['embeddings'] = df['embeddings'].apply(parse_embedding)
    return df
