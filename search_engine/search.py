import numpy as np
import faiss
from .model import embed_text

def search_reviews(df, query, sentiment_filter="POSITIVE", top_k=100, similarity_threshold=0.6):
    """
    it will Perform semantic search using faiss with L2 eucledian distance.
    Its parameters are:
        Dataset
        query: User input.
        sentiment_filter: Filter by sentiment label
        top_k : Maximum number of results to return.
        similarity_threshold : Distance threshold (lower is more similar).
    Tt returns:
        results: List of relevant review rows.
    """
    #Check if query contains any known city
    query_lower = query.lower()
    city_list = df['City'].dropna().astype(str).str.lower().unique()
    
    matched_city = None
    for city in city_list:
        if city in query_lower:
            matched_city = city
            break

    #filter dataset by city if detected
    if matched_city:
        df = df[df['City'].str.lower() == matched_city]

    #build FAISS index on review vectors
    review_vectors = np.vstack(df['embeddings'].values).astype('float32')
    dim = review_vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    #add all vectors to the index
    index.add(review_vectors)

    #Embed the query. This will convert the user query into the same type of vector 
    query_vector = embed_text(query).astype('float32').reshape(1, -1)

    # retrieve top k closest review vectors by L2 distance
    distances, indices = index.search(query_vector, len(review_vectors))

    #collect and return results with sentiment filter
    results = []
    for idx in indices[0]:
        row = df.iloc[idx]
        if row['Predicted Sentiment'] == sentiment_filter:
            results.append(row)
        if len(results) >= top_k:
            break

    return results
