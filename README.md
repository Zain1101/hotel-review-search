# Hotel Review Semantic Search

This is a semantic search web application built with **Streamlit**, allowing users to search hotel reviews using natural language queries. The system uses **FAISS** for fast similarity search and **transformer models** for embedding and sentiment classification.

---
##  Streamlit app 
- https://hotel-review-search-k4fxpag5mnfsd98tjyrqe3.streamlit.app/

---
##  Features

-  Semantic search using sentence-transformer embeddings  
-  Auto sentiment detection on query  
-  Fast similarity search using FAISS (L2 distance)  
-  Google Image & Maps integration (via Custom Search API)  
-  Clean and interactive Streamlit UI  

---

##  Requirements

- Python 3.10
- See [`requirements.txt`](./requirements.txt)

---

## ⚙️ Setup Instructions

1. **Clone the repository:**
2. Create a .env file in the root directory:
     GOOGLE_API_KEY=your_google_api_key
     GOOGLE_CSE_ID=your_custom_search_engine_id

folder should be like this
hotel_reviews-search/
├── app.py                   
├── search_engine/            
├── data/                   
├── faiss_index/             
├── notebooks/     
├── requirements.txt
└── .env              



```bash
git clone https://github.com/Zain1101/hotel-review-search.git
cd hotel-review-search     #bash

Optional:
bash:
python -m venv venv
source venv/bin/activate        #macOS/Linux
venv\Scripts\activate           #windows

compulsory:
pip install -r requirements.txt         #bash
streamlit run app.py                    #bash
