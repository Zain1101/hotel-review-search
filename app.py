import os
from dotenv import load_dotenv  

load_dotenv()

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import re
from search_engine.loader import load_data
from search_engine.model import predict_sentiment
from search_engine.search import search_reviews
from search_engine.image_fetcher import get_hotel_image_url

API_KEY = os.getenv("GOOGLE_API_KEY")
CSE_ID = os.getenv("GOOGLE_CSE_ID")

#page setup
st.set_page_config(page_title="Hotel Review Finder", layout="wide")

st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #4CAF50;
        margin-top: 20px;
        animation: fadeInDown 1s ease-out;
    }
    .hotel-card {
        border: 1px solid #e6e6e6;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 20px;
        background-color: #f9f9f9;
        box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.05);
        animation: fadeIn 0.8s ease-in-out;
    }
    .summary-box {
        background-color: #e8f5e9;
        border-left: 6px solid #4CAF50;
        padding: 16px;
        margin-top: 20px;
        border-radius: 8px;
        animation: slideUp 1s ease-out;
    }
    mark {
        background-color: #ffff99;
        padding: 0 4px;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

#title
st.markdown('<div class="main-title">🏨 Hotel Review Semantic Search</div>', unsafe_allow_html=True)


with st.spinner("Loading hotel review data..."):
    df = load_data()

#Sidebar
with st.sidebar:
    st.header("🔧 Filters")
    top_k = st.slider("Number of results:", min_value=1, max_value=30, value=5)

#user input
query = st.text_input("Enter your query:", "Luxury hotel in Lahore with friendly staff")

def highlight_query(text, query):
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    return pattern.sub(f"<mark>{query}</mark>", text)

if st.button("Search") and query:

    sentiment = predict_sentiment(query)
    results = search_reviews(df, query, sentiment, top_k)

    if results:
        st.subheader("📍 Top Matching Reviews")

        for row in results:
            with st.container():
                st.markdown('<div class="hotel-card">', unsafe_allow_html=True)
                cols = st.columns([1, 2])

                #Images
                image_url = get_hotel_image_url(row['Hotel Name'], row['City'], API_KEY, CSE_ID)
                if image_url:
                    cols[0].image(image_url, caption=row['Hotel Name'], use_container_width=True)
                #Results
                with cols[1]:
                    st.markdown(f"**Hotel:** {row['Hotel Name']}")
                    st.markdown(f"**City:** {row['City']}")
                    st.markdown(f"**Rating:** {row['Rating']}")
                    highlighted_review = highlight_query(row['Review Text'], query)
                    st.markdown(f"**Review:** {highlighted_review}", unsafe_allow_html=True)
                    st.markdown(f"**Google Maps Link:** [View Map]({row['Google Maps Link']})")
                    st.markdown(f"**Sentiment:** {row['Predicted Sentiment']}")
                
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("No relevant reviews found.")


