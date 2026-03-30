import requests
import pandas as pd
import streamlit as st
import os

TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "9676569218c06e42a88ef76b28b92d1d")

@st.cache_data(show_spinner=False, ttl=86400)
def get_poster_url(tmdb_id):
    """
    Fetches the poster URL for a given TMDB ID using the TMDB API.
    Uses st.cache_data to cache responses for 24 hours.
    """
    if pd.isna(tmdb_id) or not tmdb_id:
        return None
        
    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        print(f"Error fetching poster for TMDB ID {tmdb_id}: {e}")
    return None
