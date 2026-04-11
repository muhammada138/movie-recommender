import logging
import requests
import pandas as pd
import streamlit as st
import os

logger = logging.getLogger(__name__)

TMDB_API_KEY = os.environ.get("TMDB_API_KEY")

@st.cache_data(show_spinner=False, ttl=86400)
def get_poster_url(tmdb_id):
    """
    Fetches the poster URL for a given TMDB ID using the TMDB API.
    Uses st.cache_data to cache responses for 24 hours.
    """
    if not TMDB_API_KEY:
        return None
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
        logger.warning("Error fetching poster for TMDB ID %s: %s", tmdb_id, e)
    return None

@st.cache_data(show_spinner=False, ttl=86400)
def get_movie_details(tmdb_id):
    """
    Fetches streaming providers (US) from TMDB.
    """
    providers = []
    if not TMDB_API_KEY:
        return providers
    if pd.isna(tmdb_id) or not tmdb_id:
        return providers

    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}/watch/providers?api_key={TMDB_API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            us_data = data.get("results", {}).get("US", {})
            flatrate = us_data.get("flatrate", [])
            # Return max 3 providers to keep UI clean
            providers = [p.get("provider_name") for p in flatrate[:3]]
    except Exception as e:
        logger.warning("Error fetching providers for TMDB ID %s: %s", tmdb_id, e)

    return providers
