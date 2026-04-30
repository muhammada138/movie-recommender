import logging
import requests
import pandas as pd
import streamlit as st
import os

logger = logging.getLogger(__name__)

TMDB_API_KEY = os.environ.get("TMDB_API_KEY")

@st.cache_data(show_spinner=False, ttl=86400)
def _fetch_tmdb_data(endpoint, tmdb_id):
    """Internal helper for TMDB API requests."""
    if not TMDB_API_KEY or pd.isna(tmdb_id) or not tmdb_id:
        return None
    
    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}/{endpoint}?api_key={TMDB_API_KEY}"
    if not endpoint:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}"
        
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            logger.debug(f"TMDB API returned {response.status_code} for ID {tmdb_id}")
    except Exception as e:
        logger.warning(f"Error fetching from TMDB (ID: {tmdb_id}, endpoint: {endpoint}): {e}")
    return None

@st.cache_data(show_spinner=False, ttl=86400)
def get_poster_url(tmdb_id):
    """
    Fetches the poster URL for a given TMDB ID using the TMDB API.
    """
    data = _fetch_tmdb_data("", tmdb_id)
    if data:
        poster_path = data.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None

@st.cache_data(show_spinner=False, ttl=86400)
def get_movie_details(tmdb_id):
    """
    Fetches streaming providers (US) from TMDB.
    """
    providers = []
    data = _fetch_tmdb_data("watch/providers", tmdb_id)
    if data:
        us_data = data.get("results", {}).get("US", {})
        flatrate = us_data.get("flatrate", [])
        # Return max 3 providers to keep UI clean
        providers = [p.get("provider_name") for p in flatrate[:3]]

    return providers
