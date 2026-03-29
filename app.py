import streamlit as st
import pandas as pd
from recommender import recommend_for_user, ratings, movies, get_user_rating_count, recommend_similar_movies

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="centered",
)

st.title("Movie Recommender")
st.markdown(
    "Experimenting with recommendation algorithms on [MovieLens](https://grouplens.org/datasets/movielens/)."
)
st.markdown("---")

# sidebar controls
with st.sidebar:
    st.header("Settings")
    n_recs = st.slider("How many recommendations?", min_value=5, max_value=20, value=10)
    top_k = st.slider("Similar users to compare against", min_value=3, max_value=20, value=5)
    st.markdown("---")
    st.markdown("**About the models**")
    st.markdown(
        "**Collaborative Filtering:** Builds a user×movie rating matrix, computes cosine similarity between users, "
        "then weights the top-K most similar users' ratings to predict what you'd like. "
        "RMSE on test data: **0.9764**."
    )
    st.markdown(
        "**Content-Based:** Uses TF-IDF on movie genres to find movies with similar tags using cosine similarity."
    )

tab1, tab2 = st.tabs(["Recommend for User (Collaborative)", "Similar to Movie (Content-Based)"])

with tab1:
    user_ids = sorted(ratings["userId"].unique().tolist())
    user_id = st.selectbox("Select a user ID", user_ids, index=0)

    # Minimum ratings warning
    rating_count = get_user_rating_count(user_id, ratings)
    if rating_count < 15:
        st.warning(f"User {user_id} only has {rating_count} ratings. The model has very little history to work with, so recommendations may be noisy (the 'cold-start' problem).")

    with st.expander(f"What user {user_id} has rated"):
        user_ratings = (
            ratings[ratings["userId"] == user_id]
            .merge(movies, on="movieId")
            [["title", "genres", "rating"]]
            .sort_values("rating", ascending=False)
        )
        st.dataframe(user_ratings.reset_index(drop=True), use_container_width=True)

    st.markdown("---")

    if st.button("Get Recommendations", type="primary", key="collab_btn"):
        with st.spinner("Finding movies you'll like..."):
            try:
                recs = recommend_for_user(user_id, n=n_recs, top_k_users=top_k)
            except ValueError as e:
                st.error(str(e))
                st.stop()

        st.subheader(f"Top picks for user {user_id}")
        for i, rec in enumerate(recs, 1):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{i}. {rec['title']}**")
                st.caption(rec["genres"].replace("|", " · "))
            with col2:
                st.metric("Score", f"{rec['score']:.2f}")
            st.divider()

        st.caption(
            "Score is the weighted average rating from similar users — not a probability. "
            "Higher means the similar users loved it."
        )

with tab2:
    st.markdown(
        "Content-based filtering doesn't look at user ratings. Instead, it computes a TF-IDF vector of "
        "a movie's genres and finds other movies with the most similar vectors."
    )
    
    movie_titles = sorted(movies["title"].unique().tolist())
    selected_movie_title = st.selectbox("Select a movie you like", movie_titles, index=0)
    
    if st.button("Find Similar Movies", type="primary", key="content_btn"):
        with st.spinner("Finding similar semantic content..."):
            movie_id = movies[movies["title"] == selected_movie_title]["movieId"].values[0]
            try:
                sim_recs = recommend_similar_movies(movie_id, n=n_recs)
            except Exception as e:
                st.error(str(e))
                st.stop()
                
        st.subheader(f"Because you liked: {selected_movie_title}")
        for i, rec in enumerate(sim_recs, 1):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{i}. {rec['title']}**")
                st.caption(rec["genres"].replace("|", " · "))
            with col2:
                st.metric("Similarity", f"{rec['score']:.2f}")
            st.divider()
