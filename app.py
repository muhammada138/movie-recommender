import streamlit as st
import pandas as pd
from recommender import recommend_for_user, ratings, movies

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="centered",
)

st.title("Movie Recommender")
st.markdown(
    "Collaborative filtering on [MovieLens](https://grouplens.org/datasets/movielens/) "
    "— finds users with similar taste and surfaces movies you haven't seen yet."
)
st.markdown("---")

# sidebar controls
with st.sidebar:
    st.header("Settings")
    n_recs = st.slider("How many recommendations?", min_value=5, max_value=20, value=10)
    top_k = st.slider("Similar users to compare against", min_value=3, max_value=20, value=5)
    st.markdown("---")
    st.markdown("**About the model**")
    st.markdown(
        "Builds a user×movie rating matrix, computes cosine similarity between users, "
        "then weights the top-K most similar users' ratings to predict what you'd like. "
        "RMSE on held-out test data: **0.9764**."
    )

# user lookup
user_ids = sorted(ratings["userId"].unique().tolist())
user_id = st.selectbox("Select a user ID", user_ids, index=0)

# show what this user has already rated — useful sanity check
with st.expander(f"What user {user_id} has rated"):
    user_ratings = (
        ratings[ratings["userId"] == user_id]
        .merge(movies, on="movieId")
        [["title", "genres", "rating"]]
        .sort_values("rating", ascending=False)
    )
    st.dataframe(user_ratings.reset_index(drop=True), use_container_width=True)

st.markdown("---")

if st.button("Get Recommendations", type="primary"):
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
