import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_data(ratings_path="data/ratings.csv", movies_path="data/movies.csv", links_path="data/links.csv"):
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    links = pd.read_csv(links_path)
    movies = movies.merge(links[['movieId', 'tmdbId']], on='movieId', how='left')
    return ratings, movies


def build_user_item_matrix(ratings):
    matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)
    return matrix


def get_user_rating_count(user_id, ratings):
    return len(ratings[ratings["userId"] == user_id])


def build_similarity_matrix(matrix):
    sim = cosine_similarity(matrix)
    return pd.DataFrame(sim, index=matrix.index, columns=matrix.index)


def get_recommendations(user_id, matrix, similarity_df, n=10, top_k_users=5):
    if user_id not in matrix.index:
        raise ValueError(f"User {user_id} not found in the dataset")

    # grab the top-k most similar users (skip index 0 which is the user themselves)
    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:top_k_users + 1]

    # weighted average of what those users rated
    similar_users_ratings = matrix.loc[similar_users.index]
    weighted = similar_users_ratings.T.dot(similar_users) / similar_users.sum()

    # filter out movies the user has already seen
    already_seen = matrix.loc[user_id][matrix.loc[user_id] > 0].index
    recs = (
        weighted
        .drop(already_seen, errors="ignore")
        .sort_values(ascending=False)
        .head(n)
    )
    return recs


# load everything once at module level so the app doesn't re-read CSVs on every call
ratings, movies = load_data()
matrix = build_user_item_matrix(ratings)
similarity_df = build_similarity_matrix(matrix)

# build content-based similarity matrix globally
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

movies['genres_str'] = movies['genres'].str.replace('|', ' ')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres_str'])
genre_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


def recommend_similar_movies(movie_ids, n=10):
    if not isinstance(movie_ids, list):
        movie_ids = [movie_ids]
        
    valid_ids = [m for m in movie_ids if m in movies["movieId"].values]
    if not valid_ids:
        raise ValueError("Selected movies not found in the dataset")
        
    indices = movies.index[movies["movieId"].isin(valid_ids)].tolist()
    
    # Calculate average genre similarity vector for the selected movies
    avg_sim = genre_sim[indices].mean(axis=0)
    
    sim_scores = list(enumerate(avg_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Skip the target movies themselves
    sim_scores = [x for x in sim_scores if x[0] not in indices]
    sim_scores = sim_scores[:n]
    
    result = []
    for i, score in sim_scores:
        result.append({
            "movieId": movies.iloc[i]["movieId"],
            "title": movies.iloc[i]["title"],
            "genres": movies.iloc[i]["genres"],
            "tmdbId": movies.iloc[i]["tmdbId"],
            "score": round(float(score), 3),
        })
    return result


def recommend_for_user(user_id, n=10, top_k_users=5):
    recs = get_recommendations(user_id, matrix, similarity_df, n=n, top_k_users=top_k_users)
    result = []
    for movie_id, score in recs.items():
        title_row = movies[movies["movieId"] == movie_id]["title"].values
        genres_row = movies[movies["movieId"] == movie_id]["genres"].values
        tmdb_row = movies[movies["movieId"] == movie_id]["tmdbId"].values
        if len(title_row) > 0:
            result.append({
                "movieId": movie_id,
                "title": title_row[0],
                "genres": genres_row[0] if len(genres_row) > 0 else "",
                "tmdbId": tmdb_row[0] if len(tmdb_row) > 0 else None,
                "score": round(float(score), 3),
            })
    return result
