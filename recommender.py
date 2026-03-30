import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer


def format_title(title):
    match = re.search(r'^(.*?)(,\s*(The|A|An))((\s*\(.*?\))*)$', title)
    if match:
        base = match.group(1)
        article = match.group(3)
        rest = match.group(4)
        return f"{article} {base}{rest}".strip()
    return title

def load_data(ratings_path="data/ratings.csv", movies_path="data/movies.csv", links_path="data/links.csv", tags_path="data/tags.csv"):
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    links = pd.read_csv(links_path)
    
    try:
        tags = pd.read_csv(tags_path)
        tags['tag'] = tags['tag'].fillna('').astype(str)
        # Lowercase tags for consistency
        tags['tag'] = tags['tag'].str.lower()
        grouped_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
        movies = movies.merge(grouped_tags, on='movieId', how='left')
        movies['tag'] = movies['tag'].fillna('')
    except Exception:
        movies['tag'] = ''

    # IMDb ids might come as ints without leading zeros. TMDB formats them with 'tt' manually or we just pass the raw value.
    if 'imdbId' in links.columns:
        links['imdbId'] = links['imdbId'].apply(lambda x: str(x).zfill(7) if pd.notna(x) else x)
    movies = movies.merge(links[['movieId', 'tmdbId', 'imdbId']], on='movieId', how='left')
    
    # Extract year for time-based contextual penalization
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
    movies['year'] = movies['year'].fillna(movies['year'].median())
    
    movies['title'] = movies['title'].apply(format_title)
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

movies['genres_str'] = movies['genres'].str.replace('|', ' ').str.lower()

# Vectorize genres and tags separately to avoid dense tags destroying genre L2 norm weights
gen_tfidf = TfidfVectorizer(stop_words='english')
gen_mat = gen_tfidf.fit_transform(movies['genres_str'])

tag_tfidf = TfidfVectorizer(stop_words='english')
tag_mat = tag_tfidf.fit_transform(movies['tag'])

def recommend_similar_movies(movie_ids, n=10):
    if not isinstance(movie_ids, list):
        movie_ids = [movie_ids]
        
    valid_ids = [m for m in movie_ids if m in movies["movieId"].values]
    if not valid_ids:
        raise ValueError("Selected movies not found in the dataset")
        
    indices = movies.index[movies["movieId"].isin(valid_ids)].tolist()
    
    # Compute similarity ON-THE-FLY to prevent huge memory overhead (~2GB dense matrices)
    # Calculate average vectors for the selected movies
    target_g_vec = gen_mat[indices].mean(axis=0)
    target_t_vec = tag_mat[indices].mean(axis=0)
    
    # Calculate similarity scores against all movies
    g_scores = linear_kernel(target_g_vec, gen_mat).flatten()
    t_scores = linear_kernel(target_t_vec, tag_mat).flatten()
    
    # Blend genre (65%) and tag (35%) similarities
    avg_sim = (g_scores * 0.65) + (t_scores * 0.35)
    
    # Calculate the average year of the targeted movies
    target_year = movies.iloc[indices]["year"].mean()
    
    # Apply a Gaussian year penalty (15-year standard deviation drops score naturally over time)
    # Convert to numpy array to ensure dot matching with avg_sim
    year_diff = movies["year"].values - target_year
    year_penalty = np.exp(-(year_diff**2) / (2 * (15**2)))
    
    avg_sim = avg_sim * year_penalty
    
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
            "imdbId": movies.iloc[i].get("imdbId"),
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
        imdb_row = movies[movies["movieId"] == movie_id]["imdbId"].values
        if len(title_row) > 0:
            result.append({
                "movieId": movie_id,
                "title": title_row[0],
                "genres": genres_row[0] if len(genres_row) > 0 else "",
                "tmdbId": tmdb_row[0] if len(tmdb_row) > 0 else None,
                "imdbId": imdb_row[0] if len(imdb_row) > 0 else None,
                "score": round(float(score), 3),
            })
    return result
