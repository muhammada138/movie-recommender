import pandas as pd
import numpy as np
import re
import logging
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_title(title):
    match = re.search(r'^(.*?)(,\s*(The|A|An))((\s*\(.*?\))*)$', title)
    if match:
        base = match.group(1)
        article = match.group(3)
        rest = match.group(4)
        return f"{article} {base}{rest}".strip()
    return title

def load_data(ratings_path="data/ratings.csv", movies_path="data/movies.csv", links_path="data/links.csv", tags_path="data/tags.csv", keywords_path="data/tmdb_keywords.csv"):
    try:
        ratings = pd.read_csv(ratings_path)
        movies = pd.read_csv(movies_path)
        links = pd.read_csv(links_path)
    except FileNotFoundError as e:
        logger.error(f"Critical data file missing: {e}")
        raise

    # 1. Process MovieLens User Tags
    try:
        tags = pd.read_csv(tags_path)
        tags['tag'] = tags['tag'].fillna('').astype(str).str.lower()
        grouped_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
        movies = movies.merge(grouped_tags, on='movieId', how='left')
    except Exception as e:
        logger.warning(f"Could not process tags: {e}")
        movies['tag'] = ''
    
    # 2. Process TMDB Contextual Keywords
    try:
        kw = pd.read_csv(keywords_path)
        kw['keywords'] = kw['keywords'].fillna('').astype(str).str.lower()
        movies = movies.merge(kw, on='movieId', how='left')
    except Exception as e:
        logger.warning(f"Could not process keywords: {e}")
        movies['keywords'] = ''

    # 3. Combine both metadata sources into a single 'tag' column for TF-IDF
    movies['tag'] = movies['tag'].fillna('') + ' ' + movies['keywords'].fillna('')
    movies['tag'] = movies['tag'].str.strip()

    if 'imdbId' in links.columns:
        links['imdbId'] = links['imdbId'].apply(lambda x: str(int(x)).zfill(7) if pd.notna(x) else x)
    movies = movies.merge(links[['movieId', 'tmdbId', 'imdbId']], on='movieId', how='left')
    
    # Extract year for time-based contextual penalization
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
    movies['year'] = movies['year'].fillna(movies['year'].median())
    
    movies['title'] = movies['title'].apply(format_title)
    return ratings, movies


class RecommenderEngine:
    def __init__(self):
        try:
            self.ratings, self.movies = load_data()
            self.matrix = self._build_user_item_matrix(self.ratings)
            self.similarity_df = self._build_similarity_matrix(self.matrix)
            self._init_tfidf()
        except Exception as e:
            logger.error(f"Failed to initialize recommender engine: {e}")
            self.ratings = pd.DataFrame()
            self.movies = pd.DataFrame(columns=['movieId', 'title', 'genres', 'tag', 'year', 'tmdbId', 'imdbId'])
            self.matrix = pd.DataFrame()
            self.similarity_df = pd.DataFrame()

    def _build_user_item_matrix(self, ratings):
        return ratings.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)

    def _build_similarity_matrix(self, matrix):
        sim = cosine_similarity(matrix)
        return pd.DataFrame(sim, index=matrix.index, columns=matrix.index)

    def _init_tfidf(self):
        self.movies['genres_str'] = self.movies['genres'].str.replace('|', ' ', regex=False).str.lower()
        self.gen_tfidf = TfidfVectorizer(stop_words='english')
        self.gen_mat = self.gen_tfidf.fit_transform(self.movies['genres_str'])
        self.tag_tfidf = TfidfVectorizer(stop_words='english')
        self.tag_mat = self.tag_tfidf.fit_transform(self.movies['tag'])

    def _format_results(self, indices_with_scores):
        result = []
        for i, score in indices_with_scores:
            row = self.movies.iloc[i]
            result.append({
                "movieId": row["movieId"],
                "title": row["title"],
                "genres": row["genres"],
                "tmdbId": row["tmdbId"],
                "imdbId": row.get("imdbId"),
                "score": round(float(score), 3),
            })
        return result

    def recommend_similar(self, movie_ids, n=10):
        if not isinstance(movie_ids, list):
            movie_ids = [movie_ids]
            
        valid_ids = [m for m in movie_ids if m in self.movies["movieId"].values]
        if not valid_ids:
            raise ValueError("Selected movies not found in the dataset")
            
        indices = self.movies.index[self.movies["movieId"].isin(valid_ids)].tolist()
        
        target_g_vec = np.asarray(self.gen_mat[indices].mean(axis=0))
        target_t_vec = np.asarray(self.tag_mat[indices].mean(axis=0))
        
        g_scores = linear_kernel(target_g_vec, self.gen_mat).flatten()
        t_scores = linear_kernel(target_t_vec, self.tag_mat).flatten()
        
        avg_sim = (g_scores * 0.45) + (t_scores * 0.55)
        target_year = self.movies.iloc[indices]["year"].mean()
        
        year_diff = self.movies["year"].values - target_year
        year_penalty = np.exp(-(year_diff**2) / (2 * (15**2)))
        avg_sim = avg_sim * year_penalty
        
        sim_scores = sorted(list(enumerate(avg_sim)), key=lambda x: x[1], reverse=True)
        sim_scores = [x for x in sim_scores if x[0] not in indices][:n]
        
        return self._format_results(sim_scores)

    def recommend_for_user(self, user_id, n=10, top_k_users=5):
        if user_id not in self.matrix.index:
            raise ValueError(f"User {user_id} not found")

        similar_users = self.similarity_df[user_id].sort_values(ascending=False)[1:top_k_users + 1]
        similar_users_ratings = self.matrix.loc[similar_users.index]
        weighted = similar_users_ratings.T.dot(similar_users) / (similar_users.sum() + 1e-9)

        already_seen = self.matrix.loc[user_id][self.matrix.loc[user_id] > 0].index
        recs = weighted.drop(already_seen, errors="ignore").sort_values(ascending=False).head(n)
        
        movie_id_to_idx = pd.Series(self.movies.index, index=self.movies['movieId'])
        indices_with_scores = []
        for m_id, score in recs.items():
            if m_id in movie_id_to_idx:
                indices_with_scores.append((movie_id_to_idx[m_id], score))
                
        return self._format_results(indices_with_scores)

    def get_user_rating_count(self, user_id):
        return len(self.ratings[self.ratings["userId"] == user_id])

# Lazy loading support
_engine = None

def _get_engine():
    global _engine
    if _engine is None:
        _engine = RecommenderEngine()
    return _engine

def __getattr__(name):
    if name == "engine":
        return _get_engine()
    if name == "ratings":
        return _get_engine().ratings
    if name == "movies":
        return _get_engine().movies
    raise AttributeError(f"module {__name__} has no attribute {name}")

def recommend_for_user(user_id, n=10, top_k_users=5):
    return _get_engine().recommend_for_user(user_id, n, top_k_users)

def recommend_similar_movies(movie_ids, n=10):
    return _get_engine().recommend_similar(movie_ids, n)

def get_user_rating_count(user_id, ratings=None):
    return _get_engine().get_user_rating_count(user_id)
