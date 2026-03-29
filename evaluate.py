"""
Train/test evaluation for the collaborative filtering model.

We split the ratings into 80/20, rebuild the full pipeline on training data only
(no peeking at test ratings when computing similarity), then predict ratings for
the test set and compute RMSE. This is the correct way to do it — rebuilding on
train-only data avoids leakage where the model has already seen the test ratings.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity


def run_evaluation(ratings_path="data/ratings.csv"):
    ratings = pd.read_csv(ratings_path)

    train, test = train_test_split(ratings, test_size=0.2, random_state=42)
    print(f"Train size: {len(train):,}  |  Test size: {len(test):,}")

    # build matrix and similarity using ONLY training data
    train_matrix = train.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)

    train_similarity = pd.DataFrame(
        cosine_similarity(train_matrix),
        index=train_matrix.index,
        columns=train_matrix.index
    )

    global_mean = train["rating"].mean()

    # --- SVD (Matrix Factorization) ---
    print("Training SVD model...")
    from sklearn.decomposition import TruncatedSVD
    
    # Mean-center the ratings by user (only using non-zero ratings)
    train_matrix_nan = train.pivot_table(index="userId", columns="movieId", values="rating")
    user_ratings_mean = train_matrix_nan.mean(axis=1)
    # Filling NaNs with 0 after centering is mathematically equivalent to filling NaNs with user_mean initially
    train_matrix_centered = train_matrix_nan.sub(user_ratings_mean, axis=0).fillna(0)
    
    n_components = min(20, len(train_matrix_centered.index) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(train_matrix_centered.values)
    item_factors = svd.components_
    
    # Reconstruct the matrix and add the user means back
    svd_reconstructed = np.dot(user_factors, item_factors)
    svd_pred_df = pd.DataFrame(svd_reconstructed, index=train_matrix.index, columns=train_matrix.columns)
    svd_pred_df = svd_pred_df.add(user_ratings_mean, axis=0)

    def predict_svd(user_id, movie_id):
        if user_id in svd_pred_df.index and movie_id in svd_pred_df.columns:
            return svd_pred_df.loc[user_id, movie_id]
        return global_mean

    def predict_rating(user_id, movie_id):
        # fall back to global mean if user or movie wasn't in training data
        if movie_id not in train_matrix.columns or user_id not in train_matrix.index:
            return global_mean

        users_who_rated = train_matrix[train_matrix[movie_id] > 0].index
        if len(users_who_rated) == 0:
            return global_mean

        sims = train_similarity.loc[user_id, users_who_rated]
        movie_ratings = train_matrix.loc[users_who_rated, movie_id]

        if sims.sum() == 0:
            return global_mean

        return np.dot(sims, movie_ratings) / sims.sum()

    print("Predicting ratings on test set... (this takes a minute)")
    y_pred_cf = test.apply(
        lambda row: predict_rating(row["userId"], row["movieId"]), axis=1
    )
    y_pred_svd = test.apply(
        lambda row: predict_svd(row["userId"], row["movieId"]), axis=1
    )
    y_true = test["rating"]

    rmse_cf = np.sqrt(mean_squared_error(y_true, y_pred_cf))
    rmse_svd = np.sqrt(mean_squared_error(y_true, y_pred_svd))
    rmse_baseline = np.sqrt(mean_squared_error(y_true, [global_mean] * len(y_true)))
    
    print("\n--- RESULTS ---")
    print(f"Collaborative Filtering RMSE: {rmse_cf:.4f}")
    print(f"SVD Matrix Factorization RMSE: {rmse_svd:.4f}")
    print(f"Baseline RMSE (mean):         {rmse_baseline:.4f}")
    return {"CF": rmse_cf, "SVD": rmse_svd, "Baseline": rmse_baseline}


if __name__ == "__main__":
    run_evaluation()
