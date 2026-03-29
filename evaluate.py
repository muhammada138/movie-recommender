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
    y_pred = test.apply(
        lambda row: predict_rating(row["userId"], row["movieId"]), axis=1
    )
    y_true = test["rating"]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\nModel RMSE: {rmse:.4f}")
    print(f"Baseline RMSE (always predict mean): {np.sqrt(mean_squared_error(y_true, [global_mean] * len(y_true))):.4f}")
    return rmse


if __name__ == "__main__":
    run_evaluation()
