# Movie Recommender

Collaborative filtering on the MovieLens dataset. No LLMs, no APIs — just pandas, numpy, and scikit-learn implementing the algorithm from scratch.

Live demo: [Streamlit Cloud](https://share.streamlit.io) _(deploy link goes here after Streamlit Cloud setup)_

## How it works

The model builds a user×movie rating matrix (610 users × 9,724 movies) and computes cosine similarity between every pair of users. When you ask for recommendations for a given user, it:

1. Finds the K most similar users to you by cosine similarity
2. Takes their ratings for movies you haven't seen
3. Computes a weighted average — users who are more similar to you have more influence
4. Returns the top N movies by that weighted score

This is user-based collaborative filtering, the same core technique behind Netflix and Spotify's recommendation systems.

## Model performance

Evaluated on an 80/20 train/test split. The model sees only training ratings when computing similarity — no leakage.

| Method | RMSE |
|---|---|
| Model (collaborative filtering) | **0.9764** |
| Baseline (always predict mean) | 1.0488 |

The model beats the mean-prediction baseline by ~7%.

## Known limitations

**Cold start:** If a user has no ratings (or very few), there's nothing to compute similarity against and the recommendations will be noisy. The model falls back to the global mean rating for users or movies not seen in training data. This is a fundamental limitation of collaborative filtering — it needs history to work well. Solutions include hybrid models (combining content-based with collaborative) or matrix factorization (SVD), but those add complexity this project deliberately avoids.

**Sparsity:** The rating matrix is ~98% empty. Most users haven't rated most movies, so similarity scores are computed from relatively thin overlap.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

For EDA:
```bash
jupyter notebook notebooks/eda.ipynb
```

To run the RMSE evaluation:
```bash
python evaluate.py
```

## Dataset

[MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/) — 100,836 ratings from 610 users on 9,742 movies. Ratings are on a 0.5–5.0 scale.

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 22:1–22:19.

## Project structure

```
├── app.py              # Streamlit web app
├── recommender.py      # core model logic
├── evaluate.py         # RMSE evaluation on train/test split
├── data/
│   ├── movies.csv
│   └── ratings.csv
├── notebooks/
│   └── eda.ipynb       # exploratory data analysis
└── requirements.txt
```
