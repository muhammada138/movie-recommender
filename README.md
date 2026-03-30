# Movie Recommender

A recommendation system built completely from scratch. No APIs, no black boxes—just math, pandas, and scikit-learn proving how Netflix and Spotify powered their core models before deep learning.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)

Live demo: [Streamlit App](https://cf-movie-recommender.streamlit.app/) _(update this with your final app URL)_

![App Screenshot](assets/screenshot.png)

---

## What it does

Most tutorial ML projects call an external API. This project implements actual collaborative and content-based filtering algorithms from the ground up on a highly sparse matrix of 100,000+ real ratings. 

**Key features:**

- **User-Based Collaborative Filtering** - Computes a 610x610 user similarity matrix to recommend movies based on the tastes of your nearest neighbors
- **Content-Based Similarity** - Generates TF-IDF vectors from movie genres to recommend semantically similar films, regardless of user ratings
- **Matrix Factorization** - Evaluates a Truncated SVD model on the rating matrix to handle sparsity, achieving a much lower RMSE than pure distance metrics
- **Cold-Start Handling** - Explicit minimum ratings filters and global mean fallbacks gracefully handle new users or unrated items
- **Interactive UI** - Streamlit dashboard with dedicated tabs to test both the collaborative and content-based models live
- **Proper ML Evaluation** - Strict 80/20 train/test split. Matrix similarities are rebuilt exclusively on training data to explicitly prevent data leakage

---

## Tech stack

| Layer | Tech |
|---|---|
| Frontend / UI | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-Learn (Cosine Similarity, Linear Kernel, TruncatedSVD) |
| Dataset | MovieLens Latest Small |

---

## Getting started

You don't need any API keys. The dataset is inherently included in the `/data` directory.

### Running Locally

```bash
git clone https://github.com/muhammada138/movie-recommender.git
cd movie-recommender
pip install -r requirements.txt
streamlit run app.py
```

### Running Evaluation / EDA

To compute the RMSE of the collaborative filtering vs SVD:
```bash
python evaluate.py
```

To view the Exploratory Data Analysis:
```bash
# Note: jupyter, matplotlib, and seaborn are required for the notebook
jupyter notebook notebooks/eda.ipynb
```

---

## How the algorithm works

The core `recommend_for_user` function relies entirely on mathematical similarity:

1. Loads the 610 user × 9,724 movie sparse rating matrix (~98% empty)
2. Computes the cosine similarity between every pair of users
3. Finds the *K* most similar users to the target user
4. Takes their ratings for movies the target user hasn't seen yet
5. Computes a weighted average—giving inherently higher mathematical weight to users with a higher similarity score
6. Returns the top *N* movies sorted by predicted weighted score

---

## Model performance

Evaluated against the held-out test data.

| Method | RMSE |
|---|---|
| Latent Factor SVD (Mean-Centered) | **0.9304** |
| User-Based Collaborative Filtering | 0.9764 |
| Baseline (Predict global mean) | 1.0488 |

The basic collaborative model cleanly beats predicting the global mean by ~7%. However, the SVD matrix factorization outright outperforms it by uncovering latent features—demonstrating exactly why Matrix Factorization became the industry standard for sparse recommendation problems (like the Netflix Prize).

---

## Dataset & citations

[MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/) — 100,836 ratings from 610 users on 9,742 movies. Ratings are on a 0.5–5.0 scale.

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 22:1–22:19.
