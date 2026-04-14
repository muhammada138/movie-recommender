import pytest
import pandas as pd
import numpy as np

# A tiny dummy dataset to test logic without loading large CSVs
@pytest.fixture
def dummy_movies():
    return pd.DataFrame({
        'movieId': [1, 2, 3],
        'title': ['Toy Story', 'Jumanji', 'Grumpier Old Men'],
        'genres': ['Adventure|Animation|Children|Comedy|Fantasy', 'Adventure|Children|Fantasy', 'Comedy|Romance'],
        'keywords': ['toy, animation', 'board game, jungle', 'old men, fishing']
    })

@pytest.fixture
def dummy_ratings():
    return pd.DataFrame({
        'userId': [1, 1, 2, 2],
        'movieId': [1, 2, 2, 3],
        'rating': [5.0, 4.0, 4.5, 3.0]
    })

def test_dataframe_fixtures(dummy_movies, dummy_ratings):
    assert len(dummy_movies) == 3
    assert len(dummy_ratings) == 4

# In a real test, we would import the specific matching functions from recommender.py
# Example: 
# from recommender import build_similarity_matrix, get_recommendations
# def test_build_similarity_matrix(dummy_movies, dummy_ratings):
#     matrix = build_similarity_matrix(dummy_ratings)
#     assert matrix is not None
