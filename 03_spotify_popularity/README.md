# 🎵 Spotify Song Popularity Predictor

**Difficulty: 5/10 — First taste of live API data**

## What It Is

Use Spotify audio features like tempo, energy, danceability, and valence to train a regression model that predicts a track's popularity score. Compare tree-based models against a linear baseline and explain what actually drives popularity using SHAP values.

## Tech Stack

- Python, pandas, XGBoost, LightGBM, SHAP, scikit-learn, Matplotlib/Seaborn, Streamlit

## What You Learn

- Feature correlation analysis and handling multicollinearity
- Gradient boosting hyperparameter tuning (n_estimators, max_depth, learning_rate)
- Model explainability using SHAP values — a major portfolio differentiator
- Comparing Linear Regression, Random Forest, XGBoost, and LightGBM

## How to Run

```bash
# From the project root directory
pip install -r requirements.txt
streamlit run 03_spotify_popularity/app.py
```

## Using Real Data

The app generates synthetic Spotify-like data. To use real data:

1. Download a [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) from Kaggle
2. Place the CSV file at `03_spotify_popularity/data/spotify_tracks.csv`
3. Restart the app

Alternatively, you can use the [Spotipy](https://spotipy.readthedocs.io/) library to collect data directly from the Spotify API.
