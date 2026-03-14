# 🎵 Spotify Song Popularity Predictor

**Difficulty: 5/10 — API integration with advanced regression**

## What It Is

Use Spotify audio features like tempo, energy, danceability, and valence to train a regression model that predicts a track's popularity score. Compare tree-based models against a linear baseline and explain what actually drives popularity using SHAP values.

## Tech Stack

- Python, pandas, XGBoost, LightGBM, SHAP, scikit-learn, Matplotlib/Seaborn, Streamlit, Spotipy

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

## Data Sources

The app tries these sources in order:

1. **Local CSV** — Place a CSV file at `03_spotify_popularity/data/spotify_tracks.csv`
2. **Spotify Web API** — Set `SPOTIPY_CLIENT_ID` and `SPOTIPY_CLIENT_SECRET` environment variables ([create an app on the Spotify Developer Dashboard](https://developer.spotify.com/dashboard))
3. **Kaggle** — Automatically downloads the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) if Kaggle credentials are configured (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars)
4. **Synthetic data** — Generates ~5 000 realistic rows as a fallback
