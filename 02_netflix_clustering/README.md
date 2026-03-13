# 🎬 Netflix Show Clustering

**Difficulty: 3/10 — Clean data, no labels needed**

## What It Is

Group similar Netflix shows using unsupervised learning based on genre, rating, duration, and other metadata. Visualize the clusters to find patterns like "critically acclaimed short dramas" vs "long binge-worthy comedies."

## Tech Stack

- Python, pandas, scikit-learn (KMeans, PCA), Matplotlib/Seaborn, Streamlit

## What You Learn

- Data cleaning and encoding categorical features (genre, rating)
- Feature scaling before clustering (why KMeans breaks without it)
- Choosing optimal K using the elbow method and silhouette score
- Dimensionality reduction with PCA for visualization

## How to Run

```bash
# From the project root directory
pip install -r requirements.txt
streamlit run 02_netflix_clustering/app.py
```

## Using Real Data

The app generates synthetic Netflix-like data. To use the real dataset:

1. Download the [Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows) dataset from Kaggle
2. Place the CSV file at `02_netflix_clustering/data/netflix_titles.csv`
3. Restart the app
