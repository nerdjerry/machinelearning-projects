"""
Netflix Show Clustering with KMeans
====================================
A Streamlit application that demonstrates unsupervised-learning concepts
using Netflix catalogue data:

  1. Data cleaning and encoding categorical features (genre, rating)
  2. Why feature scaling is essential for KMeans clustering
  3. Choosing the optimal K with the Elbow Method and Silhouette Score
  4. Dimensionality reduction with PCA for 2-D visualization
  5. Interpreting what each cluster represents

Run with:  streamlit run app.py
"""

# =============================================================
# 1. Imports
# =============================================================
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Keep Streamlit output clean
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================
# 2. Page configuration
# =============================================================
st.set_page_config(
    page_title="Netflix Clustering – KMeans & PCA",
    page_icon="🎬",
    layout="wide",
)

# =============================================================
# 3. Constants
# =============================================================

# Content ratings commonly found on Netflix
RATINGS = ["TV-MA", "TV-14", "TV-PG", "R", "PG-13", "TV-Y7", "TV-Y", "PG",
           "TV-G", "NR", "G", "NC-17"]

# Genre pool used for both real and synthetic data
GENRE_POOL = [
    "International Movies", "Dramas", "Comedies", "Action & Adventure",
    "Documentaries", "Children & Family Movies", "Romantic Movies",
    "Horror Movies", "Thrillers", "Stand-Up Comedy", "Sci-Fi & Fantasy",
    "Crime TV Shows", "TV Dramas", "Reality TV", "Kids' TV",
    "Anime Series", "Classic Movies", "Music & Musicals",
    "Independent Movies", "British TV Shows",
]

# Countries commonly represented on Netflix
COUNTRY_POOL = [
    "United States", "India", "United Kingdom", "Canada", "France",
    "Japan", "South Korea", "Spain", "Mexico", "Australia",
    "Germany", "Nigeria", "Brazil", "Turkey", "Egypt",
]


# =============================================================
# 4. Data loading helpers
# =============================================================

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load Netflix titles from CSV, Kaggle, or generate a synthetic dataset.

    Priority order:
      1. Local CSV at ``data/netflix_titles.csv``
      2. Automatic download via ``kagglehub`` (requires Kaggle credentials)
      3. Synthetic data (~2 000 rows) whose distributions approximate the
         real Netflix catalogue
    """
    csv_path = os.path.join(os.path.dirname(__file__), "data", "netflix_titles.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        expected = {"type", "title", "rating", "listed_in", "release_year"}
        if expected.issubset(df.columns):
            return df

    # ----- Try Kaggle download -----
    try:
        import kagglehub  # noqa: E401 – optional dependency

        dataset_path = kagglehub.dataset_download("shivamb/netflix-shows")
        kaggle_csv = os.path.join(dataset_path, "netflix_titles.csv")
        if os.path.exists(kaggle_csv):
            df = pd.read_csv(kaggle_csv)
            expected = {"type", "title", "rating", "listed_in", "release_year"}
            if expected.issubset(df.columns):
                return df
    except Exception:
        pass  # Kaggle credentials not configured or download failed

    # ----- Synthetic data generation -----
    rng = np.random.RandomState(42)
    n = 2000

    # ~70 % Movies, ~30 % TV Shows (matches real Netflix distribution)
    types = rng.choice(["Movie", "TV Show"], size=n, p=[0.70, 0.30])

    # Release years skew recent
    release_years = rng.normal(loc=2015, scale=6, size=n).clip(1970, 2023).astype(int)

    # Ratings with realistic frequencies
    rating_weights = np.array([0.22, 0.21, 0.12, 0.10, 0.09, 0.06,
                               0.05, 0.05, 0.04, 0.03, 0.02, 0.01])
    ratings = rng.choice(RATINGS, size=n, p=rating_weights)

    # Each title gets 1-3 genres
    genres_list = []
    for _ in range(n):
        k = rng.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
        genres_list.append(", ".join(rng.choice(GENRE_POOL, size=k, replace=False)))

    # Duration depends on type
    durations = []
    for t in types:
        if t == "Movie":
            mins = int(np.clip(rng.normal(100, 25), 30, 240))
            durations.append(f"{mins} min")
        else:
            seasons = int(rng.choice([1, 2, 3, 4, 5], p=[0.45, 0.25, 0.15, 0.10, 0.05]))
            durations.append(f"{seasons} Season{'s' if seasons != 1 else ''}")

    # Simple synthetic directors, cast, and descriptions
    first_names = ["Alex", "Jordan", "Sam", "Chris", "Pat", "Morgan", "Taylor",
                   "Casey", "Riley", "Jamie", "Drew", "Quinn"]
    last_names = ["Smith", "Kim", "Garcia", "Chen", "Patel", "Müller", "Tanaka",
                  "Silva", "Johansson", "Ali", "Brown", "Okafor"]
    adjectives = ["thrilling", "heartwarming", "gripping", "hilarious",
                  "thought-provoking", "visually stunning", "intense",
                  "charming", "dark", "inspiring"]
    nouns = ["journey", "story", "adventure", "mystery", "tale", "saga",
             "drama", "comedy", "documentary", "series"]

    directors, casts, descriptions, countries, dates_added = [], [], [], [], []
    for i in range(n):
        directors.append(f"{rng.choice(first_names)} {rng.choice(last_names)}")
        cast_size = rng.randint(2, 6)
        casts.append(", ".join(
            f"{rng.choice(first_names)} {rng.choice(last_names)}"
            for _ in range(cast_size)
        ))
        descriptions.append(
            f"A {rng.choice(adjectives)} {rng.choice(nouns)} that "
            f"explores {rng.choice(adjectives)} themes."
        )
        countries.append(rng.choice(COUNTRY_POOL))
        month = rng.randint(1, 13)
        day = rng.randint(1, 29)
        year = rng.choice([2019, 2020, 2021, 2022, 2023])
        dates_added.append(f"{month:02d}/{day:02d}/{year}")

    df = pd.DataFrame({
        "show_id": [f"s{i+1}" for i in range(n)],
        "type": types,
        "title": [f"Title {i+1}" for i in range(n)],
        "director": directors,
        "cast": casts,
        "country": countries,
        "date_added": dates_added,
        "release_year": release_years,
        "rating": ratings,
        "duration": durations,
        "listed_in": genres_list,
        "description": descriptions,
    })

    return df


# =============================================================
# 5. Feature engineering helpers
# =============================================================

@st.cache_data
def engineer_features(df: pd.DataFrame) -> tuple:
    """Clean data and build a numeric feature matrix for clustering.

    Steps
    -----
    1. Label-encode the content **rating** (ordinal-like categories).
    2. One-hot encode the top genres extracted from ``listed_in``.
    3. Extract a numeric **duration** value (minutes for movies, seasons
       for TV shows).
    4. Include ``release_year`` and a binary ``is_movie`` flag.

    Returns (feature_df, encoded_df, genre_columns) where:
    - ``feature_df`` is the human-readable feature table
    - ``encoded_df`` has only the numeric columns used for clustering
    - ``genre_columns`` lists the one-hot genre column names
    """
    clean = df.copy()

    # --- 1. Rating encoding (label encoding) ---
    # Label encoding maps each unique rating to an integer.
    # This works well here because KMeans treats them as numeric distances.
    le_rating = LabelEncoder()
    clean["rating_encoded"] = le_rating.fit_transform(
        clean["rating"].fillna("NR")
    )

    # --- 2. One-hot encoding for top genres ---
    # Each title can belong to multiple genres (comma-separated).  We
    # create a binary column for each of the top genres.
    top_genres = (
        clean["listed_in"]
        .fillna("")
        .str.split(", ")
        .explode()
        .value_counts()
        .head(12)
        .index.tolist()
    )

    for genre in top_genres:
        clean[f"genre_{genre}"] = (
            clean["listed_in"].fillna("").str.contains(genre, regex=False).astype(int)
        )
    genre_cols = [c for c in clean.columns if c.startswith("genre_")]

    # --- 3. Numeric duration ---
    # Movies are stored as "90 min", TV shows as "2 Seasons".
    # We extract the leading integer so KMeans can use it.
    clean["duration_num"] = (
        clean["duration"]
        .fillna("0")
        .str.extract(r"(\d+)", expand=False)
        .astype(float)
        .fillna(0)
    )

    # --- 4. Binary type flag ---
    clean["is_movie"] = (clean["type"] == "Movie").astype(int)

    # Assemble the numeric feature matrix
    feature_cols = ["release_year", "rating_encoded", "duration_num",
                    "is_movie"] + genre_cols
    encoded_df = clean[feature_cols].copy()

    return clean, encoded_df, genre_cols


@st.cache_data
def scale_features(_encoded_df: pd.DataFrame) -> tuple:
    """Apply StandardScaler and return scaled array + scaler column names.

    Why scale?
    ----------
    KMeans uses Euclidean distance to assign points to clusters.  If one
    feature (e.g. ``release_year`` ~ 2020) has a much larger range than
    another (e.g. ``is_movie`` ∈ {0, 1}), the large-range feature will
    dominate distance calculations and the smaller features become
    irrelevant.  StandardScaler centres each feature to mean=0, std=1 so
    every feature contributes equally.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(_encoded_df)
    return scaled, _encoded_df.columns.tolist()


# =============================================================
# 6. Clustering helpers
# =============================================================

@st.cache_data
def compute_elbow_and_silhouette(
    _scaled: np.ndarray, k_range: tuple = (2, 11)
) -> pd.DataFrame:
    """Compute inertia (elbow) and silhouette scores for a range of K.

    The **elbow method** plots inertia (within-cluster sum of squares)
    versus K.  The "elbow" — the point where adding more clusters yields
    diminishing returns — suggests a good K.

    The **silhouette score** measures how similar each point is to its own
    cluster versus the nearest other cluster.  Values range from -1 to 1;
    higher is better.
    """
    records = []
    for k in range(k_range[0], k_range[1]):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(_scaled)
        records.append({
            "K": k,
            "Inertia": km.inertia_,
            "Silhouette": silhouette_score(_scaled, labels),
        })

    return pd.DataFrame(records)


@st.cache_data
def run_kmeans(_scaled: np.ndarray, k: int) -> np.ndarray:
    """Fit KMeans with *k* clusters and return cluster labels."""
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    return km.fit_predict(_scaled)


@st.cache_data
def run_pca(_scaled: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Reduce dimensionality with PCA for 2-D scatter visualisation.

    PCA finds the orthogonal directions (principal components) that
    capture the most variance.  Projecting onto 2 components lets us
    plot high-dimensional cluster assignments on a flat scatter plot.
    """
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(_scaled)


# =============================================================
# 7. Plotting helpers
# =============================================================

def plot_elbow(metrics_df: pd.DataFrame) -> plt.Figure:
    """Return a figure with the Elbow Method (inertia vs K)."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(metrics_df["K"], metrics_df["Inertia"], "o-", color="#E50914")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia (WCSS)")
    ax.set_title("Elbow Method – Inertia vs K")
    ax.set_xticks(metrics_df["K"])
    fig.tight_layout()
    return fig


def plot_silhouette(metrics_df: pd.DataFrame) -> plt.Figure:
    """Return a figure with Silhouette Score vs K."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(metrics_df["K"], metrics_df["Silhouette"], "s-", color="#221F1F")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score vs K")
    ax.set_xticks(metrics_df["K"])
    fig.tight_layout()
    return fig


def plot_pca_scatter(
    pca_result: np.ndarray, labels: np.ndarray
) -> plt.Figure:
    """Return a 2-D PCA scatter plot coloured by cluster assignment."""
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        pca_result[:, 0],
        pca_result[:, 1],
        c=labels,
        cmap="Set2",
        alpha=0.6,
        edgecolors="w",
        linewidth=0.3,
        s=30,
    )
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA – Cluster Visualisation")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    return fig


def plot_distribution(series: pd.Series, title: str, top_n: int = 10) -> plt.Figure:
    """Return a horizontal bar chart of the top-N value counts."""
    counts = series.value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(counts) + 1))
    counts.sort_values().plot.barh(ax=ax, color="#E50914")
    ax.set_title(title)
    ax.set_xlabel("Count")
    fig.tight_layout()
    return fig


# =============================================================
# 8. Main application
# =============================================================

def main() -> None:
    """Entry point for the Streamlit application."""

    # --- Header ---
    st.title("🎬 Netflix Show Clustering")
    st.markdown(
        """
        This interactive demo applies **KMeans clustering** to Netflix
        catalogue data.  You will explore:

        | Concept | What you'll see |
        |---|---|
        | **Data Cleaning** | Handling missing values, encoding categories |
        | **Feature Scaling** | Why KMeans *needs* scaled features |
        | **Elbow Method** | Picking the right number of clusters |
        | **Silhouette Score** | Quantifying cluster quality |
        | **PCA** | Visualising high-dimensional clusters in 2-D |
        """
    )
    st.divider()

    # --- Load data ---
    df = load_data()

    # --- Sidebar controls ---
    st.sidebar.header("⚙️ Settings")
    num_clusters = st.sidebar.slider(
        "Number of clusters (K)", min_value=2, max_value=10, value=4
    )

    # --- Feature engineering (runs once, cached) ---
    clean_df, encoded_df, genre_cols = engineer_features(df)
    scaled, feature_names = scale_features(encoded_df)

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Data Overview",
            "🔧 Feature Engineering",
            "📐 Optimal K Selection",
            "🗺️ Cluster Visualisation",
            "🔍 Cluster Analysis",
        ]
    )

    # ---------------------------------------------------------
    # Tab 1 – Data Overview
    # ---------------------------------------------------------
    with tab1:
        st.subheader("Dataset at a Glance")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Titles", len(df))
        with col2:
            st.metric("Movies", int((df["type"] == "Movie").sum()))
        with col3:
            st.metric("TV Shows", int((df["type"] == "TV Show").sum()))

        with st.expander("Show raw data"):
            st.dataframe(df, use_container_width=True)

        with st.expander("Descriptive statistics"):
            st.dataframe(
                df.describe(include="all").T.fillna(""),
                use_container_width=True,
            )

        # Distribution plots
        st.subheader("Distributions")
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.pyplot(plot_distribution(df["type"], "Content Type"))
            st.pyplot(plot_distribution(df["rating"].fillna("NR"), "Ratings"))

        with d_col2:
            # Explode genres for a per-genre count
            genre_series = df["listed_in"].fillna("").str.split(", ").explode()
            st.pyplot(plot_distribution(genre_series, "Top Genres", top_n=12))

            st.pyplot(
                plot_distribution(
                    df["release_year"].astype(str), "Release Year", top_n=10
                )
            )

    # ---------------------------------------------------------
    # Tab 2 – Feature Engineering
    # ---------------------------------------------------------
    with tab2:
        st.subheader("Turning Categories into Numbers")

        # --- Rating encoding explanation ---
        st.markdown("#### 1 · Label Encoding the Rating Column")
        st.markdown(
            """
            **Label encoding** assigns a unique integer to each rating
            category (e.g. *TV-MA → 7, PG-13 → 3*).  This is simple and
            compact but implies an ordinal relationship between values, so
            use it with care.  For KMeans the numeric distance is what
            matters — combined with scaling it works well enough here.
            """
        )
        # Show a small before/after table
        sample = clean_df[["title", "rating", "rating_encoded"]].drop_duplicates(
            subset="rating"
        ).head(8)
        st.dataframe(sample, use_container_width=True)

        # --- Genre one-hot explanation ---
        st.markdown("#### 2 · One-Hot Encoding Genres")
        st.markdown(
            """
            Each title can belong to **multiple genres** (comma-separated).
            One-hot encoding creates a binary column per genre — a title
            gets a **1** in every genre it belongs to.  This avoids any
            false ordinal relationship between genres.
            """
        )
        st.dataframe(
            clean_df[["title", "listed_in"] + genre_cols].head(8),
            use_container_width=True,
        )

        # --- Scaling explanation ---
        st.markdown("#### 3 · Feature Scaling (StandardScaler)")
        st.markdown(
            """
            > **Why does KMeans break without scaling?**
            >
            > KMeans minimises the *sum of squared Euclidean distances*
            > from each point to its cluster centroid.  If `release_year`
            > ranges from 1970–2023 while `is_movie` is just 0 or 1, the
            > year feature will **dominate** every distance calculation and
            > the binary features become noise.
            >
            > **StandardScaler** transforms each feature to zero mean and
            > unit variance, ensuring equal contribution.
            """
        )

        before_after = pd.DataFrame({
            "Feature": feature_names,
            "Mean (raw)": encoded_df.mean().round(2).values,
            "Std (raw)": encoded_df.std().round(2).values,
            "Mean (scaled)": scaled.mean(axis=0).round(4),
            "Std (scaled)": scaled.std(axis=0).round(4),
        })
        st.dataframe(before_after, use_container_width=True)

        st.markdown("#### 4 · Final Feature Matrix Preview")
        st.dataframe(
            pd.DataFrame(scaled, columns=feature_names).head(10).round(3),
            use_container_width=True,
        )

    # ---------------------------------------------------------
    # Tab 3 – Optimal K Selection
    # ---------------------------------------------------------
    with tab3:
        st.subheader("How Many Clusters?")
        st.markdown(
            """
            Two complementary techniques help us choose **K**:

            * **Elbow Method** — plot *inertia* (within-cluster sum of
              squares) vs K.  The "elbow" where the curve bends sharply
              suggests diminishing returns from adding more clusters.
            * **Silhouette Score** — measures how well each point fits its
              own cluster vs. the nearest neighbour cluster.  Ranges from
              −1 (wrong cluster) to +1 (dense, well-separated clusters).
              Pick the K with the highest silhouette score.
            """
        )

        metrics_df = compute_elbow_and_silhouette(scaled)

        col_e, col_s = st.columns(2)
        with col_e:
            st.pyplot(plot_elbow(metrics_df))
        with col_s:
            st.pyplot(plot_silhouette(metrics_df))

        st.dataframe(
            metrics_df.set_index("K").round(4), use_container_width=True
        )

        best_k = int(metrics_df.loc[metrics_df["Silhouette"].idxmax(), "K"])
        st.info(
            f"📌 The highest silhouette score is at **K = {best_k}**.  "
            f"Use the sidebar slider to experiment with different values."
        )

    # ---------------------------------------------------------
    # Tab 4 – Cluster Visualisation
    # ---------------------------------------------------------
    with tab4:
        st.subheader(f"PCA Scatter Plot (K = {num_clusters})")
        st.markdown(
            """
            **PCA (Principal Component Analysis)** projects the
            high-dimensional feature space onto 2 axes that capture the
            most variance.  Each dot is a Netflix title, coloured by its
            KMeans cluster assignment.
            """
        )

        labels = run_kmeans(scaled, num_clusters)
        pca_2d = run_pca(scaled)

        st.pyplot(plot_pca_scatter(pca_2d, labels))

        sil = silhouette_score(scaled, labels)
        st.metric("Silhouette Score", f"{sil:.4f}")

    # ---------------------------------------------------------
    # Tab 5 – Cluster Analysis
    # ---------------------------------------------------------
    with tab5:
        st.subheader(f"What's Inside Each Cluster? (K = {num_clusters})")
        st.markdown(
            "Below we profile each cluster by its dominant content type, "
            "ratings, genres, and average release year."
        )

        labels = run_kmeans(scaled, num_clusters)
        analysis_df = clean_df.copy()
        analysis_df["Cluster"] = labels

        for cluster_id in sorted(analysis_df["Cluster"].unique()):
            subset = analysis_df[analysis_df["Cluster"] == cluster_id]

            st.markdown(f"---\n### Cluster {cluster_id}  ({len(subset)} titles)")

            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Titles", len(subset))
            with m_col2:
                st.metric("Avg Release Year", int(subset["release_year"].mean()))
            with m_col3:
                pct_movie = (subset["type"] == "Movie").mean()
                st.metric("% Movies", f"{pct_movie:.0%}")

            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.markdown("**Top Ratings**")
                st.dataframe(
                    subset["rating"].value_counts().head(5).reset_index(),
                    use_container_width=True,
                )
            with info_col2:
                st.markdown("**Top Genres**")
                top_g = (
                    subset["listed_in"]
                    .fillna("")
                    .str.split(", ")
                    .explode()
                    .value_counts()
                    .head(5)
                    .reset_index()
                )
                st.dataframe(top_g, use_container_width=True)

            with st.expander(f"Sample titles from Cluster {cluster_id}"):
                st.dataframe(
                    subset[["title", "type", "rating", "release_year",
                            "listed_in"]].head(10),
                    use_container_width=True,
                )


# =============================================================
# 9. Run
# =============================================================
if __name__ == "__main__":
    main()
