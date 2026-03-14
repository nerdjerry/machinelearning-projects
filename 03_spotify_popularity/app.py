"""
Spotify Song Popularity Predictor
==================================
A Streamlit application that predicts song popularity using audio features.
Compares Linear Regression, Random Forest, XGBoost, and LightGBM models,
and explains predictions with SHAP values.

Run with:  streamlit run app.py
"""

# =============================================================
# Imports
# =============================================================
import os
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import streamlit as st
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Suppress noisy warnings so the Streamlit UI stays clean
warnings.filterwarnings("ignore")

# =============================================================
# Page configuration
# =============================================================
st.set_page_config(
    page_title="Spotify Popularity Predictor",
    page_icon="🎵",
    layout="wide",
)

# =============================================================
# Constants – column definitions & data paths
# =============================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "spotify_tracks.csv")

# Audio features used as model inputs
AUDIO_FEATURES = [
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]

TARGET = "popularity"


# =============================================================
# Data loading / synthetic generation
# =============================================================
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the Spotify dataset from CSV or generate a synthetic fallback.

    The synthetic dataset mimics real Spotify audio-feature distributions
    so the ML pipeline behaves realistically even without real data.
    """
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        # Make sure all required columns are present
        required = AUDIO_FEATURES + [TARGET, "track_name", "artist_name", "album_name"]
        if all(col in df.columns for col in required):
            return df

    # --- Synthetic fallback (~5000 rows) ---
    rng = np.random.RandomState(42)
    n = 5000

    df = pd.DataFrame(
        {
            "track_name": [f"Track_{i}" for i in range(n)],
            "artist_name": [f"Artist_{rng.randint(0, 500)}" for _ in range(n)],
            "album_name": [f"Album_{rng.randint(0, 1000)}" for _ in range(n)],
            # Duration: 1–7 minutes in milliseconds
            "duration_ms": rng.randint(60_000, 420_000, n),
            "explicit": rng.choice([0, 1], n, p=[0.7, 0.3]),
            # Audio features on 0–1 scale (beta distributions for realism)
            "danceability": rng.beta(5, 3, n),
            "energy": rng.beta(5, 3, n),
            "key": rng.randint(0, 12, n),
            "loudness": rng.uniform(-60, 0, n),
            "mode": rng.choice([0, 1], n, p=[0.4, 0.6]),
            "speechiness": rng.beta(1.5, 10, n),
            "acousticness": rng.beta(1.5, 5, n),
            "instrumentalness": rng.beta(1, 10, n),
            "liveness": rng.beta(2, 8, n),
            "valence": rng.beta(3, 3, n),
            "tempo": rng.uniform(50, 200, n),
            "time_signature": rng.choice([3, 4, 5, 6, 7], n, p=[0.05, 0.75, 0.1, 0.05, 0.05]),
        }
    )

    # Popularity is a weighted combination of features + noise so that
    # models have something meaningful to learn.
    df["popularity"] = np.clip(
        (
            20
            + 30 * df["danceability"]
            + 20 * df["energy"]
            + 10 * df["valence"]
            + 0.15 * df["loudness"]  # louder songs tend to be more popular
            - 15 * df["acousticness"]
            - 10 * df["instrumentalness"]
            + 5 * df["explicit"]
            + rng.normal(0, 8, n)  # random noise
        ),
        0,
        100,
    ).astype(int)

    return df


# =============================================================
# Feature engineering helpers
# =============================================================
@st.cache_data
def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return the Pearson correlation matrix for numeric columns."""
    return df[AUDIO_FEATURES + [TARGET]].corr()


@st.cache_data
def split_and_scale(df: pd.DataFrame):
    """Split data 80/20 and standard-scale the features.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Train/test splits of the features and target.
    scaler : StandardScaler
        The fitted scaler used to transform the features.
    """
    X = df[AUDIO_FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


# =============================================================
# Model training
# =============================================================
@st.cache_resource
def train_all_models(_X_train, _X_test, _y_train, _y_test):
    """Train four regression models and return them with their metrics.

    Models compared:
      1. Linear Regression – simple baseline
      2. Random Forest     – bagging ensemble
      3. XGBoost           – gradient boosting (histogram)
      4. LightGBM          – gradient boosting (leaf-wise)

    Returns a dict {name: {"model": fitted_model, "mae": …, "rmse": …, "r2": …}}
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=0,
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            verbose=-1,
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(_X_train, _y_train)
        preds = model.predict(_X_test)

        results[name] = {
            "model": model,
            "mae": mean_absolute_error(_y_test, preds),
            "rmse": np.sqrt(mean_squared_error(_y_test, preds)),
            "r2": r2_score(_y_test, preds),
        }

    return results


# =============================================================
# SHAP explainability
# =============================================================
@st.cache_data
def compute_shap_values(_model, _X_test):
    """Compute SHAP values for the best tree-based model.

    Uses TreeExplainer which is optimized for tree ensembles and
    returns a numpy array of shape (n_samples, n_features).
    """
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(_X_test)
    return shap_values


# =============================================================
# Streamlit UI
# =============================================================
def main():
    """Entry point – builds the full Streamlit interface."""
    st.title("🎵 Spotify Song Popularity Predictor")
    st.markdown(
        "Predict how popular a song will be based on its audio features. "
        "Compare multiple ML models and understand predictions with SHAP."
    )

    # ---- Load data & prepare models ----
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
    results = train_all_models(X_train, X_test, y_train, y_test)

    # ---- Tab layout ----
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Data Overview",
            "🔗 Feature Analysis",
            "🤖 Model Comparison",
            "💡 SHAP Explainability",
            "🎤 Predict Popularity",
        ]
    )

    # =========================================================
    # Tab 1 – Data Overview
    # =========================================================
    with tab1:
        st.header("Dataset Overview")
        st.write(f"**Rows:** {len(df):,}  |  **Columns:** {len(df.columns)}")

        # Show a scrollable sample of the raw data
        st.subheader("Raw Data (first 100 rows)")
        st.dataframe(df.head(100), use_container_width=True)

        # Descriptive statistics give learners a quick sense of scale
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe().round(2), use_container_width=True)

        # Popularity distribution – the target we are trying to predict
        st.subheader("Popularity Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df[TARGET], bins=40, color="mediumseagreen", edgecolor="white")
        ax.set_xlabel("Popularity")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Song Popularity")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # =========================================================
    # Tab 2 – Feature Analysis
    # =========================================================
    with tab2:
        st.header("Feature Correlation Analysis")

        corr = get_correlation_matrix(df)

        # Full correlation heatmap
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=ax,
            square=True,
            linewidths=0.5,
        )
        ax.set_title("Feature Correlation Matrix")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Bar chart of correlations with the target
        st.subheader("Top Features Correlated with Popularity")
        target_corr = (
            corr[TARGET]
            .drop(TARGET)
            .sort_values(key=abs, ascending=False)
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["indianred" if v < 0 else "steelblue" for v in target_corr.values]
        target_corr.plot.barh(ax=ax, color=colors)
        ax.set_xlabel("Pearson Correlation with Popularity")
        ax.set_title("Feature Correlation with Popularity")
        ax.axvline(0, color="grey", linewidth=0.8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Multicollinearity note
        st.info(
            "💡 **Multicollinearity check:** If two features are highly correlated "
            "(|r| > 0.8), one can be dropped to reduce redundancy without losing "
            "predictive power. Tree-based models handle multicollinearity well, "
            "but linear regression can become unstable."
        )

        # Flag highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(AUDIO_FEATURES)):
            for j in range(i + 1, len(AUDIO_FEATURES)):
                r = corr.iloc[i, j]
                if abs(r) > 0.8:
                    high_corr_pairs.append(
                        (AUDIO_FEATURES[i], AUDIO_FEATURES[j], round(r, 3))
                    )
        if high_corr_pairs:
            st.warning(
                "⚠️ Highly correlated pairs (|r| > 0.8): "
                + ", ".join(f"{a} ↔ {b} ({r})" for a, b, r in high_corr_pairs)
            )
        else:
            st.success("✅ No feature pairs exceed |r| > 0.8 — multicollinearity is low.")

    # =========================================================
    # Tab 3 – Model Comparison
    # =========================================================
    with tab3:
        st.header("Model Comparison")
        st.markdown(
            "All models are evaluated on a held-out 20 % test set using "
            "**MAE**, **RMSE**, and **R²**."
        )

        # Build a comparison DataFrame
        metrics_df = pd.DataFrame(
            {
                name: {"MAE": r["mae"], "RMSE": r["rmse"], "R²": r["r2"]}
                for name, r in results.items()
            }
        ).T.round(4)
        metrics_df.index.name = "Model"

        styled = (
            metrics_df.style
            .highlight_min(subset=["MAE", "RMSE"], color="#d4edda")
            .highlight_max(subset=["R²"], color="#d4edda")
        )
        st.dataframe(styled, use_container_width=True)

        # Bar charts for each metric
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, metric in zip(axes, ["MAE", "RMSE", "R²"]):
            metrics_df[metric].plot.bar(ax=ax, color="steelblue", edgecolor="white")
            ax.set_title(metric)
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=30)
        fig.suptitle("Model Performance Comparison", fontsize=14, y=1.02)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Highlight the best model
        best_name = metrics_df["R²"].idxmax()
        st.success(
            f"🏆 **Best model by R²:** {best_name} "
            f"(R² = {metrics_df.loc[best_name, 'R²']:.4f})"
        )

        # Hyperparameter note for learners
        with st.expander("🔧 Gradient Boosting Hyperparameters Explained"):
            st.markdown(
                """
                | Parameter | Description |
                |-----------|-------------|
                | **n_estimators** | Number of boosting rounds (trees). More trees can improve accuracy but risk overfitting. |
                | **max_depth** | Maximum depth of each tree. Deeper trees capture more complex patterns but overfit more easily. |
                | **learning_rate** | Step-size shrinkage applied per round. Lower values need more estimators but generalise better. |
                """
            )

    # =========================================================
    # Tab 4 – SHAP Explainability
    # =========================================================
    with tab4:
        st.header("SHAP Feature Importance")
        st.markdown(
            "SHAP (SHapley Additive exPlanations) reveals **how much each "
            "feature pushes** the prediction up or down."
        )

        # Pick the best tree-based model for SHAP (skip Linear Regression)
        tree_models = {k: v for k, v in results.items() if k != "Linear Regression"}
        best_tree_name = max(tree_models, key=lambda k: tree_models[k]["r2"])
        best_tree_model = tree_models[best_tree_name]["model"]

        st.write(f"Using **{best_tree_name}** (best tree-based model) for SHAP analysis.")

        shap_values = compute_shap_values(best_tree_model, X_test)

        # SHAP summary plot (bee-swarm style)
        st.subheader("SHAP Summary Plot")
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=AUDIO_FEATURES,
            show=False,
        )
        fig = plt.gcf()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close("all")

        # SHAP bar plot – mean absolute SHAP value per feature
        st.subheader("Mean |SHAP| Feature Importance")
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance_df = (
            pd.DataFrame({"Feature": AUDIO_FEATURES, "Mean |SHAP|": mean_abs_shap})
            .sort_values("Mean |SHAP|", ascending=True)
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(importance_df["Feature"], importance_df["Mean |SHAP|"], color="coral")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Feature Importance (SHAP)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # =========================================================
    # Tab 5 – Predict Popularity
    # =========================================================
    with tab5:
        st.header("Predict a Song's Popularity")
        st.markdown("Adjust the sliders to set audio features and see the predicted popularity.")

        # Pick the best overall model for prediction
        best_name = max(results, key=lambda k: results[k]["r2"])
        best_model = results[best_name]["model"]
        st.write(f"Using **{best_name}** for predictions.")

        # Two-column layout keeps the form compact
        col1, col2 = st.columns(2)

        with col1:
            danceability = st.slider("Danceability", 0.0, 1.0, 0.6, 0.01)
            energy = st.slider("Energy", 0.0, 1.0, 0.7, 0.01)
            valence = st.slider("Valence (positiveness)", 0.0, 1.0, 0.5, 0.01)
            acousticness = st.slider("Acousticness", 0.0, 1.0, 0.2, 0.01)
            speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05, 0.01)
            instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0, 0.01)
            liveness = st.slider("Liveness", 0.0, 1.0, 0.15, 0.01)

        with col2:
            loudness = st.slider("Loudness (dB)", -60.0, 0.0, -6.0, 0.5)
            tempo = st.slider("Tempo (BPM)", 50.0, 200.0, 120.0, 1.0)
            duration_ms = st.slider("Duration (ms)", 60000, 420000, 210000, 1000)
            key = st.slider("Key (0–11)", 0, 11, 5)
            mode = st.selectbox("Mode", [0, 1], format_func=lambda x: "Minor" if x == 0 else "Major")
            time_signature = st.selectbox("Time Signature", [3, 4, 5, 6, 7])
            explicit = st.selectbox("Explicit", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        # Assemble the feature vector in the same order as training data
        input_features = np.array(
            [[duration_ms, explicit, danceability, energy, key, loudness,
              mode, speechiness, acousticness, instrumentalness, liveness,
              valence, tempo, time_signature]]
        )

        # Scale using the same scaler fitted during training
        input_scaled = scaler.transform(input_features)

        if st.button("🎶 Predict Popularity", type="primary"):
            prediction = best_model.predict(input_scaled)[0]
            # Clamp to valid range
            prediction = float(np.clip(prediction, 0, 100))

            st.metric(label="Predicted Popularity", value=f"{prediction:.1f} / 100")

            # Simple colour-coded feedback
            if prediction >= 70:
                st.success("🔥 This song is predicted to be a hit!")
            elif prediction >= 40:
                st.info("🎵 Moderate popularity expected.")
            else:
                st.warning("📉 This song may struggle to gain traction.")


# =============================================================
# Run
# =============================================================
if __name__ == "__main__":
    main()
