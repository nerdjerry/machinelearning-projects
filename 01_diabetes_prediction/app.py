"""
Diabetes Prediction with Model Comparison
==========================================
A Streamlit application that demonstrates key machine-learning concepts
using the Pima Indians Diabetes dataset:

  1. Impact of feature scaling (StandardScaler vs MinMaxScaler)
  2. Comparing Logistic Regression, SVM, Random Forest & Gradient Boosting
  3. Hyperparameter tuning with GridSearchCV
  4. Evaluation via confusion matrix, ROC-AUC and F1 score

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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

# Suppress convergence warnings for cleaner Streamlit output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================
# 2. Page configuration
# =============================================================
st.set_page_config(
    page_title="Diabetes Prediction – ML Comparison",
    page_icon="🩺",
    layout="wide",
)

# =============================================================
# 3. Data loading helpers
# =============================================================

# Column names that mirror the real Pima Indians Diabetes dataset
FEATURE_COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET_COL = "Outcome"


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the diabetes dataset from CSV or generate realistic synthetic data.

    The function first looks for ``data/diabetes.csv`` relative to this
    script.  If the file is not found it falls back to generating ~768
    rows of synthetic data whose distributions closely match the real
    Pima Indians Diabetes dataset.
    """
    csv_path = os.path.join(os.path.dirname(__file__), "data", "diabetes.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Make sure the expected columns are present
        if set(FEATURE_COLS + [TARGET_COL]).issubset(df.columns):
            return df

    # ----- Synthetic data generation ----
    # We use statistics that approximate the real dataset so the demo
    # behaves realistically even without the original CSV.
    rng = np.random.RandomState(42)
    n = 768  # same size as the original dataset

    data = {
        "Pregnancies": rng.poisson(lam=3.8, size=n).clip(0, 17),
        "Glucose": rng.normal(loc=121, scale=32, size=n).clip(0, 200).astype(int),
        "BloodPressure": rng.normal(loc=69, scale=19, size=n).clip(0, 122).astype(int),
        "SkinThickness": rng.normal(loc=21, scale=16, size=n).clip(0, 99).astype(int),
        "Insulin": rng.normal(loc=80, scale=115, size=n).clip(0, 846).astype(int),
        "BMI": rng.normal(loc=32, scale=8, size=n).clip(0, 67).round(1),
        "DiabetesPedigreeFunction": rng.exponential(scale=0.47, size=n).clip(
            0.078, 2.42
        ).round(3),
        "Age": rng.normal(loc=33, scale=12, size=n).clip(21, 81).astype(int),
    }

    df = pd.DataFrame(data)

    # Generate a realistic binary outcome correlated with Glucose and BMI
    logit = (
        -8
        + 0.03 * df["Glucose"]
        + 0.05 * df["BMI"]
        + 0.02 * df["Age"]
        + 0.15 * df["Pregnancies"]
    )
    prob = 1 / (1 + np.exp(-logit))
    df[TARGET_COL] = rng.binomial(1, prob)

    return df


# =============================================================
# 4. Model training helpers
# =============================================================

def get_models() -> dict:
    """Return a dictionary of model name → scikit-learn estimator.

    Each model uses sensible defaults.  SVC uses ``probability=True``
    so we can compute ROC-AUC later.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
    }


@st.cache_data
def evaluate_scaling(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Train Logistic Regression under three scaling strategies and compare.

    Returns a DataFrame with columns [Scaling, Accuracy, F1, ROC-AUC].
    This shows learners how feature scaling affects distance-based and
    gradient-based models.
    """
    scalers = {
        "No Scaling": None,
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
    }

    rows = []
    for name, scaler in scalers.items():
        X_tr = X_train.copy()
        X_te = X_test.copy()

        if scaler is not None:
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_tr, y_train)
        preds = model.predict(X_te)
        proba = model.predict_proba(X_te)[:, 1]

        rows.append(
            {
                "Scaling": name,
                "Accuracy": round(accuracy_score(y_test, preds), 4),
                "F1": round(f1_score(y_test, preds), 4),
                "ROC-AUC": round(roc_auc_score(y_test, proba), 4),
            }
        )

    return pd.DataFrame(rows)


@st.cache_data
def compare_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Fit every model on scaled data and return a comparison DataFrame.

    Returns columns [Model, Accuracy, F1, ROC-AUC].
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    rows = []
    for name, model in get_models().items():
        model.fit(X_tr, y_train)
        preds = model.predict(X_te)
        proba = model.predict_proba(X_te)[:, 1]
        rows.append(
            {
                "Model": name,
                "Accuracy": round(accuracy_score(y_test, preds), 4),
                "F1": round(f1_score(y_test, preds), 4),
                "ROC-AUC": round(roc_auc_score(y_test, proba), 4),
            }
        )

    return pd.DataFrame(rows)


@st.cache_data
def run_grid_search(
    _X_train: np.ndarray,
    _y_train: np.ndarray,
    model_name: str,
) -> dict:
    """Run GridSearchCV for the chosen model and return the results.

    A small but representative parameter grid is used so the search
    completes quickly in a demo setting.
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(_X_train)

    # Define focused grids that run in a few seconds
    grids = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {"C": [0.01, 0.1, 1, 10]},
        },
        "SVM": {
            "model": SVC(probability=True, random_state=42),
            "params": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, None],
            },
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
            },
        },
    }

    cfg = grids[model_name]
    search = GridSearchCV(
        cfg["model"],
        cfg["params"],
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )
    search.fit(X_tr, _y_train)

    return {
        "best_params": search.best_params_,
        "best_score": round(search.best_score_, 4),
        "cv_results": pd.DataFrame(search.cv_results_)[
            ["params", "mean_test_score", "rank_test_score"]
        ].sort_values("rank_test_score"),
    }


# =============================================================
# 5. Plotting helpers
# =============================================================

def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix"
) -> plt.Figure:
    """Return a matplotlib Figure containing a styled confusion matrix."""
    fig, ax = plt.subplots(figsize=(4, 3))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_roc_curves(
    X_test: np.ndarray, y_test: np.ndarray, fitted_models: dict
) -> plt.Figure:
    """Return a Figure with overlaid ROC curves for all fitted models."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, model in fitted_models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random (AUC = 0.5)")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def plot_model_comparison_bar(results_df: pd.DataFrame) -> plt.Figure:
    """Return a grouped bar chart comparing model metrics."""
    melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=melted, x="Model", y="Score", hue="Metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# =============================================================
# 6. Main application
# =============================================================

def main() -> None:
    """Entry point for the Streamlit application."""

    # --- Header ---
    st.title("🩺 Diabetes Prediction – Model Comparison")
    st.markdown(
        """
        This interactive demo walks through a **complete ML pipeline** on the
        Pima Indians Diabetes dataset.  You will explore:

        | Concept | What you'll see |
        |---|---|
        | **Feature Scaling** | How StandardScaler / MinMaxScaler affect accuracy |
        | **Model Comparison** | Logistic Regression, SVM, Random Forest, Gradient Boosting |
        | **Hyperparameter Tuning** | GridSearchCV with cross-validation |
        | **Evaluation Metrics** | Confusion matrix, ROC-AUC, F1 score |
        | **Live Prediction** | Enter patient features and get a real-time prediction |
        """
    )
    st.divider()

    # --- Load & split data ---
    df = load_data()
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # Sidebar: let the user control the train/test split ratio
    st.sidebar.header("⚙️ Settings")
    test_size = st.sidebar.slider(
        "Test set size", min_value=0.1, max_value=0.5, value=0.2, step=0.05
    )
    random_state = st.sidebar.number_input(
        "Random state", min_value=0, max_value=999, value=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state), stratify=y
    )

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Data Overview",
            "⚖️ Feature Scaling",
            "🤖 Model Comparison",
            "🔧 Hyperparameter Tuning",
            "🔮 Make a Prediction",
        ]
    )

    # ---------------------------------------------------------
    # Tab 1 – Data Overview
    # ---------------------------------------------------------
    with tab1:
        st.subheader("Dataset at a Glance")

        col_shape, col_balance = st.columns(2)
        with col_shape:
            st.metric("Rows", df.shape[0])
            st.metric("Features", len(FEATURE_COLS))
        with col_balance:
            counts = df[TARGET_COL].value_counts()
            st.metric("Non-diabetic (0)", int(counts.get(0, 0)))
            st.metric("Diabetic (1)", int(counts.get(1, 0)))

        with st.expander("Show raw data"):
            st.dataframe(df, use_container_width=True)

        with st.expander("Descriptive statistics"):
            st.dataframe(df.describe().round(2), use_container_width=True)

        # Correlation heatmap
        st.subheader("Feature Correlations")
        fig_corr, ax_corr = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            df.corr(numeric_only=True),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax_corr,
        )
        ax_corr.set_title("Correlation Matrix")
        fig_corr.tight_layout()
        st.pyplot(fig_corr)

    # ---------------------------------------------------------
    # Tab 2 – Feature Scaling Comparison
    # ---------------------------------------------------------
    with tab2:
        st.subheader("How Feature Scaling Affects Logistic Regression")
        st.markdown(
            """
            Many ML algorithms (especially Logistic Regression and SVM) are
            sensitive to the **scale** of input features.  Here we compare
            three strategies applied to Logistic Regression:

            * **No Scaling** – raw feature values
            * **StandardScaler** – zero mean, unit variance
            * **MinMaxScaler** – values mapped to [0, 1]
            """
        )

        scaling_df = evaluate_scaling(X_train, X_test, y_train, y_test)
        st.dataframe(scaling_df.set_index("Scaling"), use_container_width=True)

        # Bar chart of scaling comparison
        melted_scaling = scaling_df.melt(
            id_vars="Scaling", var_name="Metric", value_name="Score"
        )
        fig_sc, ax_sc = plt.subplots(figsize=(7, 3.5))
        sns.barplot(data=melted_scaling, x="Scaling", y="Score", hue="Metric", ax=ax_sc)
        ax_sc.set_ylim(0, 1)
        ax_sc.set_title("Scaling Impact on Logistic Regression")
        fig_sc.tight_layout()
        st.pyplot(fig_sc)

    # ---------------------------------------------------------
    # Tab 3 – Model Comparison
    # ---------------------------------------------------------
    with tab3:
        st.subheader("Side-by-Side Model Comparison")
        st.markdown(
            """
            All models below are trained on **StandardScaler**-transformed
            features and evaluated on the held-out test set.
            """
        )

        results_df = compare_models(X_train, X_test, y_train, y_test)
        st.dataframe(results_df.set_index("Model"), use_container_width=True)

        # Grouped bar chart
        st.pyplot(plot_model_comparison_bar(results_df))

        # ----- Confusion matrices & ROC curves -----
        st.subheader("Confusion Matrices")

        # Re-fit each model on scaled data so we can plot per-model outputs
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)

        fitted = {}
        cm_cols = st.columns(len(get_models()))
        for idx, (name, model) in enumerate(get_models().items()):
            model.fit(X_tr_sc, y_train)
            fitted[name] = model
            preds = model.predict(X_te_sc)
            with cm_cols[idx]:
                st.pyplot(plot_confusion_matrix(y_test, preds, title=name))

        # ROC curves overlaid
        st.subheader("ROC Curves")
        st.pyplot(plot_roc_curves(X_te_sc, y_test, fitted))

    # ---------------------------------------------------------
    # Tab 4 – Hyperparameter Tuning
    # ---------------------------------------------------------
    with tab4:
        st.subheader("Hyperparameter Tuning with GridSearchCV")
        st.markdown(
            """
            Select a model below and run a **5-fold cross-validated grid
            search** optimized for ROC-AUC.  The table shows every
            combination tried, ranked by performance.
            """
        )

        model_choice = st.selectbox(
            "Choose a model to tune",
            list(get_models().keys()),
        )

        if st.button("Run Grid Search"):
            with st.spinner("Searching hyperparameter space …"):
                gs = run_grid_search(X_train, y_train, model_choice)

            st.success(
                f"Best CV ROC-AUC: **{gs['best_score']}**  •  "
                f"Best params: `{gs['best_params']}`"
            )
            st.dataframe(gs["cv_results"], use_container_width=True)

    # ---------------------------------------------------------
    # Tab 5 – Live Prediction
    # ---------------------------------------------------------
    with tab5:
        st.subheader("Enter Patient Features")
        st.markdown(
            "Adjust the sliders and click **Predict** to see the model's "
            "diagnosis.  The prediction uses a **Random Forest** trained on "
            "the full training set with StandardScaler."
        )

        # Two-column layout for the input sliders
        c1, c2 = st.columns(2)
        with c1:
            pregnancies = st.slider("Pregnancies", 0, 17, 1)
            glucose = st.slider("Glucose (mg/dL)", 0, 200, 120)
            blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 122, 70)
            skin_thickness = st.slider("Skin Thickness (mm)", 0, 99, 20)
        with c2:
            insulin = st.slider("Insulin (μU/mL)", 0, 846, 80)
            bmi = st.slider("BMI", 0.0, 67.0, 32.0, step=0.1)
            dpf = st.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.47, step=0.01)
            age = st.slider("Age", 21, 81, 33)

        if st.button("Predict"):
            # Scale with the same scaler used during training
            scaler = StandardScaler()
            scaler.fit(X_train)
            patient = np.array(
                [[pregnancies, glucose, blood_pressure, skin_thickness,
                  insulin, bmi, dpf, age]]
            )
            patient_scaled = scaler.transform(patient)

            # Train a Random Forest for the prediction
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(scaler.transform(X_train), y_train)
            prediction = rf.predict(patient_scaled)[0]
            probability = rf.predict_proba(patient_scaled)[0]

            st.divider()
            if prediction == 1:
                st.error(
                    f"⚠️ **Diabetic** — model confidence: {probability[1]:.1%}"
                )
            else:
                st.success(
                    f"✅ **Non-diabetic** — model confidence: {probability[0]:.1%}"
                )


# =============================================================
# 7. Run
# =============================================================
if __name__ == "__main__":
    main()
