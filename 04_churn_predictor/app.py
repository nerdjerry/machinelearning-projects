"""
Salesforce / Telecom Churn Predictor – Streamlit Application
=============================================================
An educational ML app that walks through the full churn-prediction pipeline:
  1. Exploratory data analysis
  2. Feature engineering from raw telecom data
  3. Handling class imbalance with SMOTE
  4. Decision-threshold tuning (precision ↔ recall trade-off)
  5. Model comparison & live predictions

Run with:  streamlit run app.py
"""

# ──────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "churn.csv")
RANDOM_STATE = 42
TARGET = "Churn"

# ──────────────────────────────────────────────────────────────
# 1. Data loading / synthetic generation
# ──────────────────────────────────────────────────────────────


def _generate_synthetic_data(n: int = 7000) -> pd.DataFrame:
    """Create a realistic synthetic telecom-churn dataset.

    The generator uses conditional probabilities so that customers with
    month-to-month contracts, no tech support, and high monthly charges
    churn at higher rates – mirroring real-world patterns.

    Returns a DataFrame with ~26 % churn rate.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    genders = rng.choice(["Male", "Female"], n)
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n, p=[0.30, 0.70])

    # Tenure in months (1-72).  Short-tenure customers churn more.
    tenure = rng.exponential(scale=32, size=n).clip(1, 72).astype(int)

    phone_service = rng.choice(["Yes", "No"], n, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], n, p=[0.42, 0.58]),
    )

    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]
    )

    def _internet_dependent(yes_prob: float) -> np.ndarray:
        """Helper: feature available only when internet service exists."""
        return np.where(
            internet_service == "No",
            "No internet service",
            rng.choice(["Yes", "No"], n, p=[yes_prob, 1 - yes_prob]),
        )

    online_security = _internet_dependent(0.29)
    online_backup = _internet_dependent(0.34)
    device_protection = _internet_dependent(0.34)
    tech_support = _internet_dependent(0.29)
    streaming_tv = _internet_dependent(0.38)
    streaming_movies = _internet_dependent(0.39)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]
    )
    paperless = rng.choice(["Yes", "No"], n, p=[0.60, 0.40])
    payment = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        n,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    # Monthly charges: fiber users pay more.
    base_charge = np.where(
        internet_service == "Fiber optic",
        rng.normal(80, 15, n),
        np.where(internet_service == "DSL", rng.normal(55, 12, n), rng.normal(25, 8, n)),
    ).clip(18, 120)
    monthly_charges = np.round(base_charge, 2)

    # Total charges = monthly × tenure (with a little noise).
    total_charges = np.round(
        monthly_charges * tenure * rng.uniform(0.95, 1.05, n), 2
    )

    # --- Churn probability (logistic model) ---
    # Higher churn for: short tenure, month-to-month, fiber, no support,
    # electronic-check payment, high monthly charges.
    logit = (
        -2.0
        + (-0.03 * tenure)
        + (1.2 * (contract == "Month-to-month"))
        + (-0.7 * (contract == "Two year"))
        + (0.5 * (internet_service == "Fiber optic"))
        + (-0.4 * (online_security == "Yes"))
        + (-0.4 * (tech_support == "Yes"))
        + (0.3 * (paperless == "Yes"))
        + (0.5 * (payment == "Electronic check"))
        + (0.01 * monthly_charges)
        + (0.3 * senior)
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = np.where(rng.random(n) < prob, "Yes", "No")

    df = pd.DataFrame(
        {
            "customerID": [f"C{str(i).zfill(5)}" for i in range(1, n + 1)],
            "gender": genders,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn,
        }
    )
    return df


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load churn data from CSV, falling back to synthetic generation."""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        # TotalCharges may arrive as string with blanks – coerce to numeric.
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df.dropna(subset=["TotalCharges"], inplace=True)
    else:
        df = _generate_synthetic_data()
    return df


# ──────────────────────────────────────────────────────────────
# 2. Feature engineering
# ──────────────────────────────────────────────────────────────


@st.cache_data
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features that capture domain knowledge about churn drivers.

    Engineered columns
    ------------------
    tenure_bin          : categorical bucket (New / Mid / Loyal / Long-term)
    AvgMonthlyCharge    : TotalCharges / tenure  (spending consistency)
    ChargeRatio         : MonthlyCharges / (TotalCharges + 1)
    NumServices         : count of optional services the customer subscribes to
    HasInternet         : binary flag for any internet service
    IsAutoPayment       : binary flag for automatic payment methods
    """
    out = df.copy()

    # Tenure buckets – new customers churn at much higher rates.
    out["tenure_bin"] = pd.cut(
        out["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["New", "Mid", "Loyal", "Long-term"],
    )

    # Average monthly charge over the customer's lifetime.
    out["AvgMonthlyCharge"] = np.round(out["TotalCharges"] / out["tenure"].clip(1), 2)

    # Ratio: current monthly charge relative to total spend (high = recent joiner).
    out["ChargeRatio"] = np.round(
        out["MonthlyCharges"] / (out["TotalCharges"] + 1), 4
    )

    # Count of optional services (OnlineSecurity … StreamingMovies).
    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    out["NumServices"] = out[service_cols].apply(
        lambda row: (row == "Yes").sum(), axis=1
    )

    # Binary convenience flags.
    out["HasInternet"] = (out["InternetService"] != "No").astype(int)
    out["IsAutoPayment"] = (
        out["PaymentMethod"].isin(
            ["Bank transfer (automatic)", "Credit card (automatic)"]
        )
    ).astype(int)

    return out


# ──────────────────────────────────────────────────────────────
# 3. Preprocessing for modelling
# ──────────────────────────────────────────────────────────────


@st.cache_data
def preprocess(df: pd.DataFrame):
    """Encode categoricals, scale numerics, split into train/test.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names, scaler
    """
    drop_cols = ["customerID", "tenure_bin"]
    model_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Separate target.
    y = (model_df[TARGET] == "Yes").astype(int)
    X = model_df.drop(columns=[TARGET])

    # Label-encode binary / ordinal columns; one-hot encode the rest.
    X = pd.get_dummies(X, drop_first=True)

    feature_names = X.columns.tolist()

    # Train / test split (stratified to preserve class ratio).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Standard-scale numeric features.
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

    return X_train, X_test, y_train.reset_index(drop=True), y_test.reset_index(drop=True), feature_names, scaler


# ──────────────────────────────────────────────────────────────
# 4. Model training helpers
# ──────────────────────────────────────────────────────────────


@st.cache_data
def train_models(_X_train, _y_train, use_smote: bool = False):
    """Train Logistic Regression, Random Forest, and Gradient Boosting.

    Parameters
    ----------
    _X_train, _y_train : training data (underscore prefix to skip hashing).
    use_smote : whether to apply SMOTE oversampling before training.

    Returns dict of {name: fitted model}.
    """
    X_tr = _X_train.copy()
    y_tr = _y_train.copy()

    if use_smote:
        sm = SMOTE(random_state=RANDOM_STATE)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE
        ),
    }

    for name, model in models.items():
        model.fit(X_tr, y_tr)

    return models


@st.cache_data
def evaluate_models(_models, _X_test, _y_test, threshold: float = 0.5):
    """Evaluate each model at a given decision threshold.

    Returns a DataFrame with Accuracy, Precision, Recall, F1, and ROC-AUC.
    """
    rows = []
    for name, model in _models.items():
        probs = model.predict_proba(_X_test)[:, 1]
        preds = (probs >= threshold).astype(int)
        rows.append(
            {
                "Model": name,
                "Accuracy": round(accuracy_score(_y_test, preds), 4),
                "Precision": round(precision_score(_y_test, preds, zero_division=0), 4),
                "Recall": round(recall_score(_y_test, preds, zero_division=0), 4),
                "F1": round(f1_score(_y_test, preds, zero_division=0), 4),
                "ROC-AUC": round(roc_auc_score(_y_test, probs), 4),
            }
        )
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# 5. Streamlit UI
# ──────────────────────────────────────────────────────────────


def main():
    """Entry point for the Streamlit application."""

    st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")
    st.title("📉 Telecom Customer Churn Predictor")
    st.caption(
        "An educational walkthrough: data → features → imbalance handling → "
        "threshold tuning → prediction."
    )

    # ── Load & engineer ──────────────────────────────────────
    raw_df = load_data()
    df = engineer_features(raw_df)
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess(df)

    # ── Tabs ─────────────────────────────────────────────────
    tabs = st.tabs(
        [
            "📊 Data Overview",
            "🔧 Feature Engineering",
            "⚖️ Imbalanced Data",
            "🎯 Threshold Tuning",
            "🔮 Predict Churn",
        ]
    )

    # ============================================================
    # TAB 1 – Data Overview
    # ============================================================
    with tabs[0]:
        st.header("Raw Data & Churn Distribution")

        churn_pct = (raw_df[TARGET] == "Yes").mean() * 100
        st.markdown(
            f"The dataset contains **{len(raw_df):,}** customers with "
            f"**{churn_pct:.1f} %** churn rate. "
            "This imbalance is typical in telecom: most customers stay."
        )

        # Show a sample of the raw data.
        with st.expander("🔍 View raw data sample", expanded=False):
            st.dataframe(raw_df.head(100), use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Churn Distribution")
            fig, ax = plt.subplots(figsize=(4, 3))
            counts = raw_df[TARGET].value_counts()
            colors = ["#2ecc71", "#e74c3c"]
            ax.bar(counts.index, counts.values, color=colors, edgecolor="white")
            for i, v in enumerate(counts.values):
                ax.text(i, v + 40, f"{v:,}", ha="center", fontweight="bold", fontsize=10)
            ax.set_ylabel("Customers")
            ax.set_title("Churn vs Retained")
            sns.despine()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col2:
            st.subheader("Key Numeric Stats")
            st.dataframe(
                raw_df[["tenure", "MonthlyCharges", "TotalCharges"]].describe().round(2),
                use_container_width=True,
            )

        # Churn rate by contract type – a quick insight.
        st.subheader("Churn Rate by Contract Type")
        fig, ax = plt.subplots(figsize=(5, 3))
        churn_by_contract = (
            raw_df.groupby("Contract")[TARGET]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .sort_values(ascending=False)
        )
        churn_by_contract.plot.barh(ax=ax, color="#3498db", edgecolor="white")
        ax.set_xlabel("Churn Rate (%)")
        ax.set_title("Month-to-month contracts churn the most")
        sns.despine()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ============================================================
    # TAB 2 – Feature Engineering
    # ============================================================
    with tabs[1]:
        st.header("Feature Engineering")
        st.markdown(
            """
            Raw columns (tenure, charges) aren't always enough. **Feature
            engineering** creates new signals that help models find patterns:

            | Feature | Logic | Why it helps |
            |---|---|---|
            | `tenure_bin` | Bucket tenure into New / Mid / Loyal / Long-term | Captures non-linear churn decay |
            | `AvgMonthlyCharge` | TotalCharges ÷ tenure | Spending consistency signal |
            | `ChargeRatio` | MonthlyCharges ÷ (TotalCharges + 1) | High ratio → short tenure |
            | `NumServices` | Count of opted-in add-on services | More services → stickier customer |
            | `HasInternet` | Binary: has any internet service | Internet users churn differently |
            | `IsAutoPayment` | Binary: auto-pay method | Manual payers leave more often |
            """
        )

        with st.expander("🔍 View engineered features sample"):
            eng_cols = [
                "customerID", "tenure", "tenure_bin", "MonthlyCharges",
                "TotalCharges", "AvgMonthlyCharge", "ChargeRatio",
                "NumServices", "HasInternet", "IsAutoPayment", "Churn",
            ]
            st.dataframe(df[eng_cols].head(50), use_container_width=True)

        # Feature importance from a quick Random Forest.
        st.subheader("Feature Importance (Random Forest)")
        models_plain = train_models(X_train, y_train, use_smote=False)
        rf = models_plain["Random Forest"]
        importances = pd.Series(rf.feature_importances_, index=feature_names)
        top15 = importances.nlargest(15).sort_values()

        fig, ax = plt.subplots(figsize=(6, 4))
        top15.plot.barh(ax=ax, color="#8e44ad", edgecolor="white")
        ax.set_xlabel("Importance")
        ax.set_title("Top 15 Features")
        sns.despine()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.info(
            "💡 **Takeaway:** Contract type, tenure, and monthly charges dominate. "
            "Engineered features like NumServices and ChargeRatio also rank high, "
            "showing that domain knowledge adds modelling value."
        )

    # ============================================================
    # TAB 3 – Imbalanced Data
    # ============================================================
    with tabs[2]:
        st.header("The Accuracy Trap & SMOTE")

        st.markdown(
            """
            ### Why accuracy is the *wrong* metric for churn

            With ~74 % non-churn, a model that **always predicts "No Churn"**
            would score **74 % accuracy** – yet it catches **zero** churners!

            We need metrics that care about the *minority* class:
            - **Precision** – of predicted churners, how many actually churn?
            - **Recall** – of actual churners, how many did we catch?
            - **F1 Score** – harmonic mean of precision & recall.
            """
        )

        # Demonstrate the "dumb" baseline.
        majority_acc = round((y_test == 0).mean() * 100, 1)
        st.metric("\"Always predict No Churn\" accuracy", f"{majority_acc} %")
        st.warning(
            f"☝️ That {majority_acc} % accuracy catches 0 % of churners. "
            "Accuracy alone is misleading on imbalanced data."
        )

        # Train with and without SMOTE.
        models_no_smote = train_models(X_train, y_train, use_smote=False)
        models_smote = train_models(X_train, y_train, use_smote=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Without SMOTE")
            res_no = evaluate_models(models_no_smote, X_test, y_test)
            st.dataframe(res_no, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("With SMOTE")
            res_sm = evaluate_models(models_smote, X_test, y_test)
            st.dataframe(res_sm, use_container_width=True, hide_index=True)

        # SMOTE visualisation: class distribution before & after.
        st.subheader("Class Distribution Before & After SMOTE")
        sm = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = sm.fit_resample(X_train, y_train)

        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        for ax, data, title in [
            (axes[0], y_train, "Before SMOTE"),
            (axes[1], y_res, "After SMOTE"),
        ]:
            counts = pd.Series(data).value_counts().sort_index()
            ax.bar(
                ["No Churn (0)", "Churn (1)"],
                counts.values,
                color=["#2ecc71", "#e74c3c"],
                edgecolor="white",
            )
            ax.set_title(title)
            ax.set_ylabel("Samples")
            for i, v in enumerate(counts.values):
                ax.text(i, v + 20, f"{v:,}", ha="center", fontsize=9)
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.info(
            "💡 **SMOTE** (Synthetic Minority Oversampling Technique) generates "
            "synthetic churn examples so the model trains on a balanced dataset. "
            "Notice recall improves – the model catches more actual churners."
        )

    # ============================================================
    # TAB 4 – Threshold Tuning  (the educational centrepiece)
    # ============================================================
    with tabs[3]:
        st.header("Decision Threshold Tuning")

        st.markdown(
            """
            ### Every classifier outputs a *probability*, not a label

            By default we use **0.50** as the cut-off: if P(churn) ≥ 0.50 →
            predict churn. But that default is rarely optimal.

            - **Lower the threshold** → catch more churners (↑ recall) but
              also more false alarms (↓ precision).
            - **Raise the threshold** → fewer false alarms (↑ precision) but
              you miss real churners (↓ recall).

            > **Business context matters.** If losing a customer costs $500 but
            > a retention offer costs $20, you should favour recall – cast a
            > wider net.

            Use the slider below to explore this trade-off in real time.
            """
        )

        # Use the best model (Gradient Boosting with SMOTE).
        models_smote = train_models(X_train, y_train, use_smote=True)
        best_model = models_smote["Gradient Boosting"]
        probs = best_model.predict_proba(X_test)[:, 1]

        # Precision-recall curve (computed once).
        precisions, recalls, thresholds_pr = precision_recall_curve(y_test, probs)

        # ── Interactive slider ───────────────────────────────
        threshold = st.slider(
            "Decision threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.01,
            help="Drag to see how precision, recall, and F1 change.",
        )

        preds = (probs >= threshold).astype(int)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        acc = accuracy_score(y_test, preds)

        # Live metric cards.
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{prec:.3f}")
        m2.metric("Recall", f"{rec:.3f}")
        m3.metric("F1 Score", f"{f1:.3f}")
        m4.metric("Accuracy", f"{acc:.3f}")

        # ── Precision-Recall curve with draggable marker ─────
        st.subheader("Precision–Recall Curve")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(recalls, precisions, color="#2c3e50", linewidth=2, label="PR curve")

        # Mark the current threshold on the curve.
        # Find the threshold index closest to the slider value.
        idx = np.searchsorted(thresholds_pr, threshold, side="right")
        idx = min(idx, len(precisions) - 1)
        ax.scatter(
            [recalls[idx]], [precisions[idx]],
            color="#e74c3c", s=120, zorder=5,
            label=f"Threshold = {threshold:.2f}",
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision vs Recall — Gradient Boosting (SMOTE)")
        ax.legend(loc="lower left")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        sns.despine()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ── Threshold sweep chart ────────────────────────────
        st.subheader("Metrics Across All Thresholds")
        sweep_thresholds = np.arange(0.05, 0.96, 0.01)
        sweep_data = []
        for t in sweep_thresholds:
            p = (probs >= t).astype(int)
            sweep_data.append(
                {
                    "Threshold": round(t, 2),
                    "Precision": precision_score(y_test, p, zero_division=0),
                    "Recall": recall_score(y_test, p, zero_division=0),
                    "F1": f1_score(y_test, p, zero_division=0),
                }
            )
        sweep_df = pd.DataFrame(sweep_data)

        fig2, ax2 = plt.subplots(figsize=(7, 3.5))
        ax2.plot(sweep_df["Threshold"], sweep_df["Precision"], label="Precision", linewidth=2)
        ax2.plot(sweep_df["Threshold"], sweep_df["Recall"], label="Recall", linewidth=2)
        ax2.plot(sweep_df["Threshold"], sweep_df["F1"], label="F1", linewidth=2, linestyle="--")
        ax2.axvline(threshold, color="#e74c3c", linestyle=":", linewidth=1.5, label=f"Current ({threshold:.2f})")
        ax2.set_xlabel("Threshold")
        ax2.set_ylabel("Score")
        ax2.set_title("How Metrics Change with Threshold")
        ax2.legend(fontsize=8)
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        # Confusion matrix at current threshold.
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, preds)
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            ax=ax3,
        )
        ax3.set_xlabel("Predicted")
        ax3.set_ylabel("Actual")
        ax3.set_title(f"Threshold = {threshold:.2f}")
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

    # ============================================================
    # TAB 5 – Live Prediction
    # ============================================================
    with tabs[4]:
        st.header("Predict Churn for a Customer")
        st.markdown("Fill in the customer details below and click **Predict**.")

        # Use SMOTE Gradient Boosting as production model.
        models_smote = train_models(X_train, y_train, use_smote=True)
        prod_model = models_smote["Gradient Boosting"]

        # Input form.
        with st.form("predict_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior = st.selectbox("Senior Citizen", [0, 1])
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["Yes", "No"])
                tenure = st.slider("Tenure (months)", 1, 72, 12)
                phone = st.selectbox("Phone Service", ["Yes", "No"])
            with c2:
                multi = st.selectbox(
                    "Multiple Lines",
                    ["Yes", "No", "No phone service"],
                )
                internet = st.selectbox(
                    "Internet Service", ["DSL", "Fiber optic", "No"]
                )
                security = st.selectbox(
                    "Online Security", ["Yes", "No", "No internet service"]
                )
                backup = st.selectbox(
                    "Online Backup", ["Yes", "No", "No internet service"]
                )
                protection = st.selectbox(
                    "Device Protection", ["Yes", "No", "No internet service"]
                )
                tech = st.selectbox(
                    "Tech Support", ["Yes", "No", "No internet service"]
                )
            with c3:
                stv = st.selectbox(
                    "Streaming TV", ["Yes", "No", "No internet service"]
                )
                smov = st.selectbox(
                    "Streaming Movies", ["Yes", "No", "No internet service"]
                )
                contract = st.selectbox(
                    "Contract",
                    ["Month-to-month", "One year", "Two year"],
                )
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment = st.selectbox(
                    "Payment Method",
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                )
                monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)
                total = st.number_input(
                    "Total Charges ($)", 0.0, 10000.0, monthly * tenure
                )

            submitted = st.form_submit_button("🔮 Predict Churn")

        if submitted:
            # Build a single-row DataFrame matching the training schema.
            service_cols_vals = [security, backup, protection, tech, stv, smov]
            num_services = sum(1 for s in service_cols_vals if s == "Yes")

            row = pd.DataFrame(
                [
                    {
                        "gender": gender,
                        "SeniorCitizen": senior,
                        "Partner": partner,
                        "Dependents": dependents,
                        "tenure": tenure,
                        "PhoneService": phone,
                        "MultipleLines": multi,
                        "InternetService": internet,
                        "OnlineSecurity": security,
                        "OnlineBackup": backup,
                        "DeviceProtection": protection,
                        "TechSupport": tech,
                        "StreamingTV": stv,
                        "StreamingMovies": smov,
                        "Contract": contract,
                        "PaperlessBilling": paperless,
                        "PaymentMethod": payment,
                        "MonthlyCharges": monthly,
                        "TotalCharges": total,
                        # Engineered features.
                        "AvgMonthlyCharge": round(total / max(tenure, 1), 2),
                        "ChargeRatio": round(monthly / (total + 1), 4),
                        "NumServices": num_services,
                        "HasInternet": int(internet != "No"),
                        "IsAutoPayment": int(
                            payment
                            in [
                                "Bank transfer (automatic)",
                                "Credit card (automatic)",
                            ]
                        ),
                    }
                ]
            )

            # One-hot encode to match training columns.
            row_encoded = pd.get_dummies(row, drop_first=True)
            # Align columns with training set (fill missing with 0).
            row_encoded = row_encoded.reindex(columns=feature_names, fill_value=0)
            # Scale using the same scaler.
            row_scaled = pd.DataFrame(
                scaler.transform(row_encoded), columns=feature_names
            )

            prob = prod_model.predict_proba(row_scaled)[0][1]
            label = "⚠️ **Likely to Churn**" if prob >= 0.5 else "✅ **Likely to Stay**"

            st.divider()
            st.subheader("Prediction Result")
            res1, res2 = st.columns(2)
            res1.metric("Churn Probability", f"{prob:.1%}")
            res2.markdown(f"### {label}")

            if prob >= 0.5:
                st.error(
                    "This customer has a high churn risk. Consider offering a "
                    "retention discount, upgrading their plan, or assigning a "
                    "dedicated support contact."
                )
            else:
                st.success(
                    "This customer appears stable. Continue monitoring their "
                    "usage patterns for early warning signs."
                )


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
