# 🤖 Machine Learning Projects

A collection of four hands-on machine learning projects, each built as a standalone Streamlit application. Projects are ordered by difficulty and designed to demonstrate key ML concepts with well-commented, learner-friendly code.

## Quick Start

```bash
# Install all dependencies
pip install -r requirements.txt

# Run any project
streamlit run 01_diabetes_prediction/app.py
streamlit run 02_netflix_clustering/app.py
streamlit run 03_spotify_popularity/app.py
streamlit run 04_churn_predictor/app.py
```

> Each app generates synthetic data on startup so you can run them immediately. See individual project READMEs for instructions on using real Kaggle datasets.

## Projects

| # | Project | Difficulty | Key Concepts |
|---|---------|-----------|--------------|
| 1 | [🩺 Diabetes Prediction](01_diabetes_prediction/) | 2/10 | Feature scaling, model comparison, GridSearchCV, ROC-AUC |
| 2 | [🎬 Netflix Clustering](02_netflix_clustering/) | 3/10 | KMeans, PCA, elbow method, silhouette score |
| 3 | [🎵 Spotify Popularity](03_spotify_popularity/) | 5/10 | XGBoost, LightGBM, SHAP explainability, correlation analysis |
| 4 | [📊 Churn Predictor](04_churn_predictor/) | 5/10 | SMOTE, threshold tuning, precision-recall tradeoff, imbalanced data |

## Tech Stack

- **Language:** Python 3.10+
- **ML Libraries:** scikit-learn, XGBoost, LightGBM, imbalanced-learn
- **Explainability:** SHAP
- **Visualization:** Matplotlib, Seaborn
- **UI:** Streamlit

## Project Details

### 1. Diabetes Prediction with Model Comparison (Difficulty: 2/10)

Use the Pima Indians Diabetes dataset to build and compare multiple classification models. Demonstrates preprocessing, feature scaling, and hyperparameter tuning with visual evaluation.

**What you learn:** StandardScaler vs MinMaxScaler · Logistic Regression vs SVM vs Random Forest vs Gradient Boosting · GridSearchCV · Confusion matrix, ROC-AUC, F1 score

### 2. Netflix Show Clustering (Difficulty: 3/10)

Group similar Netflix shows using unsupervised learning based on genre, rating, and duration. Visualize clusters to find patterns in the catalogue.

**What you learn:** Categorical encoding · Feature scaling for KMeans · Elbow method & silhouette score · PCA dimensionality reduction

### 3. Spotify Song Popularity Predictor (Difficulty: 5/10)

Predict track popularity from audio features using gradient boosting models. Explain predictions with SHAP values.

**What you learn:** Feature correlation & multicollinearity · XGBoost/LightGBM tuning · SHAP explainability · Regression model comparison

### 4. Salesforce Churn Predictor (Difficulty: 5/10)

Build a churn classifier on telecom data, handling class imbalance and optimizing the decision threshold for business value.

**What you learn:** Feature engineering · Accuracy trap for imbalanced data · SMOTE · Threshold tuning & precision-recall tradeoff
