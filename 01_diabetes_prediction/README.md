# 🩺 Diabetes Prediction with Model Comparison

**Difficulty: 2/10 — Perfect starting point**

## What It Is

Use the Pima Indians Diabetes dataset to build and compare multiple classification models. The focus is on the engineering side — showing how preprocessing, feature scaling, and hyperparameter tuning each improve performance step by step.

## Tech Stack

- Python, pandas, scikit-learn, Matplotlib/Seaborn, Streamlit

## What You Learn

- Impact of feature scaling (StandardScaler vs MinMaxScaler) on model accuracy
- Comparing Logistic Regression, SVM, Random Forest, and Gradient Boosting
- Hyperparameter tuning with GridSearchCV
- Evaluating with confusion matrix, ROC-AUC, and F1 score

## How to Run

```bash
# From the project root directory
pip install -r requirements.txt
streamlit run 01_diabetes_prediction/app.py
```

## Data Sources

The app tries these sources in order:

1. **Local CSV** — Place the CSV file at `01_diabetes_prediction/data/diabetes.csv`
2. **Kaggle** — Automatically downloads the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) if Kaggle credentials are configured (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars)
3. **Synthetic data** — Generates ~768 realistic rows as a fallback
