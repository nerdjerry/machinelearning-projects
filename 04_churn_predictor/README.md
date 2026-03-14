# 📊 Salesforce Churn Predictor

**Difficulty: 5/10 — Real business framing, imbalanced data**

## What It Is

Take a telecom churn dataset, engineer features from usage patterns and account history, and build a binary classifier. The core challenge is learning to optimize for recall vs precision based on the real business cost of missing a churning customer.

## Tech Stack

- Python, pandas, scikit-learn, imbalanced-learn, Matplotlib/Seaborn, Streamlit

## What You Learn

- Feature engineering from raw transactional and usage data
- Why accuracy is the wrong metric for imbalanced churn datasets
- Decision threshold tuning and the precision-recall tradeoff
- Handling imbalanced data with SMOTE
- Building an interactive prediction app in Streamlit

## How to Run

```bash
# From the project root directory
pip install -r requirements.txt
streamlit run 04_churn_predictor/app.py
```

## Data Sources

The app tries these sources in order:

1. **Local CSV** — Place the CSV file at `04_churn_predictor/data/churn.csv`
2. **Kaggle** — Automatically downloads the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset if Kaggle credentials are configured (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars)
3. **Synthetic data** — Generates ~7 000 realistic rows (~26 % churn rate) as a fallback
