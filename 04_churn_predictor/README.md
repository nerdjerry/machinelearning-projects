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

## Using Real Data

The app generates synthetic telecom churn data (~26% churn rate). To use real data:

1. Download the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset from Kaggle
2. Place the CSV file at `04_churn_predictor/data/churn.csv`
3. Restart the app
