# Credit Card Fraud Detection

A Streamlit app to explore a credit-card transactions dataset, handle the extreme
class imbalance and compare classification models for fraud detection.

## Features

- Upload a CSV from the sidebar (columns `Time`, `V1`–`V28`, `Amount`, `Class`, the layout
  of the public credit-card fraud dataset). The small sample file in this repository can be
  uploaded for a quick test.
- **Imbalance handling**: SMOTE, ADASYN, RandomOverSampler, RandomUnderSampler,
  SMOTETomek, SMOTEENN (via `imbalanced-learn`).
- **Models**: Logistic Regression, Random Forest, Gradient Boosting.
- **Evaluation**: accuracy, precision, recall, F1, ROC-AUC, average precision,
  confusion matrix and precision–recall curve, with plain-language interpretation.
- Interactive charts with Plotly.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `creditcard - Copie.csv` | Small sample dataset (about 430 rows) |
| `requirements.txt` | Dependencies |
| `bibliotheques_utilisees.txt` | Notes on the libraries used |

## Stack

Python, Streamlit, pandas, NumPy, scikit-learn, imbalanced-learn, Plotly, matplotlib, seaborn.

## Limitations

- The bundled sample is far too small to draw conclusions; use a full dataset for real results.
- Resampling is applied inside the app on the data you provide; no trained model is saved.
- Educational project, not a production fraud-detection system.
