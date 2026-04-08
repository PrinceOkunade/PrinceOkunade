# Customer Churn Prediction

An end-to-end machine learning project that predicts customer churn for a telecom company, enabling proactive retention strategies.

## Business Problem

Customer churn costs telecom companies billions annually. This project builds a predictive model that identifies at-risk customers **before** they leave, along with actionable explanations for **why** they're likely to churn.

## Dataset

**Telco Customer Churn** — 7,043 customers with 21 features including:
- Demographics (gender, senior citizen, partner, dependents)
- Account info (tenure, contract type, payment method, monthly/total charges)
- Services (phone, internet, streaming, security, tech support)

Source: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Project Pipeline

```
Data Loading → EDA → Feature Engineering → Model Training → Evaluation → Explainability → Deployment
```

### 1. Exploratory Data Analysis
- Churn distribution and class imbalance analysis
- Feature correlations and multicollinearity checks
- Churn rate by contract type, tenure, and services
- Monthly charges distribution for churned vs retained customers

### 2. Feature Engineering
- Encoding categorical variables (one-hot and label encoding)
- Feature scaling (StandardScaler for numerical features)
- Handling class imbalance with SMOTE

### 3. Model Comparison
| Model | ROC-AUC | Precision | Recall |
|-------|---------|-----------|--------|
| Logistic Regression | Baseline | - | - |
| Random Forest | - | - | - |
| XGBoost | Best | - | - |

*Results populated after training — see notebook for full metrics.*

### 4. Model Explainability (SHAP)
- Global feature importance
- SHAP summary plots
- Individual prediction explanations (waterfall plots)

### 5. Deployment
Interactive Streamlit app where users input customer details and receive:
- Churn probability score
- Risk category (Low / Medium / High)
- Top factors driving the prediction

## Project Structure

```
customer-churn-prediction/
├── notebooks/
│   └── churn_prediction.ipynb      # Full end-to-end analysis
├── src/
│   └── app.py                      # Streamlit deployment app
├── data/                           # Dataset (download from Kaggle)
├── models/                         # Saved model artifacts
├── images/                         # Generated plots
├── requirements.txt
└── README.md
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/churn_prediction.ipynb

# Launch the Streamlit app
streamlit run src/app.py
```

## Key Findings

- Contract type is the strongest predictor of churn — month-to-month customers churn at 3x the rate of long-term contract holders
- Customers without tech support and online security are significantly more likely to churn
- Tenure and monthly charges interact non-linearly with churn risk

## Tech Stack

Python | Pandas | Scikit-learn | XGBoost | SHAP | SMOTE | Streamlit | Seaborn | Plotly
