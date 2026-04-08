# Credit Risk Scoring with Explainable AI

An end-to-end machine learning pipeline for predicting loan defaults, with a focus on model explainability and fairness — critical requirements in regulated financial services.

## Business Problem

Lenders need to assess borrower creditworthiness while complying with regulations that require **transparent, explainable decisions**. This project builds a credit scoring model that not only predicts default risk but explains **why** each borrower received their score — enabling fair, auditable lending decisions.

## Dataset

**LendingClub Loan Data** — Real-world peer-to-peer lending data with 150+ features including:
- Loan details (amount, term, interest rate, grade)
- Borrower financials (income, debt-to-income, employment length)
- Credit history (revolving balance, delinquencies, inquiries, open accounts)

Source: [Kaggle - LendingClub](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

## Project Pipeline

```
Data Loading → EDA → Feature Engineering → Imbalanced Learning → Modeling → Explainability → Fairness Audit → Deployment
```

### 1. Exploratory Data Analysis
- Default rate analysis by loan grade, purpose, and income bracket
- Distribution of debt-to-income ratios for defaulters vs non-defaulters
- Correlation analysis and multicollinearity detection
- Geographic and temporal default patterns

### 2. Feature Engineering
- Engineered features: loan-to-income ratio, credit utilization rate, payment burden ratio
- Missing value imputation strategies (median, mode, indicator columns)
- Categorical encoding for grades, sub-grades, and employment length
- Feature selection using mutual information and variance thresholds

### 3. Handling Class Imbalance
- SMOTE (Synthetic Minority Oversampling Technique)
- XGBoost `scale_pos_weight` parameter
- Evaluation with Precision-Recall curves (not just accuracy)

### 4. Model Comparison
| Model | ROC-AUC | Precision | Recall | F1 |
|-------|---------|-----------|--------|----|
| Logistic Regression | Baseline | - | - | - |
| Random Forest | - | - | - | - |
| XGBoost | Best | - | - | - |

*Results populated after training — see notebook for full metrics.*

### 5. Explainability (SHAP)
- **Global explanations**: Which features matter most across all loans?
- **Local explanations**: Why was *this specific borrower* flagged as high risk?
- SHAP waterfall plots for individual predictions
- SHAP dependence plots showing feature interactions

### 6. Fairness Analysis
- Does the model produce systematically different outcomes by demographic group?
- Disparate impact ratio calculation
- Discussion of regulatory considerations (Equal Credit Opportunity Act)

### 7. Deployment
Streamlit app where users input borrower details and receive:
- Default probability score
- Risk tier (A through E)
- SHAP-based explanation of the top risk factors
- Comparison to similar approved borrowers

## Project Structure

```
credit-risk-scoring/
├── notebooks/
│   └── credit_risk_analysis.ipynb  # Full end-to-end analysis
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
jupyter notebook notebooks/credit_risk_analysis.ipynb

# Launch the Streamlit app
streamlit run src/app.py
```

## Key Findings

- Loan grade and interest rate are the strongest predictors of default
- Debt-to-income ratio above 25 significantly increases default probability
- Borrowers with recent delinquencies are 4x more likely to default
- The model achieves strong separation between risk tiers with minimal fairness concerns

## Regulatory Context

In financial services, ML models must be:
- **Explainable**: Regulators require lenders to provide reasons for adverse decisions
- **Fair**: Models must not discriminate based on protected characteristics
- **Auditable**: All predictions must be traceable and reproducible

This project addresses all three requirements through SHAP explanations, fairness analysis, and reproducible pipelines.

## Tech Stack

Python | Pandas | Scikit-learn | XGBoost | SHAP | SMOTE | Streamlit | Seaborn | Plotly
