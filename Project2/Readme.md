# Customer Churn Prediction

Predicting which telecom customers are likely to churn, using historical customer data (contract type, tenure, billing, service usage) to flag at-risk customers before they leave.

## Problem Statement

Customer churn is expensive for subscription-based businesses — acquiring a new customer costs significantly more than retaining an existing one. This project builds a classification model that predicts whether a customer will churn (`Yes`/`No`) based on their profile and account details, so that retention efforts can be targeted proactively.

## Dataset

- **Size:** 250,000 customer records, 21 features
- **Target:** `Churn` (Yes/No) — 65.6% No, 34.4% Yes (moderately imbalanced)
- **Features:** Demographics (gender, senior citizen, partner, dependents), account info (tenure, contract type, payment method, billing), and services subscribed (internet, phone, streaming, tech support, etc.)

## Exploratory Data Analysis — Key Findings

| Finding | Detail |
|---|---|
| **Contract type is the strongest churn driver** | Month-to-month customers churn at **51%**, vs. 19% for one-year and only 10% for two-year contracts |
| **New customers churn more** | Customers with 0–12 months tenure churn at 43%, vs. 21% for customers with 48–72 months tenure |
| **Fiber optic users churn more** | 47% churn rate vs. 34% for DSL and 10% for no internet service |
| **Electronic check payers churn more** | Highest churn rate among all payment methods (38%) |
| **Higher-paying customers churn more, but total revenue at risk is flat** | Churned customers pay more per month (avg ₹80 vs ₹63) but leave earlier, so average `TotalCharges` ends up similar between churned and retained customers — two opposing effects cancel out |

## Preprocessing Pipeline

1. Dropped `CustomerID` (unique identifier, no predictive value)
2. Label-encoded binary columns (`Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Gender`, `Churn`)
3. One-hot encoded multi-category columns (`Contract`, `InternetService`, `PaymentMethod`, etc.) with `drop_first=True` to avoid the dummy variable trap
4. Train-test split: 80/20, stratified on `Churn` to preserve class balance
5. Scaled numerical columns (`Tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler` — **fit only on the training set** to prevent data leakage into the test set

## Models Trained & Compared

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 75.8% | 65.0% | 64.0% | 64.5% |
| Random Forest (tuned) | 75.5% | 65.9% | 59.6% | 62.6% |
| XGBoost | 75.7% | 64.7% | 64.5% | 64.6% |

All three models perform similarly, suggesting the underlying relationships in this dataset are largely captured by a small number of strong features (contract type, internet service) rather than requiring complex non-linear modeling.

**Note on Random Forest:** an initial untuned Random Forest scored 99.99% training accuracy but only 74% test accuracy — a clear overfitting signal. Constraining `max_depth`, `min_samples_split`, and `min_samples_leaf` closed this gap to under 1%, producing a properly generalized model.

**Recall was prioritized as a key metric** alongside F1, since a missed churn prediction (false negative) is more costly to the business than a false alarm — a customer who churns unnoticed cannot be retained.

## Feature Importance (best model)

Top predictors: `Contract type` (one-year / two-year) and `InternetService` (fiber optic / none) dominate the model's decisions, accounting for the large majority of predictive signal — consistent with the EDA findings above.

## Project Structure

```
├── EDA.ipynb                       # Exploratory analysis & step-by-step walkthrough
├── churn_prediction_pipeline.py    # Production pipeline (load → preprocess → train → evaluate → export)
├── customer_churn.csv              # Raw dataset
├── churn_model.pkl                 # Saved best-performing trained model
├── scaler.pkl                      # Saved StandardScaler (must be used for any new inference data)
├── test_input.csv                  # Held-out test features (simulates new/unseen customer data)
├── test_predictions.csv            # Model predictions vs. actual outcomes on the test set
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn xgboost joblib
python churn_prediction_pipeline.py
```

This trains all three models, prints a comparison, saves the best model (`churn_model.pkl`) and scaler (`scaler.pkl`), and exports `test_input.csv` / `test_predictions.csv`.

## Using the Saved Model for New Predictions

```python
import joblib

model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

# new_data must go through the same preprocessing/encoding as training data
new_data[['Tenure', 'Monthlycharges', 'Totalcharges']] = scaler.transform(
    new_data[['Tenure', 'Monthlycharges', 'Totalcharges']]
)
predictions = model.predict(new_data)
```

## Business Recommendations

1. **Incentivize longer contracts** — month-to-month customers churn ~5x more than two-year customers; targeted discounts for contract upgrades could meaningfully reduce churn.
2. **Investigate fiber optic service quality/pricing** — this segment churns at nearly 5x the rate of no-internet customers.
3. **Target retention offers at new, high-paying customers** — this group combines high monthly value with the highest churn risk, making it the most impactful segment for proactive retention.

## Future Improvements

- Feature engineering (e.g., interaction terms like `tenure × monthlycharges`) to reduce model over-reliance on just two dominant features
- Hyperparameter tuning via `GridSearchCV` / `RandomizedSearchCV`
- Try SMOTE or class-weighting to address class imbalance more directly
- Deploy as a simple API (Flask/FastAPI) for real-time predictions