# Loan Status Binary Classification Models

## a. Problem Statement

**Objective:** Predict loan approval status based on applicant demographics and financial information.

**Classification Type:** Binary Classification (0 = Not Approved, 1 = Approved)

**Target Variable:** `loan_status`

**Business Context:** Automated loan decision-making system for financial institutions.

---

## b. Dataset Description

**Dataset:** 45,000 loan application records (45,000 → 44,995 after preprocessing)

**Features:** 13 features + 1 target variable

**Target Distribution:**
- Class 0 (Not Approved): 77.8% (35,000 records)
- Class 1 (Approved): 22.2% (10,000 records)
- Imbalanced dataset

**Features:**

| Category | Feature | Type | Details |
|----------|---------|------|---------|
| Demographic | person_age | Numeric | 21-120 years (outliers removed) |
| Demographic | person_gender | Categorical | male/female |
| Demographic | person_education | Categorical | High School, Associate, Bachelor, Master, Doctorate |
| Demographic | person_income | Numeric | Annual income (~$12K - $600K) |
| Demographic | person_emp_exp | Numeric | Employment experience (years) |
| Home | person_home_ownership | Categorical | RENT, OWN, MORTGAGE, OTHER |
| Loan Details | loan_amnt | Numeric | Requested loan amount ($1K-$35K) |
| Loan Details | loan_intent | Categorical | PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION |
| Loan Details | loan_int_rate | Numeric | Interest rate (%) |
| Financial | loan_percent_income | Numeric | Loan as % of income |
| Financial | cb_person_cred_hist_length | Numeric | Credit history length (years) |
| Financial | credit_score | Numeric | Credit score (500-789) |
| Financial | previous_loan_defaults_on_file | Categorical | Yes/No |

**Preprocessing Steps:**
- Removed age outliers (person_age > 120)
- Applied StandardScaler to numeric features
- Applied LabelEncoder to categorical features
- Train-Test Split: 80-20 with stratification (35,996 train, 8,999 test)

---

## c. Models Used

### 6 Classification Models Implemented

1. **Logistic Regression** - Linear probabilistic baseline model
2. **Decision Tree Classifier** - Tree-based non-linear model (max_depth=10)
3. **K-Nearest Neighbors** - Instance-based model (k=5)
4. **Naive Bayes** - Probabilistic Gaussian classifier
5. **Random Forest** - Ensemble of 100 decision trees
6. **XGBoost** - Gradient boosted ensemble classifier

### Evaluation Metrics Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8927 | 0.9507 | 0.7630 | 0.7500 | 0.7564 | 0.6876 |
| Decision Tree | 0.9204 | 0.9605 | 0.8877 | 0.7350 | 0.8042 | 0.7601 |
| KNN | 0.8884 | 0.9292 | 0.7672 | 0.7150 | 0.7402 | 0.6699 |
| Naive Bayes | 0.7282 | 0.9403 | 0.4498 | 1.0000 | 0.6205 | 0.5410 |
| Random Forest (Ensemble) | 0.9211 | 0.9742 | 0.8781 | 0.7490 | 0.8084 | 0.7629 |
| **XGBoost (Ensemble)** | **0.9311** | **0.9788** | **0.8868** | **0.7910** | **0.8362** | **0.7948** |

**Metrics Definitions:**
- **Accuracy:** Overall correctness (0-1, higher is better)
- **AUC:** Discrimination ability between classes (0-1, higher is better)
- **Precision:** True positives / All positive predictions (reduces false approvals)
- **Recall:** True positives / All actual positives (captures valid approvals)
- **F1 Score:** Harmonic mean of precision and recall
- **MCC:** Matthews Correlation Coefficient (-1 to +1, works well with imbalanced data)

---

## d. Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Moderate baseline (89.27% accuracy). Interpretable with good AUC (0.9507), but linear assumptions limit non-linear pattern capture. Suitable for simple baseline comparisons. |
| **Decision Tree** | Good performance (92.04% accuracy, 0.9605 AUC). Interpretable rules and feature interactions handled naturally. Conservative predictions (0.735 recall) may reject valid loans. Max depth=10 balances complexity and interpretability. |
| **kNN** | Good accuracy (88.84%, 0.9292 AUC) capturing local patterns without assumptions. Computationally expensive for large datasets and sensitive to feature scaling. Similar precision-recall balance to Logistic Regression. |
| **Naive Bayes** | Poor performance (72.82% accuracy). Critical issue: Extremely high recall (100%) but very low precision (0.45%) - approves almost all loans, unacceptable for production. Conditional independence assumption violated in loan data. High AUC (0.9403) doesn't guarantee practical performance. |
| **Random Forest** | Excellent performance (92.11% accuracy, 0.9742 AUC). Robust ensemble handling non-linear patterns with feature importance insights. Better generalization than single decision tree. 100 trees reduce overfitting through averaging. Strong production candidate. |
| **XGBoost 🏆** | **Best overall model (93.11% accuracy, 0.9788 AUC, 0.8362 F1, 0.7948 MCC).** Sequential boosting corrects errors iteratively. Handles imbalanced data natively. Best precision-recall balance. Optimal for production deployment. More stable predictions than Random Forest. |

### Key Learning - Imbalanced Data:
- Accuracy alone is misleading (77.8% vs 22.2% class distribution)
- AUC, F1, and MCC are more informative metrics for imbalanced datasets
- Ensemble methods (Random Forest, XGBoost) naturally handle class imbalance better
- Trade-off between precision and recall critical in loan approval decisions

### Model Selection by Priority:
- **Maximum Accuracy:** XGBoost (93.11%)
- **Balanced Performance:** Random Forest (92.11% accuracy + interpretability)
- **Interpretability:** Decision Tree (92.04%) or Logistic Regression (89.27%)
- **Production Use:** XGBoost (best AUC 0.9788 + F1 0.8362 + MCC 0.7948)

---

## Repository Structure

```
project-folder/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── model/
    ├── train_models.py             # Training script
    ├── loan_data.csv               # Original dataset
    ├── logistic_regression.joblib  # Trained model
    ├── decision_tree.joblib        # Trained model
    ├── knn.joblib                  # Trained model
    ├── naive_bayes.joblib          # Trained model
    ├── random_forest.joblib        # Trained model
    ├── xgboost.joblib              # Trained model
    ├── scaler.joblib               # StandardScaler
    ├── label_encoders.joblib       # Categorical encoders
    ├── feature_names.joblib        # Feature names
    ├── test_data.csv               # Test dataset
    └── model_results.csv           # Evaluation metrics
```

## How to Use

**Train Models (Optional):**
```bash
cd model/
python3 train_models.py
```

**Run Streamlit App:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

**Access Live App:**
- Local: `http://localhost:8501`
- Cloud: Deployed on Streamlit Community Cloud

## Streamlit App Features

✅ Model selection dropdown (6 models)  
✅ CSV upload for test data  
✅ Performance metrics display  
✅ Confusion matrix visualization  
✅ Classification report  
✅ Model comparison charts  
✅ Sample predictions  


