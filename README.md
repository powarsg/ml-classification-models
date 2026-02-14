# Loan Status Binary Classification Models

## a. Problem Statement [1 mark]

**Objective:** Predict loan approval status based on applicant demographics and financial information.

**Classification Type:** Binary Classification (0 = Not Approved, 1 = Approved)

**Target Variable:** `loan_status`

**Business Context:** Automated loan decision-making system for financial institutions.

---

## b. Dataset Description [1 mark]

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

## c. Models Used [6 marks - 1 mark for all metrics of each model]

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
| Logistic Regression | 0.8924 | 0.9486 | 0.7614 | 0.7515 | 0.7564 | 0.6874 |
| Decision Tree | 0.9204 | 0.9597 | 0.8863 | 0.7365 | 0.8045 | 0.7602 |
| KNN | 0.8900 | 0.9241 | 0.7678 | 0.7240 | 0.7452 | 0.6756 |
| Naive Bayes | 0.7312 | 0.9392 | 0.4525 | 0.9985 | 0.6228 | 0.5433 |
| Random Forest (Ensemble) | 0.9245 | 0.9720 | 0.8906 | 0.7530 | 0.8160 | 0.7733 |
| XGBoost (Ensemble) | 0.9284 | 0.9755 | 0.8835 | 0.7810 | 0.8291 | 0.7864 |

**Metrics Definitions:**
- **Accuracy:** Overall correctness (0-1, higher is better)
- **AUC:** Discrimination ability between classes (0-1, higher is better)
- **Precision:** True positives / All positive predictions (reduces false approvals)
- **Recall:** True positives / All actual positives (captures valid approvals)
- **F1 Score:** Harmonic mean of precision and recall
- **MCC:** Matthews Correlation Coefficient (-1 to +1, works well with imbalanced data)

---

## d. Observations on Model Performance [3 marks]

### 1. Logistic Regression
- **Performance:** Moderate baseline (89.24% accuracy)
- **Strengths:** Interpretable, provides probability calibration
- **Weaknesses:** Linear assumptions; cannot capture non-linear loan approval patterns
- **Key Metric:** AUC 0.9486 shows good discrimination despite moderate accuracy

### 2. Decision Tree
- **Performance:** Good (92.04% accuracy, 0.9597 AUC)
- **Strengths:** Interpretable rules; handles feature interactions naturally
- **Weaknesses:** Conservative predictions (0.74 recall) may reject valid loans
- **Insights:** Max depth=10 balances complexity and interpretability

### 3. K-Nearest Neighbors
- **Performance:** Good (89.00% accuracy, 0.9241 AUC)
- **Strengths:** Captures local patterns without assumptions
- **Weaknesses:** Computationally expensive for large datasets; sensitive to feature scaling
- **Trade-off:** Similar precision-recall balance to Logistic Regression

### 4. Naive Bayes
- **Performance:** Lower (73.12% accuracy)
- **Critical Issue:** Extremely high recall (99.85%) but low precision (0.45)
  - Approves almost all loans → unacceptable for production
- **Lesson:** High AUC doesn't guarantee practical performance
- **Problem:** Conditional independence assumption violated in loan data (income and credit score are correlated)

### 5. Random Forest
- **Performance:** Excellent (92.45% accuracy, 0.9720 AUC)
- **Strengths:** Robust ensemble; handles non-linear patterns; provides feature importance
- **Advantages:** Better generalization than single decision tree
- **Insight:** 100 trees reduce overfitting through averaging

### 6. XGBoost ⭐ **Best Model**
- **Performance:** Best overall (92.84% accuracy, 0.9755 AUC, 0.8291 F1)
- **Strengths:** Sequential boosting corrects errors; handles imbalanced data natively
- **Precision-Recall:** Best balance with highest F1 and MCC (0.7864)
- **Recommendation:** Optimal for production deployment
- **Advantages over Random Forest:** Better convergence, more stable predictions

### Comparative Insights

**Model Selection by Priority:**
- **Maximum Accuracy:** XGBoost (92.84%)
- **Balanced Performance:** Random Forest (accuracy + interpretability)
- **Interpretability:** Decision Tree or Logistic Regression
- **Production Use:** XGBoost (best AUC + F1 + MCC)

**Key Learning - Imbalanced Data:**
- Accuracy alone is misleading (77.8% vs 22.2% class distribution)
- AUC, F1, MCC are more informative
- Ensemble methods naturally handle class imbalance better

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


