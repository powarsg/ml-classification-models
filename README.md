# Loan Status Binary Classification Models

Build an interactive Streamlit web application to demonstrate multiple classification models predicting loan approval status.

## a. Problem Statement [1 mark]

**Objective:** Predict whether a loan application will be approved or not based on applicant's demographic and financial information.

**Classification Type:** Binary Classification (Loan Approved: Yes/No)

**Dataset:** Loan application data containing 14 features capturing applicant information, loan details, and credit history.

**Target Variable:** `loan_status` (0 = Not Approved, 1 = Approved)

**Business Impact:** This model helps financial institutions make faster and more consistent lending decisions by automatically scoring loan applications.

---

## b. Dataset Description [1 mark]

### Dataset Overview
- **Source:** Public loan dataset (45,000 loan records)
- **Total Records:** 45,000 (after preprocessing: 44,995)
- **Total Features:** 13 (after removing target variable)
- **Target Distribution:** 
  - Class 0 (Not Approved): ~77.8%
  - Class 1 (Approved): ~22.2%
  - Imbalanced dataset - requires careful evaluation metrics

### Features

#### Demographic Features
1. **person_age** (Numeric) - Age of loan applicant (range: 21-120 years)
2. **person_gender** (Categorical) - Gender (male/female)
3. **person_education** (Categorical) - Education level (High School, Associate, Bachelor, Master, Doctorate)
4. **person_income** (Numeric) - Annual income in USD (~$12K - $600K+)
5. **person_emp_exp** (Numeric) - Employment experience in years

#### Home Ownership
6. **person_home_ownership** (Categorical) - Home ownership status (RENT, OWN, MORTGAGE, OTHER)

#### Loan Details
7. **loan_amnt** (Numeric) - Loan amount requested ($1K - $35K)
8. **loan_intent** (Categorical) - Purpose of loan (PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION)
9. **loan_int_rate** (Numeric) - Interest rate (%)

#### Financial Indicators
10. **loan_percent_income** (Numeric) - Loan amount as percentage of annual income
11. **cb_person_cred_hist_length** (Numeric) - Credit history length in years
12. **credit_score** (Numeric) - Credit score (range: ~500-789)
13. **previous_loan_defaults_on_file** (Categorical) - Previous default history (Yes/No)

### Data Preprocessing
- **Missing Values:** No missing values detected
- **Outlier Treatment:** Removed records with person_age > 120 (data quality issues)
- **Feature Scaling:** StandardScaler applied to all numeric features
- **Categorical Encoding:** LabelEncoder applied to all categorical variables
- **Train-Test Split:** 80-20 split with stratification (35,996 train, 8,999 test)

---

## c. Models Used [6 marks - 1 mark for all metrics of each model]

### Model Implementations

All 6 ML models were trained on the same preprocessed dataset with consistent train-test splits.

#### 1. **Logistic Regression**
- **Type:** Linear probabilistic classifier
- **Hyperparameters:** max_iter=1000, solver='lbfgs'
- **Purpose:** Baseline model for binary classification
- **Strengths:** Interpretable, fast, good for linearly separable data
- **Use Case:** When model interpretability is crucial

#### 2. **Decision Tree Classifier**
- **Type:** Tree-based greedy algorithm
- **Hyperparameters:** max_depth=10, min_samples_split=10, random_state=42
- **Purpose:** Non-linear decision boundaries
- **Strengths:** Handles non-linear patterns, interpretable decision rules
- **Use Case:** When feature interactions matter

#### 3. **K-Nearest Neighbors (KNN)**
- **Type:** Instance-based/lazy learning
- **Hyperparameters:** n_neighbors=5, metric='euclidean'
- **Purpose:** Local decision boundaries based on similarity
- **Strengths:** No training needed, adapts to local patterns
- **Use Case:** When data has clear local clusters

#### 4. **Naive Bayes (Gaussian)**
- **Type:** Probabilistic generative model
- **Hyperparameters:** Default Gaussian implementation
- **Purpose:** Probabilistic classification with conditional independence
- **Strengths:** Fast training, good with high-dimensional data
- **Use Case:** When computational efficiency is important

#### 5. **Random Forest (Ensemble)**
- **Type:** Ensemble of decision trees
- **Hyperparameters:** n_estimators=100, max_depth=15, min_samples_split=10, random_state=42
- **Purpose:** Reduces overfitting through averaging multiple trees
- **Strengths:** Robust, handles non-linear patterns, feature importance
- **Use Case:** Production models requiring balanced performance

#### 6. **XGBoost (Extreme Gradient Boosting)**
- **Type:** Sequential ensemble boosting
- **Hyperparameters:** n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
- **Purpose:** Optimal gradient boosting with regularization
- **Strengths:** State-of-the-art performance, handles imbalanced data
- **Use Case:** When maximum predictive accuracy is needed

---

## Models Evaluation Metrics Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8924 | 0.9486 | 0.7614 | 0.7515 | 0.7564 | 0.6874 |
| Decision Tree | 0.9204 | 0.9597 | 0.8863 | 0.7365 | 0.8045 | 0.7602 |
| KNN | 0.8900 | 0.9241 | 0.7678 | 0.7240 | 0.7452 | 0.6756 |
| Naive Bayes | 0.7312 | 0.9392 | 0.4525 | 0.9985 | 0.6228 | 0.5433 |
| Random Forest (Ensemble) | 0.9245 | 0.9720 | 0.8906 | 0.7530 | 0.8160 | 0.7733 |
| XGBoost (Ensemble) | 0.9284 | 0.9755 | 0.8835 | 0.7810 | 0.8291 | 0.7864 |

### Evaluation Metrics Definitions

1. **Accuracy:** Overall correctness of predictions (correct predictions / total predictions)
   - Range: 0 to 1 (higher is better)
   - Limitation: Can be misleading with imbalanced data

2. **AUC (Area Under ROC Curve):** Probability that model ranks a random positive example higher than a random negative example
   - Range: 0 to 1 (higher is better, 0.5 = random)
   - Advantage: Threshold-independent, good for imbalanced data

3. **Precision:** Of predicted positives, what fraction are actually positive
   - Formula: TP / (TP + FP)
   - Importance: Minimizes false positives

4. **Recall:** Of actual positives, what fraction did the model identify
   - Formula: TP / (TP + FN)
   - Importance: Minimizes false negatives

5. **F1 Score:** Harmonic mean of Precision and Recall
   - Formula: 2 × (Precision × Recall) / (Precision + Recall)
   - Use: Single metric balancing both precision and recall

6. **Matthews Correlation Coefficient (MCC):** Correlation between predicted and actual values
   - Range: -1 to +1 (+1 = perfect, 0 = random, -1 = inverse)
   - Advantage: Works well with imbalanced datasets, more informative than accuracy

---

## d. Observations on Model Performance [3 marks]

### 1. Logistic Regression
- **Performance:** Moderate baseline performance (89.24% accuracy)
- **Strengths:** Provides probability estimates with calibrated probabilities
- **Weaknesses:** Assumes linear relationship; struggles with complex non-linear patterns in loan data
- **AUC:** 0.9486 indicates good discrimination ability despite moderate accuracy
- **Trade-off:** High precision (0.76) but moderate recall (0.75) - balanced approach
- **Insight:** Useful for interpretability; coefficients show which features influence approval

### 2. Decision Tree
- **Performance:** Good accuracy (92.04%) with balanced precision-recall
- **Strengths:** Naturally handles feature interactions; easily interpretable decision rules
- **Weaknesses:** Max depth of 10 prevents overfitting but may underfit complex patterns
- **AUC:** 0.9597 shows excellent ranking ability
- **Trade-off:** High precision (0.89) but lower recall (0.74) - conservative predictions
- **Insight:** Rules like "if credit_score > X and loan_amount < Y then approve" are directly interpretable

### 3. K-Nearest Neighbors
- **Performance:** Good accuracy (89.00%) with strong AUC (0.9241)
- **Strengths:** Non-parametric approach captures local patterns without strong assumptions
- **Weaknesses:** Sensitive to feature scaling (mitigated by StandardScaler); slow predictions on large datasets
- **Trade-off:** Similar precision-recall balance to Logistic Regression
- **Computational Cost:** k=5 uses 5 nearest neighbors; higher k would be more robust but slower
- **Insight:** Performs well when similar loan profiles tend to have similar outcomes

### 4. Naive Bayes
- **Performance:** Lower accuracy (73.12%) but excellent AUC (0.9392)
- **Strengths:** Very fast training and prediction; handles high-dimensional data naturally
- **Weaknesses:** Conditional independence assumption violated in loan data (e.g., income and credit score are correlated)
- **Critical Issue:** Extremely high recall (99.85%) but very low precision (0.45)
  - **Implication:** Approves almost all loans (high false positive rate)
  - **Problem:** Not suitable for real-world deployment where false approvals are costly
- **AUC Paradox:** High AUC (0.9392) despite low accuracy indicates good ranking but poor calibration
- **Insight:** Probabilistic nature useful for ranking, but decision threshold needs adjustment for practical use

### 5. Random Forest
- **Performance:** Excellent accuracy (92.45%) with state-of-the-art AUC (0.9720)
- **Strengths:** 
  - Robust ensemble reduces overfitting through averaging 100 trees
  - Handles both linear and non-linear patterns
  - Provides feature importance scores for business insights
- **Weaknesses:** Less interpretable than single decision tree; requires more memory
- **Precision-Recall:** High precision (0.89) with reasonable recall (0.75)
- **Advantage over Decision Tree:** Better generalization due to ensemble averaging
- **Feature Importance:** Can identify which features most influence loan decisions
- **Insight:** Production-grade model balancing performance and robustness

### 6. XGBoost
- **Performance:** **Best overall performance** (92.84% accuracy, 0.9755 AUC)
- **Strengths:**
  - Sequential boosting corrects previous trees' errors
  - Built-in regularization prevents overfitting
  - Handles imbalanced data naturally
  - Fastest inference time among ensemble methods
- **Precision-Recall:** Highest F1 score (0.8291) balances both metrics excellently
- **MCC Score:** Best MCC (0.7864) indicates best overall correlation
- **Advantages:**
  - Converges faster with fewer trees needed
  - Better handles non-linear interactions
  - More stable across different data samples
- **Trade-off:** More complex hyperparameters, less interpretable than Random Forest
- **Insight:** Recommended for production deployment when maximum accuracy is required

### Comparative Analysis

**Best Model Selection Criteria:**

1. **For Maximum Accuracy:** XGBoost (92.84%)
2. **For Balance:** Random Forest (excellent accuracy + interpretability)
3. **For Interpretability:** Decision Tree or Logistic Regression
4. **For Fast Inference:** Naive Bayes (despite lower accuracy)
5. **For Business Risk:** Random Forest/XGBoost (high precision reduces false approvals)

**Why XGBoost Wins:**
- Achieves highest AUC (0.9755) → Best discrimination between approved/rejected
- Achieves highest F1 (0.8291) → Best overall precision-recall balance
- Achieves highest MCC (0.7864) → Best correlation with actual outcomes
- Handles imbalanced data (77.8% vs 22.2%) naturally
- Provides both predictions and probability estimates

**Lesson from Naive Bayes:**
- High AUC doesn't guarantee good practical performance
- Precision matters when false positives are expensive (e.g., bad loan approvals)
- Threshold adjustment can improve poor precision, but drastically reduces recall

**Imbalanced Data Impact:**
- Dataset heavily skewed toward rejections (77.8% rejected)
- Recall metric becomes critical (not missing valid approvals)
- AUC and MCC provide better evaluation than accuracy alone
- Ensemble methods (Random Forest, XGBoost) naturally handle imbalance

---

## How to Use

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train models (optional):**
   ```bash
   python train_models.py
   ```

3. **Run Streamlit app:**
   ```bash
   streamlit run app.py
   ```

4. **Access the app:**
   - Open browser to `http://localhost:8501`

### Features

✅ **Model Selection Dropdown** - Choose any of 6 models to evaluate  
✅ **CSV Data Upload** - Upload test data (optional)  
✅ **Performance Metrics Display** - View accuracy, AUC, precision, recall, F1, MCC  
✅ **Confusion Matrix Visualization** - Interactive heatmap of predictions  
✅ **Classification Report** - Detailed per-class metrics  
✅ **Comparison Charts** - Bar charts comparing all 6 models  
✅ **Sample Predictions** - Preview predictions on test data  

---

## Repository Structure

```
project-folder/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── model/
    ├── train_models.py             # Training script
    ├── loan_data.csv               # Original training dataset
    ├── logistic_regression.joblib  # Trained Logistic Regression model
    ├── decision_tree.joblib        # Trained Decision Tree model
    ├── knn.joblib                  # Trained KNN model
    ├── naive_bayes.joblib          # Trained Naive Bayes model
    ├── random_forest.joblib        # Trained Random Forest model
    ├── xgboost.joblib              # Trained XGBoost model
    ├── scaler.joblib               # StandardScaler for feature normalization
    ├── label_encoders.joblib       # Label encoders for categorical features
    ├── feature_names.joblib        # Feature names list
    ├── test_data.csv               # Test dataset with predictions
    └── model_results.csv           # Evaluation metrics for all models
```

---

## Technical Stack

- **ML Framework:** scikit-learn, XGBoost
- **Web Framework:** Streamlit
- **Data Processing:** pandas, numpy
- **Visualization:** Plotly, matplotlib, seaborn
- **Serialization:** joblib
- **Python Version:** 3.8+

---

## Author

BITS Pilani - ML Assignment 2  
Submitted: February 15, 2026

---

## Conclusion

This project demonstrates the complete ML workflow from data preprocessing to model deployment. XGBoost emerged as the best-performing model with 92.84% accuracy and 0.9755 AUC, making it the recommended model for production loan approval prediction. The interactive Streamlit application allows users to explore model performance, upload custom data, and generate predictions dynamically.

Key learnings:
- Ensemble methods consistently outperform single models
- Imbalanced data requires careful metric selection (AUC, MCC, F1 over Accuracy)
- Feature scaling and encoding significantly impact model performance
- Metrics should align with business objectives (precision vs. recall trade-off)
