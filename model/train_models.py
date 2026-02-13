"""
Training script for Loan Status Classification Models
Trains 6 different ML models and saves them to the model/ directory
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, 
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Load dataset
print("Loading loan_data.csv...")
df = pd.read_csv('loan_data.csv')

print(f"Dataset shape: {df.shape}")
print(f"Target variable distribution:\n{df['loan_status'].value_counts()}")

# Data Preprocessing
print("\n" + "="*60)
print("DATA PREPROCESSING")
print("="*60)

# Handle missing values
print(f"\nMissing values before handling:\n{df.isnull().sum()}")
df = df.dropna()
print(f"\nDataset shape after dropping NAs: {df.shape}")

# Remove age outliers (person_age = 144 is unrealistic)
df = df[df['person_age'] <= 120]
print(f"Dataset shape after removing age outliers: {df.shape}")

# Separate features and target
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Train target distribution:\n{y_train.value_counts()}")
print(f"Test target distribution:\n{y_test.value_counts()}")

# Store test set for later use in Streamlit app
test_data = pd.DataFrame(X_test, columns=X.columns)
test_data['loan_status'] = y_test.values
test_data.to_csv('test_data.csv', index=False)
print("\nTest data saved to test_data.csv")

# ============================================================
# MODEL TRAINING AND EVALUATION
# ============================================================

results = []

# 1. Logistic Regression
print("\n" + "="*60)
print("1. LOGISTIC REGRESSION")
print("="*60)
lr = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_pred_proba_lr = lr.predict_proba(X_test)[:, 1]

acc_lr = accuracy_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_pred_proba_lr)
prec_lr = precision_score(y_test, y_pred_lr, zero_division=0)
rec_lr = recall_score(y_test, y_pred_lr, zero_division=0)
f1_lr = f1_score(y_test, y_pred_lr, zero_division=0)
mcc_lr = matthews_corrcoef(y_test, y_pred_lr)

print(f"Accuracy:  {acc_lr:.4f}")
print(f"AUC:       {auc_lr:.4f}")
print(f"Precision: {prec_lr:.4f}")
print(f"Recall:    {rec_lr:.4f}")
print(f"F1 Score:  {f1_lr:.4f}")
print(f"MCC:       {mcc_lr:.4f}")

results.append({
    'Model': 'Logistic Regression',
    'Accuracy': acc_lr,
    'AUC': auc_lr,
    'Precision': prec_lr,
    'Recall': rec_lr,
    'F1': f1_lr,
    'MCC': mcc_lr
})

joblib.dump(lr, 'logistic_regression.joblib')
print("✓ Model saved to logistic_regression.joblib")

# 2. Decision Tree Classifier
print("\n" + "="*60)
print("2. DECISION TREE CLASSIFIER")
print("="*60)
dt = DecisionTreeClassifier(max_depth=10, random_state=42, min_samples_split=10)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
y_pred_proba_dt = dt.predict_proba(X_test)[:, 1]

acc_dt = accuracy_score(y_test, y_pred_dt)
auc_dt = roc_auc_score(y_test, y_pred_proba_dt)
prec_dt = precision_score(y_test, y_pred_dt, zero_division=0)
rec_dt = recall_score(y_test, y_pred_dt, zero_division=0)
f1_dt = f1_score(y_test, y_pred_dt, zero_division=0)
mcc_dt = matthews_corrcoef(y_test, y_pred_dt)

print(f"Accuracy:  {acc_dt:.4f}")
print(f"AUC:       {auc_dt:.4f}")
print(f"Precision: {prec_dt:.4f}")
print(f"Recall:    {rec_dt:.4f}")
print(f"F1 Score:  {f1_dt:.4f}")
print(f"MCC:       {mcc_dt:.4f}")

results.append({
    'Model': 'Decision Tree',
    'Accuracy': acc_dt,
    'AUC': auc_dt,
    'Precision': prec_dt,
    'Recall': rec_dt,
    'F1': f1_dt,
    'MCC': mcc_dt
})

joblib.dump(dt, 'decision_tree.joblib')
print("✓ Model saved to decision_tree.joblib")

# 3. K-Nearest Neighbors
print("\n" + "="*60)
print("3. K-NEAREST NEIGHBORS CLASSIFIER")
print("="*60)
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
y_pred_proba_knn = knn.predict_proba(X_test)[:, 1]

acc_knn = accuracy_score(y_test, y_pred_knn)
auc_knn = roc_auc_score(y_test, y_pred_proba_knn)
prec_knn = precision_score(y_test, y_pred_knn, zero_division=0)
rec_knn = recall_score(y_test, y_pred_knn, zero_division=0)
f1_knn = f1_score(y_test, y_pred_knn, zero_division=0)
mcc_knn = matthews_corrcoef(y_test, y_pred_knn)

print(f"Accuracy:  {acc_knn:.4f}")
print(f"AUC:       {auc_knn:.4f}")
print(f"Precision: {prec_knn:.4f}")
print(f"Recall:    {rec_knn:.4f}")
print(f"F1 Score:  {f1_knn:.4f}")
print(f"MCC:       {mcc_knn:.4f}")

results.append({
    'Model': 'KNN',
    'Accuracy': acc_knn,
    'AUC': auc_knn,
    'Precision': prec_knn,
    'Recall': rec_knn,
    'F1': f1_knn,
    'MCC': mcc_knn
})

joblib.dump(knn, 'knn.joblib')
print("✓ Model saved to knn.joblib")

# 4. Naive Bayes (Gaussian)
print("\n" + "="*60)
print("4. NAIVE BAYES CLASSIFIER (GAUSSIAN)")
print("="*60)
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
y_pred_proba_nb = nb.predict_proba(X_test)[:, 1]

acc_nb = accuracy_score(y_test, y_pred_nb)
auc_nb = roc_auc_score(y_test, y_pred_proba_nb)
prec_nb = precision_score(y_test, y_pred_nb, zero_division=0)
rec_nb = recall_score(y_test, y_pred_nb, zero_division=0)
f1_nb = f1_score(y_test, y_pred_nb, zero_division=0)
mcc_nb = matthews_corrcoef(y_test, y_pred_nb)

print(f"Accuracy:  {acc_nb:.4f}")
print(f"AUC:       {auc_nb:.4f}")
print(f"Precision: {prec_nb:.4f}")
print(f"Recall:    {rec_nb:.4f}")
print(f"F1 Score:  {f1_nb:.4f}")
print(f"MCC:       {mcc_nb:.4f}")

results.append({
    'Model': 'Naive Bayes',
    'Accuracy': acc_nb,
    'AUC': auc_nb,
    'Precision': prec_nb,
    'Recall': rec_nb,
    'F1': f1_nb,
    'MCC': mcc_nb
})

joblib.dump(nb, 'naive_bayes.joblib')
print("✓ Model saved to naive_bayes.joblib")

# 5. Random Forest (Ensemble)
print("\n" + "="*60)
print("5. RANDOM FOREST CLASSIFIER (ENSEMBLE)")
print("="*60)
rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, min_samples_split=10)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

acc_rf = accuracy_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
prec_rf = precision_score(y_test, y_pred_rf, zero_division=0)
rec_rf = recall_score(y_test, y_pred_rf, zero_division=0)
f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)
mcc_rf = matthews_corrcoef(y_test, y_pred_rf)

print(f"Accuracy:  {acc_rf:.4f}")
print(f"AUC:       {auc_rf:.4f}")
print(f"Precision: {prec_rf:.4f}")
print(f"Recall:    {rec_rf:.4f}")
print(f"F1 Score:  {f1_rf:.4f}")
print(f"MCC:       {mcc_rf:.4f}")

results.append({
    'Model': 'Random Forest',
    'Accuracy': acc_rf,
    'AUC': auc_rf,
    'Precision': prec_rf,
    'Recall': rec_rf,
    'F1': f1_rf,
    'MCC': mcc_rf
})

joblib.dump(rf, 'random_forest.joblib')
print("✓ Model saved to random_forest.joblib")

# 6. XGBoost
print("\n" + "="*60)
print("6. XGBOOST CLASSIFIER (ENSEMBLE)")
print("="*60)
xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
y_pred_proba_xgb = xgb.predict_proba(X_test)[:, 1]

acc_xgb = accuracy_score(y_test, y_pred_xgb)
auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
prec_xgb = precision_score(y_test, y_pred_xgb, zero_division=0)
rec_xgb = recall_score(y_test, y_pred_xgb, zero_division=0)
f1_xgb = f1_score(y_test, y_pred_xgb, zero_division=0)
mcc_xgb = matthews_corrcoef(y_test, y_pred_xgb)

print(f"Accuracy:  {acc_xgb:.4f}")
print(f"AUC:       {auc_xgb:.4f}")
print(f"Precision: {prec_xgb:.4f}")
print(f"Recall:    {rec_xgb:.4f}")
print(f"F1 Score:  {f1_xgb:.4f}")
print(f"MCC:       {mcc_xgb:.4f}")

results.append({
    'Model': 'XGBoost',
    'Accuracy': acc_xgb,
    'AUC': auc_xgb,
    'Precision': prec_xgb,
    'Recall': rec_xgb,
    'F1': f1_xgb,
    'MCC': mcc_xgb
})

joblib.dump(xgb, 'xgboost.joblib')
print("✓ Model saved to xgboost.joblib")

# ============================================================
# SAVE RESULTS
# ============================================================

print("\n" + "="*60)
print("SUMMARY RESULTS")
print("="*60)

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# Save results to CSV
results_df.to_csv('model_results.csv', index=False)
print("\n✓ Results saved to model_results.csv")

# Save scaler and label encoders for later use in app
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(label_encoders, 'label_encoders.joblib')
joblib.dump(X.columns.tolist(), 'feature_names.joblib')
print("✓ Scaler, label encoders, and feature names saved")

print("\n" + "="*60)
print("✓ TRAINING COMPLETED SUCCESSFULLY")
print("="*60)
print("\nAll models are ready for deployment!")
