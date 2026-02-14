"""
Streamlit Web Application for Loan Status Classification
Interactive demo of 6 ML models trained on loan approval prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
)
import plotly.figure_factory as ff
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Loan Classification Models",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .metric-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .header-title {
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<div class='header-title'>💰 Loan Status Classification Models</div>", unsafe_allow_html=True)
st.markdown("""
This interactive application demonstrates **6 ML classification models** trained to predict 
loan approval status. Explore model performance, upload test data, and generate predictions.
""")

st.divider()

# Load models and preprocessors
@st.cache_resource
def load_models():
    models = {
        'Logistic Regression': joblib.load('model/logistic_regression.joblib'),
        'Decision Tree': joblib.load('model/decision_tree.joblib'),
        'KNN': joblib.load('model/knn.joblib'),
        'Naive Bayes': joblib.load('model/naive_bayes.joblib'),
        'Random Forest': joblib.load('model/random_forest.joblib'),
        'XGBoost': joblib.load('model/xgboost.joblib'),
    }
    return models

@st.cache_resource
def load_preprocessors():
    scaler = joblib.load('model/scaler.joblib')
    label_encoders = joblib.load('model/label_encoders.joblib')
    feature_names = joblib.load('model/feature_names.joblib')
    return scaler, label_encoders, feature_names

@st.cache_data
def load_test_data():
    return pd.read_csv('model/test_data.csv')

@st.cache_data
def load_results():
    return pd.read_csv('model/model_results.csv')

# Load all resources
models = load_models()
scaler, label_encoders, feature_names = load_preprocessors()
test_data = load_test_data()
results_df = load_results()

# Sidebar
st.sidebar.header("📊 Configuration")

# Section 0: Test Data Download
st.sidebar.subheader("0️⃣ Test Data Download")
st.sidebar.download_button(
    label="📥 Download Test Data (CSV)",
    data=test_data.to_csv(index=False),
    file_name="test_data.csv",
    mime="text/csv",
    help="Download test data to verify model predictions"
)
st.sidebar.info("💾 Download test data and upload back to app to verify model predictions")

st.sidebar.divider()

# Section 1: Model Selection
st.sidebar.subheader("1️⃣ Model Selection")
selected_model = st.sidebar.selectbox(
    "Select a classification model:",
    list(models.keys()),
    help="Choose the ML model to use for predictions"
)

st.sidebar.divider()

# Section 2: Data Upload
st.sidebar.subheader("2️⃣ Data Upload")
st.sidebar.info("💡 Upload CSV file with loan features (optional). If not uploaded, test data will be used.")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file for predictions",
    type="csv",
    help="CSV should contain the same features as training data"
)

if uploaded_file is not None:
    try:
        user_data = pd.read_csv(uploaded_file)
        # Drop loan_status if it exists in uploaded data
        if 'loan_status' in user_data.columns:
            user_data = user_data.drop('loan_status', axis=1)
        
        # Apply label encoders to categorical columns
        for col in label_encoders.keys():
            if col in user_data.columns:
                user_data[col] = label_encoders[col].transform(user_data[col])
        
        st.sidebar.success(f"✓ Loaded {len(user_data)} records")
        data_to_use = user_data
        use_user_data = True
    except Exception as e:
        st.sidebar.error(f"Error loading file: {str(e)}")
        data_to_use = test_data.drop('loan_status', axis=1)
        use_user_data = False
else:
    data_to_use = test_data.drop('loan_status', axis=1)
    use_user_data = False

st.sidebar.divider()

# Main content area - Two columns
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📈 Model Performance Metrics")
    
    # Get metrics for selected model
    model_row = results_df[results_df['Model'] == selected_model].iloc[0]
    
    # Display metrics in grid
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric(
            "Accuracy",
            f"{model_row['Accuracy']:.4f}",
            help="Percentage of correct predictions"
        )
        st.metric(
            "Precision",
            f"{model_row['Precision']:.4f}",
            help="True positives / All positive predictions"
        )
    
    with metric_col2:
        st.metric(
            "AUC-ROC",
            f"{model_row['AUC']:.4f}",
            help="Area under ROC curve (0-1, higher is better)"
        )
        st.metric(
            "Recall",
            f"{model_row['Recall']:.4f}",
            help="True positives / All actual positives"
        )
    
    with metric_col3:
        st.metric(
            "F1 Score",
            f"{model_row['F1']:.4f}",
            help="Harmonic mean of precision and recall"
        )
        st.metric(
            "MCC",
            f"{model_row['MCC']:.4f}",
            help="Matthews Correlation Coefficient (-1 to 1)"
        )

with col2:
    st.subheader("🏆 All Models Comparison")
    
    # Create comparison chart
    comparison_data = results_df[['Model', 'Accuracy', 'AUC', 'F1']].copy()
    comparison_data = comparison_data.sort_values('Accuracy', ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Accuracy', x=comparison_data['Accuracy'], y=comparison_data['Model'], orientation='h'))
    fig.add_trace(go.Bar(name='AUC', x=comparison_data['AUC'], y=comparison_data['Model'], orientation='h'))
    fig.add_trace(go.Bar(name='F1', x=comparison_data['F1'], y=comparison_data['Model'], orientation='h'))
    
    fig.update_layout(
        height=400,
        barmode='group',
        showlegend=True,
        margin=dict(l=150, r=50, t=50, b=50)
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Section: Detailed Model Performance
st.subheader("📊 Complete Metrics Comparison Table")

metrics_table = results_df.round(4)
st.dataframe(
    metrics_table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Model": st.column_config.TextColumn(width=150),
        "Accuracy": st.column_config.NumberColumn(format="%.4f"),
        "AUC": st.column_config.NumberColumn(format="%.4f"),
        "Precision": st.column_config.NumberColumn(format="%.4f"),
        "Recall": st.column_config.NumberColumn(format="%.4f"),
        "F1": st.column_config.NumberColumn(format="%.4f"),
        "MCC": st.column_config.NumberColumn(format="%.4f"),
    }
)

st.divider()

# Section: Predictions on Test Data
st.subheader("🔮 Model Predictions on Test Data")

model = models[selected_model]

# Get predictions - ensure only feature columns are used
X_test = data_to_use[feature_names].copy()
y_pred = model.predict(X_test)

if not use_user_data:
    y_true = test_data['loan_status'].values
    
    # Calculate confusion matrix with labels in order: 0=Not Approved, 1=Approved
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # cm structure: rows=[0,1], cols=[0,1]
    # Row 0: Not Approved actual [TN, FP]
    # Row 1: Approved actual [FN, TP]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Confusion Matrix")
        st.write("**Rows: Actual | Columns: Predicted**")
        
        # For display order: Approved (1) top, Not Approved (0) bottom
        # X-axis: Approved (1) left, Not Approved (0) right
        # We need to rearrange both rows and columns
        cm_display = cm[[1, 0], :][:, [1, 0]]
        # cm_display Row 0: Approved actual [TP, FN] 
        # cm_display Row 1: Not Approved actual [FP, TN]
        
        # Create heatmap
        # Plotly will display: first array element at bottom, last at top
        # So reverse the data array to match reversed y-labels
        cm_for_plot = cm_display[::-1]  # Flip rows for plotly display
        
        fig_cm = ff.create_annotated_heatmap(
            z=cm_for_plot,
            x=['Approved (1)', 'Not Approved (0)'],
            y=['Not Approved (0)', 'Approved (1)'],
            colorscale='Blues',
            showscale=True,
            text=cm_for_plot,
            texttemplate='%{text}'
        )
        fig_cm.update_layout(title_text='', height=400, width=400)
        fig_cm.update_xaxes(title_text='Predicted Label')
        fig_cm.update_yaxes(title_text='Actual Label')
        
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        st.subheader("Classification Report")
        
        # Get classification report with explicit order: 0, 1
        class_report = classification_report(
            y_true, y_pred,
            labels=[0, 1],
            target_names=['Not Approved (0)', 'Approved (1)'],
            output_dict=True
        )
        
        report_df = pd.DataFrame(class_report).transpose().round(4)
        st.dataframe(
            report_df,
            use_container_width=True,
            column_config={
                "precision": st.column_config.NumberColumn(format="%.4f"),
                "recall": st.column_config.NumberColumn(format="%.4f"),
                "f1-score": st.column_config.NumberColumn(format="%.4f"),
                "support": st.column_config.NumberColumn(format="%.0f"),
            }
        )
else:
    st.info("📝 Upload complete test data with 'loan_status' column to view confusion matrix and classification report")

st.divider()

# Show sample predictions
st.subheader("📋 Sample Predictions")
st.write(f"Showing first 10 predictions from {len(data_to_use)} records")

predictions_df = X_test.head(10).copy()
predictions_df['Prediction'] = y_pred[:10]
predictions_df['Prediction'] = predictions_df['Prediction'].map({0: 'Not Approved', 1: 'Approved'})

if not use_user_data and 'loan_status' in test_data.columns:
    predictions_df['Actual'] = test_data['loan_status'].head(10).map({0: 'Not Approved', 1: 'Approved'})

st.dataframe(predictions_df, use_container_width=True, hide_index=True)

st.divider()

# Footer
st.markdown("""
---
**Model Details:**
- **Logistic Regression**: Linear model for binary classification
- **Decision Tree**: Tree-based model with max depth 10
- **KNN**: K-Nearest Neighbors with k=5
- **Naive Bayes**: Gaussian probabilistic classifier
- **Random Forest**: Ensemble of 100 decision trees
- **XGBoost**: Gradient boosted trees ensemble

All models use StandardScaler for feature normalization and LabelEncoder for categorical features.
""")
