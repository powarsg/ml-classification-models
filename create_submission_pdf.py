"""
Create Final Submission PDF for ML Classification Models Assignment
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, KeepTogether
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from datetime import datetime
import os
from PIL import Image as PILImage

# Define paths
workspace_path = "/Users/sagar.powar/sagar/my_data/bits/github-assignments/ml-classification-models"
screenshot_path = os.path.join(workspace_path, "bits_lab_screenshot.png")
pdf_path = os.path.join(workspace_path, "ML_Assignment_2_Submission.pdf")

# Create PDF document
doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                       rightMargin=0.75*inch, leftMargin=0.75*inch,
                       topMargin=0.75*inch, bottomMargin=0.75*inch)

# Container for PDF elements
story = []

# Define styles
styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontSize=24,
    textColor=colors.HexColor('#1f77b4'),
    spaceAfter=12,
    alignment=TA_CENTER,
    fontName='Helvetica-Bold'
)

heading_style = ParagraphStyle(
    'CustomHeading',
    parent=styles['Heading2'],
    fontSize=14,
    textColor=colors.HexColor('#2ca02c'),
    spaceAfter=10,
    spaceBefore=10,
    fontName='Helvetica-Bold'
)

link_style = ParagraphStyle(
    'LinkStyle',
    parent=styles['Normal'],
    fontSize=11,
    textColor=colors.HexColor('#0066cc'),
    spaceAfter=6,
    leftIndent=20
)

normal_style = ParagraphStyle(
    'NormalStyle',
    parent=styles['Normal'],
    fontSize=10,
    alignment=TA_JUSTIFY,
    spaceAfter=8
)

# ============ TITLE PAGE ============
story.append(Spacer(1, 0.5*inch))
story.append(Paragraph("Loan Approval Status Classification Models", title_style))
story.append(Paragraph("ML Assignment 2 - Final Submission", styles['Heading3']))
story.append(Spacer(1, 0.3*inch))
story.append(Paragraph(f"Submitted: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
story.append(Spacer(1, 0.5*inch))

# ============ SECTION 1: GITHUB REPOSITORY LINK ============
story.append(Paragraph("1. GitHub Repository Link", heading_style))
story.append(Paragraph(
    '<b>Repository:</b> <a href="https://github.com/powarsg/ml-classification-models">https://github.com/powarsg/ml-classification-models</a>',
    link_style
))
story.append(Paragraph(
    "Complete source code including app.py, requirements.txt, model training scripts, and comprehensive README.md",
    normal_style
))
story.append(Spacer(1, 0.2*inch))

# ============ SECTION 2: LIVE STREAMLIT APP LINK ============
story.append(Paragraph("2. Live Streamlit App Link", heading_style))
story.append(Paragraph(
    '<b>Live Application:</b> <a href="https://powarsg-ml-classification-models.streamlit.app">https://powarsg-ml-classification-models.streamlit.app</a>',
    link_style
))
story.append(Paragraph(
    "Interactive frontend deployed on Streamlit Community Cloud. Test model predictions with CSV upload functionality.",
    normal_style
))
story.append(Spacer(1, 0.2*inch))

# ============ SECTION 3: BITS LAB SCREENSHOT ============
story.append(Paragraph("3. BITS Virtual Lab Execution Screenshot", heading_style))

# Look for screenshot in multiple locations
screenshot_found = False
possible_paths = [
    screenshot_path,
    os.path.expanduser("~/Desktop/bits_lab_screenshot.png"),
    os.path.expanduser("~/Downloads/bits_lab_screenshot.png"),
    os.path.expanduser("~/Desktop/screenshot.png"),
    os.path.expanduser("~/Downloads/screenshot.png"),
]

for path in possible_paths:
    if os.path.exists(path):
        screenshot_path = path
        screenshot_found = True
        break

try:
    if screenshot_found:
        # Verify image dimensions
        img = PILImage.open(screenshot_path)
        img_width, img_height = img.size
        aspect_ratio = img_height / img_width
        
        # Scale to fit page width (max 6.5 inches)
        max_width = 6.5 * inch
        img_obj = Image(screenshot_path, width=max_width, height=max_width * aspect_ratio)
        story.append(img_obj)
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(
            "<i>Screenshot showing successful execution of assignment on BITS Virtual Lab</i>",
            ParagraphStyle('Caption', parent=styles['Normal'], fontSize=9, alignment=TA_CENTER, textColor=colors.grey)
        ))
        print(f"✅ Screenshot included: {screenshot_path}")
    else:
        story.append(Paragraph(
            f"<b>Note:</b> Screenshot should be placed at: {screenshot_path}",
            ParagraphStyle('Warning', parent=styles['Normal'], fontSize=10, textColor=colors.red)
        ))
        print(f"⚠️  Screenshot not found at expected paths")
except Exception as e:
    story.append(Paragraph(f"<b>Error loading screenshot:</b> {str(e)}", normal_style))
    print(f"❌ Error: {str(e)}")

story.append(Spacer(1, 0.2*inch))
story.append(PageBreak())

# ============ SECTION 4: README CONTENT ============
story.append(Paragraph("4. Project Documentation (README.md)", heading_style))
story.append(Spacer(1, 0.15*inch))

# Problem Statement
story.append(Paragraph("a. Problem Statement", styles['Heading3']))
story.append(Paragraph(
    "Predict loan approval status based on applicant demographics and financial information. "
    "Binary classification task (0 = Not Approved, 1 = Approved) for automated loan decision-making systems.",
    normal_style
))
story.append(Spacer(1, 0.1*inch))

# Dataset Description
story.append(Paragraph("b. Dataset Description", styles['Heading3']))
story.append(Paragraph(
    "<b>Dataset:</b> 45,000 loan application records (44,995 after preprocessing)<br/>"
    "<b>Features:</b> 13 features + 1 target variable<br/>"
    "<b>Target Distribution:</b> Imbalanced - 77.8% Not Approved (35,000), 22.2% Approved (10,000)<br/>"
    "<b>Preprocessing:</b> Removed age outliers, applied StandardScaler to numeric features, "
    "LabelEncoder to categorical features, Train-Val-Test Split: 80-10-10",
    normal_style
))
story.append(Spacer(1, 0.1*inch))

# Models Used
story.append(Paragraph("c. Models Used", styles['Heading3']))
story.append(Paragraph(
    "<b>6 Classification Models:</b><br/>"
    "1. Logistic Regression (89.27% accuracy, 0.9507 AUC)<br/>"
    "2. Decision Tree (92.04% accuracy, 0.9605 AUC)<br/>"
    "3. K-Nearest Neighbors (88.84% accuracy, 0.9292 AUC)<br/>"
    "4. Naive Bayes (72.82% accuracy, 0.9403 AUC)<br/>"
    "5. Random Forest (92.11% accuracy, 0.9742 AUC)<br/>"
    "6. XGBoost - Best Model (93.11% accuracy, 0.9788 AUC)<br/>",
    normal_style
))

# Create metrics table with better sizing
metrics_data = [
    ['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
    ['Logistic Regression', '0.8927', '0.9507', '0.7630', '0.7500', '0.7564', '0.6876'],
    ['Decision Tree', '0.9204', '0.9605', '0.8877', '0.7350', '0.8042', '0.7601'],
    ['KNN', '0.8884', '0.9292', '0.7672', '0.7150', '0.7402', '0.6699'],
    ['Naive Bayes', '0.7282', '0.9403', '0.4498', '1.0000', '0.6205', '0.5410'],
    ['Random Forest', '0.9211', '0.9742', '0.8781', '0.7490', '0.8084', '0.7629'],
    ['XGBoost', '0.9311', '0.9788', '0.8868', '0.7910', '0.8362', '0.7948'],
]

metrics_table = Table(metrics_data, colWidths=[1.0*inch, 0.85*inch, 0.75*inch, 0.85*inch, 0.75*inch, 0.65*inch, 0.65*inch])
metrics_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ca02c')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 8),
    ('FONTSIZE', (0, 1), (-1, -1), 7),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
]))
story.append(metrics_table)
story.append(Spacer(1, 0.15*inch))

# Model Observations
story.append(Paragraph("d. Observations on Model Performance", styles['Heading3']))

# Create smaller observation paragraphs instead of large table
obs_models = [
    ("Logistic Regression", "Moderate baseline (89.27% accuracy, 0.9507 AUC). Interpretable but linear assumptions limit non-linear pattern capture."),
    ("Decision Tree", "Good performance (92.04% accuracy, 0.9605 AUC). Interpretable rules and feature interactions handled naturally."),
    ("kNN", "Good accuracy (88.84%, 0.9292 AUC) capturing local patterns without assumptions. Computationally expensive for large datasets."),
    ("Naive Bayes", "Poor performance (72.82% accuracy). Critical issue: 100% recall but 0.45% precision - approves almost all loans, unacceptable for production."),
    ("Random Forest", "Excellent performance (92.11% accuracy, 0.9742 AUC). Robust ensemble handling non-linear patterns. Strong production candidate."),
    ("XGBoost ⭐", "Best overall (93.11% accuracy, 0.9788 AUC, 0.8362 F1, 0.7948 MCC). Sequential boosting handles imbalanced data natively. Optimal for production."),
]

small_style = ParagraphStyle(
    'SmallNormal',
    parent=styles['Normal'],
    fontSize=9,
    alignment=TA_LEFT,
    spaceAfter=6,
    leftIndent=10
)

for model_name, observation in obs_models:
    story.append(Paragraph(f"<b>{model_name}:</b> {observation}", small_style))
story.append(Spacer(1, 0.15*inch))

# Key Learning
story.append(Paragraph("Key Learning - Imbalanced Data:", styles['Heading4']))
story.append(Paragraph(
    "• Accuracy alone is misleading (77.8% vs 22.2% class distribution)<br/>"
    "• AUC, F1, and MCC are more informative metrics for imbalanced datasets<br/>"
    "• Ensemble methods (Random Forest, XGBoost) naturally handle class imbalance better<br/>"
    "• Trade-off between precision and recall critical in loan approval decisions",
    normal_style
))
story.append(Spacer(1, 0.15*inch))

# Repository Structure
story.append(Paragraph("Repository Structure:", styles['Heading4']))
story.append(Paragraph(
    "<b>project-folder/</b><br/>"
    "├── app.py (Streamlit web application)<br/>"
    "├── requirements.txt (Python dependencies)<br/>"
    "├── README.md (Documentation)<br/>"
    "└── model/<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;├── train_models.py (Training script)<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;├── loan_data.csv (Original dataset)<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;├── [6 model .joblib files]<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;├── scaler.joblib, label_encoders.joblib<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;├── test_data.csv (Test dataset)<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;└── model_results.csv (Evaluation metrics)",
    normal_style
))

# Build PDF
doc.build(story)
print(f"✅ PDF Created: {pdf_path}")
print(f"📄 File size: {os.path.getsize(pdf_path) / 1024:.1f} KB")
print(f"\n📋 PDF Contents:")
print("  1. GitHub Repository Link")
print("  2. Live Streamlit App Link")
print("  3. BITS Lab Execution Screenshot")
print("  4. Complete README Documentation")
print("     - Problem Statement")
print("     - Dataset Description")
print("     - Models Used (6 models with metrics table)")
print("     - Model Performance Observations (table format)")
print("     - Key Learnings & Repository Structure")
