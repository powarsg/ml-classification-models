#!/usr/bin/env python3
"""
Create Final Submission PDF for ML Classification Models Assignment
Fixed version with better table formatting
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
import os

# Paths
workspace = "/Users/sagar.powar/sagar/my_data/bits/github-assignments/ml-classification-models"
pdf_path = os.path.join(workspace, "ML_Assignment_2_Submission.pdf")

doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                       rightMargin=0.6*inch, leftMargin=0.6*inch,
                       topMargin=0.6*inch, bottomMargin=0.6*inch)

story = []
styles = getSampleStyleSheet()

# Custom styles
title_style = ParagraphStyle('Title', parent=styles['Heading1'], 
                            fontSize=22, textColor=colors.HexColor('#1f77b4'),
                            spaceAfter=10, alignment=TA_CENTER, fontName='Helvetica-Bold')

head_style = ParagraphStyle('Head', parent=styles['Heading2'],
                           fontSize=13, textColor=colors.HexColor('#2ca02c'),
                           spaceAfter=8, spaceBefore=8, fontName='Helvetica-Bold')

link_style = ParagraphStyle('Link', parent=styles['Normal'],
                           fontSize=10, textColor=colors.HexColor('#0066cc'),
                           spaceAfter=4, leftIndent=15)

body_style = ParagraphStyle('Body', parent=styles['Normal'],
                           fontSize=9.5, alignment=TA_JUSTIFY, spaceAfter=6)

small_style = ParagraphStyle('Small', parent=styles['Normal'],
                            fontSize=8.5, spaceAfter=4, leftIndent=12)

# ============ TITLE PAGE ============
story.append(Spacer(1, 0.3*inch))
story.append(Paragraph("Loan Approval Status Classification Models", title_style))
story.append(Paragraph("ML Assignment 2 - Submission", styles['Heading3']))
story.append(Spacer(1, 0.2*inch))
story.append(Paragraph(f"Submitted: {datetime.now().strftime('%B %d, %Y')}", body_style))
story.append(Spacer(1, 0.3*inch))

# ============ SUBMISSION LINKS ============
story.append(Paragraph("1. GitHub Repository", head_style))
story.append(Paragraph(
    '<link href="https://github.com/powarsg/ml-classification-models">https://github.com/powarsg/ml-classification-models</link>',
    link_style
))
story.append(Paragraph(
    "Complete source code with app.py, requirements.txt, model training scripts, and comprehensive README",
    body_style
))
story.append(Spacer(1, 0.15*inch))

story.append(Paragraph("2. Live Streamlit App", head_style))
story.append(Paragraph(
    '<link href="https://powarsg-ml-classification-models.streamlit.app">https://powarsg-ml-classification-models.streamlit.app</link>',
    link_style
))
story.append(Paragraph(
    "Interactive frontend deployed on Streamlit Community Cloud with model selection, CSV upload, and real-time predictions",
    body_style
))
story.append(Spacer(1, 0.15*inch))

# ============ SCREENSHOT ============
story.append(Paragraph("3. BITS Virtual Lab Execution", head_style))

screenshot_path = os.path.join(workspace, "bits_lab_screenshot.png")
if os.path.exists(screenshot_path):
    try:
        img = Image(screenshot_path, width=6.0*inch, height=4.0*inch)
        story.append(img)
        story.append(Spacer(1, 0.08*inch))
        story.append(Paragraph(
            "<i>Screenshot showing successful assignment execution on BITS Virtual Lab</i>",
            ParagraphStyle('Caption', parent=styles['Normal'], fontSize=8.5, 
                          alignment=TA_CENTER, textColor=colors.grey)
        ))
    except:
        story.append(Paragraph(
            "Screenshot: bits_lab_screenshot.png (place in project folder)",
            body_style
        ))
else:
    story.append(Paragraph(
        "Screenshot: Place bits_lab_screenshot.png in project folder",
        body_style
    ))

story.append(Spacer(1, 0.15*inch))
story.append(PageBreak())

# ============ README DOCUMENTATION ============
story.append(Paragraph("4. Project Documentation", head_style))
story.append(Spacer(1, 0.1*inch))

# a. Problem Statement
story.append(Paragraph("<b>a. Problem Statement</b>", styles['Heading3']))
story.append(Paragraph(
    "Predict loan approval status based on applicant demographics and financial information. "
    "Binary classification: 0 = Not Approved, 1 = Approved",
    body_style
))
story.append(Spacer(1, 0.08*inch))

# b. Dataset Description
story.append(Paragraph("<b>b. Dataset Description</b>", styles['Heading3']))
story.append(Paragraph(
    "<b>Dataset:</b> 45,000 records (44,995 after outlier removal) | "
    "<b>Features:</b> 13 input features + target | "
    "<b>Split:</b> 80% train, 10% val, 10% test",
    small_style
))
story.append(Paragraph(
    "<b>Class Distribution:</b> 77.8% Not Approved (35,000) | 22.2% Approved (10,000) - Imbalanced",
    small_style
))
story.append(Paragraph(
    "<b>Preprocessing:</b> StandardScaler (numeric), LabelEncoder (categorical), Stratified split",
    small_style
))
story.append(Spacer(1, 0.08*inch))

# c. Models Used
story.append(Paragraph("<b>c. Models Used</b>", styles['Heading3']))
story.append(Paragraph(
    "6 Classification Models: Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost",
    small_style
))
story.append(Spacer(1, 0.08*inch))

# Metrics Table
metrics_data = [
    ['Model', 'Accuracy', 'AUC', 'F1', 'MCC'],
    ['Logistic Regression', '0.8927', '0.9507', '0.7564', '0.6876'],
    ['Decision Tree', '0.9204', '0.9605', '0.8042', '0.7601'],
    ['KNN', '0.8884', '0.9292', '0.7402', '0.6699'],
    ['Naive Bayes', '0.7282', '0.9403', '0.6205', '0.5410'],
    ['Random Forest', '0.9211', '0.9742', '0.8084', '0.7629'],
    ['XGBoost', '0.9311', '0.9788', '0.8362', '0.7948'],
]

metrics_table = Table(metrics_data, colWidths=[1.3*inch, 0.9*inch, 0.8*inch, 0.8*inch, 0.8*inch])
metrics_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ca02c')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 8),
    ('FONTSIZE', (0, 1), (-1, -1), 7),
    ('TOPPADDING', (0, 0), (-1, -1), 3),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
]))
story.append(metrics_table)
story.append(Spacer(1, 0.12*inch))

# d. Model Observations
story.append(Paragraph("<b>d. Model Performance Observations</b>", styles['Heading3']))

observations = [
    ("Logistic Regression", "Moderate baseline (89.27% accuracy, 0.9507 AUC). Interpretable but limited to linear decision boundaries."),
    ("Decision Tree", "Good performance (92.04% accuracy, 0.9605 AUC). Natural feature interaction handling with interpretable rules."),
    ("KNN", "Solid performance (88.84% accuracy, 0.9292 AUC). Requires feature scaling; computationally expensive at scale."),
    ("Naive Bayes", "Poor practical performance (72.82% accuracy). Critical flaw: 100% recall but 0.45% precision - unacceptable for production."),
    ("Random Forest", "Excellent (92.11% accuracy, 0.9742 AUC). Robust ensemble with feature importance insights. Strong candidate."),
    ("XGBoost ⭐", "BEST MODEL (93.11% accuracy, 0.9788 AUC, 0.8362 F1). Sequential boosting, native imbalance handling. Recommended for production."),
]

for name, obs in observations:
    story.append(Paragraph(f"<b>{name}:</b> {obs}", small_style))

story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("<b>Key Insights on Imbalanced Data:</b>", styles['Heading4']))
story.append(Paragraph(
    "• Accuracy misleading (77.8% vs 22.2% class ratio) • AUC, F1, MCC more informative • "
    "Ensemble methods handle imbalance naturally • Precision-recall trade-off critical for loan decisions",
    small_style
))

# Build and save
doc.build(story)
print(f"✅ PDF created: {pdf_path}")
print(f"📄 Size: {os.path.getsize(pdf_path) / 1024:.1f} KB")
print("\n📋 Contents:")
print("  ✓ GitHub Repository Link")
print("  ✓ Live Streamlit App Link")
print("  ✓ BITS Lab Screenshot (if bits_lab_screenshot.png exists)")
print("  ✓ Complete README with:")
print("    - Problem Statement")
print("    - Dataset Description")
print("    - 6 Models with Metrics Table")
print("    - Model Performance Observations")
print("    - Key Insights on Imbalanced Data")
