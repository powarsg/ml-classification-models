#!/usr/bin/env python3
"""
Helper script to add screenshot to PDF submission
Run this after saving bits_lab_screenshot.png to the project folder
"""
import os
import sys

workspace = "/Users/sagar.powar/sagar/my_data/bits/github-assignments/ml-classification-models"
screenshot = os.path.join(workspace, "bits_lab_screenshot.png")

print("📸 Screenshot Setup Instructions")
print("="*60)
print(f"\n1. Save your BITS Lab screenshot as:")
print(f"   {screenshot}")
print(f"\n2. The screenshot should show the successful execution of:")
print(f"   - Code running in BITS Virtual Lab")
print(f"   - Models being trained")
print(f"   - Results/output displayed")
print(f"\n3. Once saved, the PDF will automatically include it next time")
print(f"   you run: python3 create_pdf.py")
print(f"\n" + "="*60)

if os.path.exists(screenshot):
    print(f"\n✅ Screenshot found at: {screenshot}")
    print(f"✅ PDF will include it when regenerated")
    # Regenerate PDF
    os.system("python3 create_pdf.py")
else:
    print(f"\n⚠️  Screenshot not found. Please save it to the path above.")
    print(f"\n   Quick save options:")
    print(f"   • Firefox: Print → Save as PDF → Select Screenshot → Save")
    print(f"   • macOS: Cmd+Shift+4 → Select area → Drag to folder")
    print(f"   • Terminal: screencapture -i {screenshot}")
