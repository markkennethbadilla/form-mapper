# Form Mapper

A Streamlit app to digitize physical pre-printed forms for automated printing.

## Features

- 📄 Generate calibration PDFs with ArUco markers
- 🎯 Automatic perspective correction using homography
- ✏️ Interactive field mapping with drawable canvas
- 📊 Export field positions as percentage-based JSON

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python3 -m streamlit run app.py
```

## Workflow

1. **Setup**: Enter paper dimensions and generate calibration PDF
2. **Print & Scan**: Print PDF onto pre-printed form, then scan
3. **Map Fields**: Upload scan, draw rectangles to define fields
4. **Export**: Download JSON with percentage-based coordinates

## Deployment

Deploy to Streamlit Cloud:
1. Push to GitHub
2. Connect repository to Streamlit Cloud
3. Deploy!

## Requirements

- Python 3.8+
- OpenCV for ArUco marker detection
- ReportLab for PDF generation
- Streamlit for UI
