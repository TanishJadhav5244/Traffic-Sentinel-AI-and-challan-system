# 🏍️ Traffic Sentinel AI — Automated Helmet Violation Detection & E-Challan System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Computer_Vision-YOLOv8-FF6F00.svg?logo=ultralytics&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/Image_Processing-OpenCV-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![EasyOCR](https://img.shields.io/badge/OCR-EasyOCR-009688.svg)](https://github.com/JaidedAI/EasyOCR)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, real-time AI computer vision system for detecting two-wheeler helmet violations, extracting vehicle license plates using adaptive low-light enhanced OCR, querying RTO vehicle registration databases, generating digital E-Challan tickets, and providing actionable analytics via an interactive web dashboard.

---

## 📌 Project Overview

Urban traffic management faces severe challenges in enforcing two-wheeler helmet compliance. **Traffic Sentinel AI** automates this process end-to-end:

1. **Scans** live traffic video feeds or images for two-wheelers.
2. **Detects** riders, helmet/no-helmet status, and license plate bounding boxes using YOLOv8.
3. **Associates** unhelmeted riders directly to their vehicle's license plate using spatial bounding-box mapping algorithms.
4. **Enhances** dark or low-light license plate crops with multi-scale illumination algorithms (Night Vision Mode).
5. **Recognizes** license plate text via EasyOCR or Tesseract OCR engines with automated deskewing and Otsu binarization.
6. **Queries** RTO databases to fetch vehicle owner details and insurance status.
7. **Issues** official digital **E-Challan tickets** with evidence crops, timestamped logs, and fine breakdowns.

---

## ⚙️ End-to-End System Pipeline

```
  ┌─────────────────────────────────────────────────────────────┐
  │                 Input Feed (Image / Video)                  │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │             1. Object Detection (YOLOv8 Engine)             │
  │     Detects: Rider, Helmet, No-Helmet, & License Plate       │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │           2. Violation Association & Mapping Logic          │
  │    Maps unhelmeted rider box to overlapping license plate   │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │        3. Low-Light / Night-Vision Enhancement (Retinex)    │
  │    Auto-detects low luminance & boosts contrast with CLAHE   │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │           4. OCR Preprocessing & Text Extraction             │
  │  Bilateral Filtering ➔ Deskew ➔ Otsu Binary ➔ EasyOCR / Tess │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │           5. RTO Database Lookup & E-Challan Builder        │
  │     Fetches owner details & builds printable PNG ticket     │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │       6. Streamlit Interactive Dashboard & Analytics        │
  │    Displays real-time scans, logs, map trends, & exports     │
  └─────────────────────────────────────────────────────────────┘
```

### Detailed Pipeline Stages

| Stage | Module | Functionality |
|---|---|---|
| **1. Detection** | `backend/detector.py` | Utilizes YOLOv8 fine-tuned weights (`best_detector.pt`) to detect riders, helmet compliance, and license plates simultaneously with configurable confidence thresholds. |
| **2. Association** | `backend/detector.py` | Calculates spatial bounding box overlap (IoU and Euclidean distance) to match each flagged `no-helmet` rider with their respective vehicle's license plate. |
| **3. Night Vision** | `backend/image_enhancer.py` | Evaluates mean luminance ($L < 85$). Applies Multi-Scale Retinex illumination correction, CLAHE (Contrast Limited Adaptive Histogram Equalization), and bilateral noise removal. |
| **4. OCR Engine** | `backend/ocr_engine.py` | Upscales plate crops 2x, removes skew angles via minimum-area bounding rectangles, applies Otsu thresholding, and parses text matching Indian license plate regex. |
| **5. E-Challan** | `backend/challan_generator.py` | Renders a official E-Challan ticket containing timestamp, location, violator headshot crop, plate crop, owner info, fine amount, and verification QR code. |
| **6. Database & UI**| `app.py` & `frontend/` | Streamlit web application featuring live video/image processing, searchable violation database, OCR Preprocessing Lab, RTO lookup, and Plotly analytics dashboard. |

---

## 🌟 Key Features

- 🎯 **Real-time Video & Image Scanning**: Processes single images or full MP4 video files with automatic deduplication over rolling time windows.
- 🌙 **Night Vision & Low-Light Enhancement**: Retinex and CLAHE filters ensure high OCR accuracy even on dark night-time camera captures.
- 🔤 **Dual OCR Engine**: Switch seamlessly between **EasyOCR** and **Google Tesseract OCR** from the sidebar configuration panel.
- 🔍 **RTO Registry Lookup**: Instant access to registered vehicle owner names, vehicle models, registration dates, and insurance validity status.
- 📄 **Automatic E-Challan Generation**: Generates high-resolution PNG ticket tickets ready for downloading and emailing.
- 📈 **Analytical Dashboard**: Interactive Plotly charts analyzing violations over time, peak hours, manufacturer breakdown, state distribution, and repeat offenders.
- 🔬 **OCR Preprocessing Lab**: Visual debug tab allowing developers to inspect each image transformation step (Grayscale, Bilateral Filter, Deskew, Otsu Binary).

---

## 📁 Repository Structure

```
Traffic-Sentinel-AI-and-challan-system/
├── app.py                      # Main Streamlit Dashboard Application
├── config.yaml                 # System Thresholds, Model Paths, & OCR Settings
├── requirements.txt            # Python Dependencies
├── ocr_report.md               # Technical OCR Performance Report & Metrics
├── .gitignore                  # Git Ignore Rules for Python/PyTorch/Windows
│
├── backend/                    # Core Vision & Processing Engine
│   ├── detector.py             # YOLOv8 Detection & Association Engine
│   ├── ocr_engine.py           # EasyOCR & Tesseract Handler with Deskewing
│   ├── image_enhancer.py       # Retinex & CLAHE Low-Light Enhancement
│   ├── db_helper.py            # SQLite/CSV Violation Storage & Aggregations
│   ├── challan_generator.py    # E-Challan Ticket Image Generator
│   ├── rto_helper.py           # RTO Vehicle Registry Helper
│   └── models/
│       ├── download_weights.py # Automated YOLOv8 Model Weights Loader
│       └── train_detector.py   # YOLOv8 Training Script
│
├── frontend/                   # Dashboard Styling & Custom Components
│   ├── components.py           # Custom Metric Cards, Panels, & Chips
│   └── styles.py               # Custom CSS Glassmorphism Styles
│
├── css/
│   └── styles.css              # Dashboard Design Tokens & CSS Utilities
│
├── test_assets/                # Sample Traffic Images & Video Clips for Testing
│   ├── plate_clean.png
│   ├── plate_blurry.png
│   ├── plate_noisy.png
│   ├── plate_skewed.png
│   └── traffic_sample.png
│
├── tests/                      # Automated System Tests
│   └── test_pipeline.py        # End-to-End Pipeline Unit Test Suite
│
└── violations/                 # Runtime Generated Log Outputs (Ignored in Git)
    ├── crops/                  # Saved Plate & Rider Headshot Bounding Crops
    └── challans/               # Generated E-Challan PNG Tickets
```

---

## 🚀 Quick Start Guide

### Prerequisites

- **Python**: 3.10 or higher
- **Git**: Installed on your system
- *(Optional)* **Tesseract-OCR**: If using Tesseract engine ([Download Windows Installer](https://github.com/UB-Mannheim/tesseract/wiki)).

### 1. Clone the Repository

```bash
git clone https://github.com/TanishJadhav5244/Traffic-Sentinel-AI-and-challan-system.git
cd Traffic-Sentinel-AI-and-challan-system
```

### 2. Set Up Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Model Weights & Demo Assets

Run the helper script to auto-fetch YOLOv8 weights and sample assets:

```bash
python backend/models/download_weights.py
```

### 5. Launch the Dashboard

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🧪 Running Unit Tests

Run the test suite to verify the detector, image enhancer, OCR engine, and database pipeline:

```bash
python -m unittest tests/test_pipeline.py
```

---

## 🛠️ Configuration (`config.yaml`)

You can customize system parameters directly in [`config.yaml`](config.yaml):

```yaml
models:
  detector_weights: "models/best_detector.pt"
  confidence:
    rider: 0.35
    helmet: 0.40
    no_helmet: 0.35
    license_plate: 0.40

ocr:
  default_engine: "easyocr"   # Options: "easyocr" or "tesseract"
  tesseract_path: "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
  enhancer:
    auto_low_light: true
    luminance_threshold: 85
    force_enhancement: false
  plate_regex: "^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$"
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request:
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<p center>Crafted for Intelligent Transportation Systems & Smart City Enforcement 🚦</p>
