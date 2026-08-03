# 👁️ Face-Analytics: Real-Time Recognition & Age Estimation

A high-performance, modular Computer Vision pipeline designed for **Real-Time Face Detection, Identity Recognition, and Age Estimation**. This repository serves as a portfolio demonstration of a production-ready edge analytics system.

> **Disclaimer:** This is a public showcase version. Proprietary model weights, enterprise multiprocessing optimizations, and internal datasets have been excluded.

## 🚀 Key Features

- **Zero-Redundancy Pipeline:** Implements `detector_backend="skip"` to prevent redundant cascading detections, significantly boosting FPS.
- **Adaptive UI/UX:** Dynamic bounding box rendering, confidence scores, and adaptive font sizing based on input resolution (supports 480p up to 4K streams).
- **Smart Thresholding:** Custom distance thresholds for ArcFace embeddings to minimize False Positives (Unknown face handling).
- **Multi-Source Support:** Seamlessly process static images, local video files, webcams, and RTSP IP-camera streams via CLI arguments.

## 🧠 Architecture & Tech Stack

- **Face Detection:** RetinaFace (MobileNetV1 backbone) for fast, robust localization.
- **Feature Extraction & Matching:** ArcFace embeddings (via DeepFace) for high-accuracy facial recognition.
- **Demographics:** DeepFace Action Analyzers for age estimation.
- **Frameworks:** Python, OpenCV, NumPy, TensorFlow/PyTorch backend.

## 📂 Repository Structure & Database Setup

To utilize the facial recognition feature, structure your database directory (`db/`) with individual folders for each identity. The pipeline dynamically reads the folder name as the identity label.

```text
db/
├── John_Doe/
│   ├── img1.jpg
│   └── img2.jpg
└── Jane_Smith/
    └── img1.jpg
```

---

# 💻 Quick Start

## 1. Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/parsa-mjls/Face-Analytics.git
cd Face-Analytics
pip install -r requirements.txt
```

## 2. Inference & Execution

The pipeline is entirely CLI-driven using `argparse`.

### Run on Webcam

```bash
python inference.py --source 0
```

### Run on a Video File or RTSP Stream

```bash
python inference.py --source "data/sample_video.mp4"
```

### Run on a Static Image (High-Speed `imread` Mode)

```bash
python inference.py --source "data/test_image.jpg"
```

### Advanced Usage (Custom Threshold & Database)

```bash
python inference.py --source 0 --db "path/to/custom_db" --threshold 0.35
```
