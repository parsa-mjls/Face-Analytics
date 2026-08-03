# Face-Analytics
Real-Time Face Recognition &amp; Age Estimation Pipeline


# 👁️ Face-Analytics: Real-Time Recognition & Age Estimation

A high-performance, modular Computer Vision pipeline designed for **Real-Time Face Detection, Identity Recognition, and Age Estimation**. This repository serves as a portfolio demonstration of a production-ready edge analytics system.

> **Disclaimer:** This is a public showcase version. Proprietary model weights, enterprise multiprocessing optimizations, and internal datasets have been excluded.

## 🚀 Key Features

* **Zero-Redundancy Pipeline:** Implements `detector_backend="skip"` to prevent redundant cascading detections, significantly boosting FPS.
* **Adaptive UI/UX:** Dynamic bounding box rendering, confidence scores, and adaptive font sizing based on input resolution (supports 480p up to 4K streams).
* **Smart Thresholding:** Custom distance thresholds for ArcFace embeddings to minimize False Positives (Unknown face handling).
* **Multi-Source Support:** Seamlessly process static images, local video files, webcams, and RTSP IP-camera streams via CLI arguments.

## 🧠 Architecture & Tech Stack

* **Face Detection:** RetinaNet (MobileNetV1 backbone) for fast, robust localization.
* **Feature Extraction & Matching:** ArcFace embeddings (via DeepFace) for high-accuracy facial recognition.
* **Demographics:** DeepFace Action Analyzers for age estimation.
* **Frameworks:** Python, OpenCV, NumPy, TensorFlow/PyTorch backend.

## 📂 Repository Structure & Database Setup

To utilize the facial recognition feature, structure your database directory (`db/`) with individual folders for each identity. The pipeline dynamically reads the folder name as the identity label.

```text
db/
 ├── John_Doe/
 │   ├── img1.jpg
 │   └── img2.jpg
 └── Jane_Smith/
     └── img1.jpg

💻 Quick Start
1. Installation
Clone the repository and install the required dependencies:

git clone [https://github.com/YOUR_USERNAME/Face-Analytics.git](https://github.com/YOUR_USERNAME/Face-Analytics.git)
cd Face-Analytics
pip install -r requirements.txt
