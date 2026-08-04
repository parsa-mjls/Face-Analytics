# 👁️ Face-Analytics: Real-Time Recognition & Age Estimation

A high-performance, modular Computer Vision pipeline designed for **Real-Time Face Detection, Face Alignment, Identity Recognition, and Age Estimation**. This repository serves as a portfolio demonstration of a production-ready edge analytics system optimized for speed, precision, and version-agnostic stability.

> **Disclaimer:** This is a public showcase version. Proprietary model weights, enterprise multiprocessing optimizations, and internal datasets have been excluded.

---

## 🚀 Key Features

- **Pre-Aligned Face Embeddings:** Explicitly utilizes aligned face tensors (`align=True`) before passing crops to ArcFace, dramatically increasing recognition accuracy on non-frontal and tilted faces.
- **Automated Output Recording (`--output`):** Intelligent multi-format saving pipeline. Automatically saves annotated images (`.jpg`, `.png`, etc.) for static inputs, or records high-FPS annotated video files (`.mp4`) for video files, webcams, and RTSP streams.
- **Zero-Redundancy & Frame-Skipping:** Implements `detector_backend="skip"` during face matching and analysis to eliminate redundant detection passes, paired with configurable frame-skipping for real-time RTSP/camera processing.
- **Smart Size & Confidence Filtering:** Eliminates background noise and distant low-res faces via customizable pixel-dimension (`--min-face-size`) and confidence thresholds.
- **Dynamic Distance Column Resolution:** Uses robust dynamic column searching rather than hardcoded indexes, making the pipeline resilient against breaking updates across different DeepFace versions.
- **Adaptive UI/UX:** Dynamic bounding box rendering, color-coded identity status (Green for known, Orange for unknown), confidence scores, and auto-scaling font sizes (supports 480p up to 4K streams).
- **Multi-Source Support:** Seamlessly process static images, local video files, live webcams, and IP RTSP streams via CLI arguments.

---

## 🧠 Architecture & Tech Stack

- **Face Detection:** YOLOv8 (PyTorch backend, default for high FPS) with optional support for RetinaFace, OpenCV, and SSD backends.
- **Face Alignment & Feature Extraction:** ArcFace embeddings with aligned face preprocessing via DeepFace.
- **Demographics:** DeepFace Action Analyzers for age estimation.
- **Image Processing & I/O:** OpenCV (VideoWriter, VideoCapture, GUI rendering) & NumPy.
- **Frameworks:** Python 3.8+, OpenCV, PyTorch, DeepFace, Pandas.

---

## 📂 Repository Structure & Database Setup

To utilize the facial recognition feature, structure your database directory (`db/` or custom path) with individual subfolders for each identity. The pipeline dynamically extracts the parent folder name as the person's identity label.

```text
Face-Analytics/
├── db/
│   ├── John_Doe/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── Jane_Smith/
│       └── img1.jpg
├── data/
│   ├── sample_video.mp4
│   └── test_image.jpg
├── inference.py
├── requirements.txt
└── README.md
```

## 💻 Quick Start

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/parsa-mjls/Face-Analytics.git
cd Face-Analytics
pip install -r requirements.txt
```

---

### 2. CLI Options & Parameters

The inference script is fully configurable via the Command Line Interface (CLI).

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--source` | `str` | **Required** | Path to an image/video file, webcam index (e.g., `0`), or RTSP stream URL |
| `--output` | `str` | `None` | Optional path to save the annotated output (image or `.mp4` video) |
| `--db` | `str` | `my_db2` | Path to the face database directory |
| `--threshold` | `float` | `0.5` | ArcFace cosine distance threshold (lower = stricter matching) |
| `--conf` | `float` | `0.78` | Minimum face detection confidence |
| `--min-face-size` | `int` | `80` | Minimum face width/height (pixels) required for processing |
| `--detector` | `str` | `yolov8` | Detection backend (`yolov8`, `retinaface`, `opencv`, `ssd`) |
| `--frame-skip` | `int` | `15` | Perform heavy inference every N frames for improved real-time performance |

---

### 3. Inference & Execution Examples

#### 📷 Processing Static Images

Process a static image and save the annotated result.

```bash
python inference.py \
    --source "data/test_image.jpg" \
    --output "output/annotated_image.jpg"
```

---

#### 🎥 Processing Video Files

Run inference on a video file while skipping every 10 frames.

```bash
python inference.py \
    --source "data/sample_video.mp4" \
    --output "output/annotated_video.mp4" \
    --frame-skip 10
```

---

#### 📹 Live Webcam Stream

Run real-time analytics using the default webcam.

```bash
python inference.py --source 0
```

---

#### 📡 RTSP IP Camera Stream with Output Recording

Run inference on an RTSP stream while filtering out small faces and recording the output.

```bash
python inference.py \
    --source "rtsp://admin:password@192.168.1.100:554/stream" \
    --output "output/rtsp_recording.mp4" \
    --min-face-size 100 \
    --conf 0.80
```

---

#### ⚙️ Advanced Custom Configuration

Use a custom face database with stricter recognition settings.

```bash
python inference.py \
    --source 0 \
    --db "custom_faces_db" \
    --threshold 0.40 \
    --detector yolov8 \
    --frame-skip 5
```

---

## ⚙️ Technical Highlights

### 🎯 Face Alignment Optimization

Raw face crops often produce suboptimal ArcFace embeddings when faces are rotated or tilted. This pipeline extracts **aligned face tensors** (`aligned_face`) directly from `extract_faces()`, ensuring significantly more robust and discriminative feature representations.

### 🛡️ Dynamic DataFrame Safe-Check

Instead of relying on hardcoded DataFrame indices (e.g., `df.columns[-1]`), the pipeline dynamically locates the appropriate distance metric column, making it resilient to API and version changes in DeepFace.

### 💾 Automated Media Writer

The pipeline automatically determines the input media type during initialization:

- Static images are saved using `cv2.imwrite()`.
- Videos, webcams, and RTSP streams are recorded using `cv2.VideoWriter()`.

This allows a single `--output` argument to seamlessly support both images and videos without additional configuration.

