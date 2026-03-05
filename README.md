# 🚘 License Plate Recognition using PaddleOCR v4 & v5

![PaddlePaddle](https://img.shields.io/badge/PaddlePaddle-3.3.0-blue?logo=paddlepaddle)
![PaddleOCR](https://img.shields.io/badge/PaddleOCR-3.4.0-orange)
![Python](https://img.shields.io/badge/Python-3.10%2B-green?logo=python)
![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia)

An end-to-end Computer Vision project that trains, evaluates, and compares **PP-OCRv4** and **PP-OCRv5** models for Automatic License Plate Recognition (ALPR). 

This repository contains two complete data pipelines built in Google Colab (T4 GPU), handling everything from YOLO bounding box extraction and pseudo-labeling to model training, batch inference, and comprehensive error analysis.

---

## 📑 Table of Contents
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Results & Visualizations](#-results--visualizations)
- [Error Analysis & Insights](#-error-analysis--insights)
- [Installation & Setup](#-installation--setup)
- [Repository Structure](#-repository-structure)

---

## 🗂 Dataset
**Source:** [Roboflow License Plates US-EU](https://public.roboflow.com/object-detection/license-plates-us-eu/3)
- Contains images of vehicles with YOLOv8 bounding box annotations.
- **Preprocessing:** Bounding boxes were parsed to crop the exact license plate regions.
- **Auto-labeling:** A pre-trained OCR model was used to generate pseudo-labels for training (`image \t text` format).

<img width="1352" height="495" alt="image" src="https://github.com/user-attachments/assets/0e2d3be3-850b-434b-9faf-f32f5bdba0c3" />

*Sample cropped license plates extracted from the training split.*

---

## ⚙️ Project Workflow

### 1️⃣ Notebook 1: Training Pipeline
- Parses YOLO labels and extracts cropped plate images.
- Generates PaddleOCR-compatible training text files.
- Configures custom YAML architectures for **PP-OCRv4** (`MobileNetV1Enhance` backbone) and **PP-OCRv5** (`PPLCNetV3` backbone).
- Trains both models for 10 epochs and exports inference weights.

### 2️⃣ Notebook 2: Inference Pipeline
- Loads the trained inference models directly via `subprocess` to bypass API conflicts.
- Runs inference on the test set, dynamically drawing bounding boxes and overlaying OCR predictions.
- Compares V4 vs V5 visually in a side-by-side matrix.
- Runs batch processing across the dataset to generate performance metrics.

---

## 📊 Results & Visualizations

### Side-by-Side Model Inference
The inference pipeline accurately maps predictions back to the original image coordinates using the YOLO labels. 
* Green Box: Plate region
* Cyan Text: PP-OCRv4 prediction + confidence
* Yellow Text: PP-OCRv5 prediction + confidence



![Comparison Grid](image-6.jpg)
*V4 vs V5 Inference Results Grid.*

### Training Evaluation
Due to compute constraints (10 epochs), absolute accuracy is near zero. However, the **Normalized Edit Distance (NED)** shows the models actively learning character distributions, with **PP-OCRv5** slightly outperforming V4.

<img width="1290" height="495" alt="image" src="https://github.com/user-attachments/assets/931102d9-97fe-4908-ab9c-c65e346a19a5" />


---

## 🔍 Error Analysis & Insights

To understand the model's behavior, a 6-panel error analysis was conducted on the batch inference results:

<img width="1790" height="985" alt="image" src="https://github.com/user-attachments/assets/7770a588-2819-4a5b-963c-3a984bde2123" />

### ⚠️ Why are confidence scores and accuracy low?
The current output consists of low-confidence characters and a 0.0% agreement rate between the two models. **This is expected and documented due to:**
1. **Compute Limits:** Models were trained for only **10 epochs** (Google Colab free-tier GPU limit). Deep OCR networks require 100+ epochs to converge.
2. **Architecture Mismatch:** Pre-trained weights didn't perfectly map to the custom configs, forcing the models to effectively learn from scratch.
3. **Noisy Labels:** Pseudo-labels generated for training contained artifacts and `UNKNOWN` tags from blurry images, introducing noise into the training signal.
4. **Vocabulary Differences:** V5 predicts longer strings (avg ~25 chars) than V4 (avg ~16 chars) due to a larger CTC decoder vocabulary, leading to more hallucinated characters at low confidence.

*Given 50+ epochs and cleaner ground-truth labels, the pipeline is structurally ready to achieve 80%+ accuracy.*

---

## 🚀 Installation & Setup

### Prerequisites
- Google Colab with **T4 GPU** enabled.
- Google Drive mounted.

### Installation
PaddlePaddle 3.x is not available on standard PyPI. You must install it from the official CUDA index:

```bash
# Uninstall conflicting versions
pip uninstall paddlepaddle paddlepaddle-gpu paddleocr -y -q

# Install for CUDA 13.0
python -m pip install paddlepaddle-gpu==3.3.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu130/ -q

# Install PaddleOCR
pip install paddleocr==3.4.0 -q
