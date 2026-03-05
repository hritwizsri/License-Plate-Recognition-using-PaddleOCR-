# 🚘 License Plate Recognition using PaddleOCR v4 & v5

![PaddlePaddle](https://img.shields.io/badge/PaddlePaddle-3.3.0-blue?logo=paddlepaddle)
![PaddleOCR](https://img.shields.io/badge/PaddleOCR-3.4.0-orange)
![Python](https://img.shields.io/badge/Python-3.10%2B-green?logo=python)
![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia)

End-to-end Computer Vision pipeline for Automatic License Plate Recognition (ALPR), comparing **PP-OCRv4** vs **PP-OCRv5**.

This repo contains two notebooks:
- **Notebook 1 (Training):** YOLO bbox → plate crops → pseudo-labels → train v4/v5 → evaluate → export.
- **Notebook 2 (Inference):** load trained models → run OCR on test images → annotate → compare → batch stats → error analysis.

---

## 🗂 Dataset
**Source:** Roboflow License Plates US-EU (YOLO format: images + bbox labels).

We crop plates using YOLO bounding boxes to create recognition-ready images.

![Sample cropped license plates](images/dataset_crops.jpg)

---

## ⚙️ Project workflow

### Notebook 1 — Training
1. Install PaddlePaddle + PaddleOCR.
2. Prepare plate crops from YOLO bboxes.
3. Auto-label crops to create `image	text` ground truth files.
4. Train **PP-OCRv4** and **PP-OCRv5** for 10 epochs.
5. Evaluate and export models.

### Notebook 2 — Inference
1. Load trained checkpoints/configs from Notebook 1.
2. Run inference on plate crops (v4 + v5).
3. Annotate images with bbox + predicted text + confidence.
4. Create side-by-side comparison grid.
5. Batch-process all test images and run error analysis.

---

## 🖼️ Visual results

### Inference comparison grid (Original vs V4 vs V5)
Green box = YOLO plate region; Cyan = V4 text+conf; Yellow = V5 text+conf.

![Inference grid](images/inference_grid_v4_vs_v5.jpg)

### Training evaluation snapshot
(Accuracy and NED from the validation set.)

![Training evaluation](images/training_eval_metrics.jpg)

---

## 🔍 Error analysis
The notebook produces a 6-panel error analysis plot (confidence distributions, scatter, agreement, text length, etc.).

![Error analysis](images/error_analysis.png.jpg)

---

## 🚀 How to run

### 1) Open notebooks in Google Colab
- Upload the notebooks, or open from GitHub.
- Enable GPU: `Runtime → Change runtime type → T4 GPU`.

### 2) Install dependencies (CUDA 13.0)
> PaddlePaddle 3.x must be installed from the official index (not standard PyPI).

```bash
pip uninstall paddlepaddle paddlepaddle-gpu paddleocr -y -q
python -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/ -q
pip install paddleocr==3.4.0 -q
```

### 3) Run notebooks in order
- Run **Notebook 1** fully to generate `output/v4_rec`, `output/v5_rec`, and logs.
- Then run **Notebook 2** to generate `output/notebook2/*` (CSV + images).

---

## 📌 Notes on low accuracy
Low exact-match accuracy and low confidence are expected here because training was limited to **10 epochs**, pseudo-labels contain noise, and pretrain→config mismatches can force partial training-from-scratch.

---

## 📁 Suggested repo structure
```text
.
├── PPOCR_License_Plate_Training_Formatted.ipynb
├── PPOCR_License_Plate_Inference_Formatted.ipynb
├── README.md
└── images/
    ├── dataset_crops.jpg
    ├── inference_grid_v4_vs_v5.jpg
    ├── error_analysis.png.jpg
    └── training_eval_metrics.jpg
```
