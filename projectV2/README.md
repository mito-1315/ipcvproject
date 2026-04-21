# Face Recognition v2 — ArcFace + InsightFace on ORL/AT&T Dataset

A deep-learning-based face recognition system using **ArcFace pretrained embeddings** via InsightFace. A single command generates every figure, table, and metric needed for an academic report.

---

## 1. Relationship to Project Version One

Project Version One (in `../Project Version One/`) used classical PCA + KNN/SVM with scikit-learn. **v2 replaces the feature extractor** with pretrained ArcFace 512-dim embeddings via InsightFace's `buffalo_l` model — no classical dimensionality reduction needed. **v1 is fully preserved and untouched.**

---

## 2. Dataset Description

| Property | Value |
|---|---|
| Database | ORL / AT&T Face Database |
| Subjects | 40 (s1–s40) |
| Images per subject | 10 (1.pgm–10.pgm) |
| Total images | 400 |
| Image size | 112 × 92 px, grayscale PGM |
| Resized to | 160 × 160 px (BGR for InsightFace) |
| Train / Test split | 280 / 120 (7/3 per subject, stratified) |

---

## 3. Folder Structure

```
projectV2/
├── src/
│   ├── data_loader.py        # PGM→BGR, folder tree, sample grid
│   ├── preprocessing.py      # 4 OpenCV filters + CLAHE, Canny, FFT, landmarks
│   ├── embeddings.py         # InsightFace ArcFace embedding with fallback
│   ├── classifier.py         # KNN cosine + CosineThresholdMatcher
│   ├── train.py              # Stratified split + training pipeline
│   ├── evaluate.py           # Confusion matrix, metrics, prediction grids
│   ├── predict.py            # CLI inference
│   └── visualization.py      # t-SNE, PCA 2D, pipeline diagram
├── outputs/
│   ├── 01_dataset_structure/ ├── 02_sample_images/
│   ├── 03_filters/           ├── 04_features/
│   ├── 05_results/           ├── 06_metrics/
│   ├── 07_pipeline/          └── 08_extra_ipcv/  ← new IPCV screenshots
├── models/
│   ├── arcface_embeddings.npy  ← cached 400×512
│   ├── labels.npy
│   └── classifier.pkl
├── planning_document.txt
├── config.yaml
├── main.py
├── requirements.txt
├── setup_venv.ps1
└── README.md
```

---

## 4. Virtual Environment Setup

### Option A — One-click (Windows PowerShell, Recommended)

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\setup_venv.ps1
```

### Option B — Manual

```powershell
# Windows
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **First run note:** InsightFace downloads the `buffalo_l` model (~300 MB) to `~/.insightface/` automatically on first execution. Internet connection required.

---

## 5. Installation

```bash
pip install -r requirements.txt
```

---

## 6. How to Run

```powershell
# Always activate venv first
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux

python main.py
```

**Expected runtime:** 5–15 minutes (embedding extraction is the bottleneck on CPU; t-SNE takes ~30s).

---

## 7. How to Run Individual Modules

```bash
# Extract embeddings only (and cache to models/)
python -c "
import yaml, cv2, numpy as np
from src.data_loader import load_dataset
from src.embeddings import init_insightface, extract_all_embeddings
cfg = yaml.safe_load(open('config.yaml'))
imgs, lbls, _ = load_dataset(cfg['paths']['dataset_dir'])
app = init_insightface()
extract_all_embeddings(app, imgs, lbls, cfg['insightface'], cfg['paths']['models_dir'])
"

# Train only (requires embeddings cached)
python -c "
import yaml, numpy as np
from src.train import train_pipeline
cfg = yaml.safe_load(open('config.yaml'))
embs = np.load('models/arcface_embeddings.npy')
lbls = np.load('models/labels.npy')
train_pipeline(embs, lbls, cfg)
"

# Predict on a single image (requires trained model)
python src/predict.py --image ../Dataset/s5/8.pgm

# Predict with custom output path
python src/predict.py --image ../Dataset/s10/3.pgm --output outputs/05_results/custom_demo.png
```

---

## 8. Output Guide

| Folder | Contents | Report Section |
|---|---|---|
| `outputs/01_dataset_structure/` | folder_structure.txt, folder_tree_visualization.png | Dataset Description |
| `outputs/02_sample_images/` | samples_per_class.png, images_per_class_table.csv | Dataset Description |
| `outputs/03_filters/` | 4 before/after PNGs, filter_explanations.txt | Preprocessing |
| `outputs/04_features/` | t-SNE, PCA 2D, embedding histogram, HOG viz, comparison CSV | Feature Extraction |
| `outputs/05_results/` | prediction grids, inference_demo_single.png | Results |
| `outputs/06_metrics/` | confusion matrix, classification report, metrics CSV, tuning CSV | Evaluation |
| `outputs/07_pipeline/` | pipeline_diagram.png | Introduction / Methodology |
| `outputs/08_extra_ipcv/` | CLAHE, Canny, FFT, histogram CLAHE, landmarks, annotated preds | IPCV Screenshots 1–8 |

---

## 9. Report Artifact Checklist

| # | Artifact | File Path |
|---|---|---|
| 1 | Pipeline diagram | `outputs/07_pipeline/pipeline_diagram.png` |
| 2 | Folder structure screenshot | `outputs/01_dataset_structure/folder_tree_visualization.png` |
| 3 | Folder structure text | `outputs/01_dataset_structure/folder_structure.txt` |
| 4 | Sample images (40 classes) | `outputs/02_sample_images/samples_per_class.png` |
| 5 | Image count per class | `outputs/02_sample_images/images_per_class_table.csv` |
| 6 | Before/after histogram eq | `outputs/03_filters/before_after_histogram_eq.png` |
| 7 | Before/after Gaussian blur | `outputs/03_filters/before_after_gaussian.png` |
| 8 | Before/after median filter | `outputs/03_filters/before_after_median.png` |
| 9 | Before/after sharpening | `outputs/03_filters/before_after_sharpen.png` |
| 10 | Filter code + rationale | `outputs/03_filters/filter_explanations.txt` |
| 11 | t-SNE embedding plot | `outputs/04_features/arcface_embedding_tsne.png` |
| 12 | PCA 2D projection | `outputs/04_features/arcface_embedding_pca.png` |
| 13 | Embedding histogram | `outputs/04_features/embedding_histogram.png` |
| 14 | HOG visualization | `outputs/04_features/hog_visualization_sample.png` |
| 15 | Feature comparison table | `outputs/04_features/feature_comparison_table.csv` |
| 16 | Predictions with bboxes | `outputs/05_results/predictions_with_bboxes.png` |
| 17 | Single inference demo | `outputs/05_results/inference_demo_single.png` |
| 18 | Performance metrics | `outputs/06_metrics/performance_metrics.csv` |
| 19 | Confusion matrix | `outputs/06_metrics/confusion_matrix.png` |
| 20 | Before/after tuning | `outputs/06_metrics/before_after_tuning.csv` |
| **IPCV Extra Screenshots** | | |
| 21 | Screenshot 1&2: CLAHE comparison | `outputs/08_extra_ipcv/clahe_comparison.png` |
| 22 | Screenshot 3: Canny edge map | `outputs/08_extra_ipcv/canny_edge_map.png` |
| 23 | Screenshot 4: Histogram before/after CLAHE | `outputs/08_extra_ipcv/histogram_clahe_comparison.png` |
| 24 | Screenshot 5: Fourier Transform spectrum | `outputs/08_extra_ipcv/fourier_transform_spectrum.png` |
| 25 | Screenshot 6: RetinaFace 5 landmarks | `outputs/08_extra_ipcv/retinaface_landmarks.png` |
| 26 | Screenshot 7: Annotated predictions | `outputs/08_extra_ipcv/annotated_predictions.png` |
| 27 | Screenshot 8: Confusion matrix heatmap | `outputs/06_metrics/confusion_matrix.png` |

---

## 10. Demo Video Instructions (2-minute MP4)

1. **(0:00–0:20)** Open PowerShell, activate venv: `venv\Scripts\activate`
2. **(0:20–1:00)** Run `python main.py` — show terminal output scrolling with ✅ marks
3. **(1:00–1:30)** Open Windows Explorer → `outputs/` and display:
   - `outputs/07_pipeline/pipeline_diagram.png`
   - `outputs/04_features/arcface_embedding_tsne.png`
   - `outputs/06_metrics/confusion_matrix.png`
   - `outputs/08_extra_ipcv/retinaface_landmarks.png`
4. **(1:30–2:00)** Live inference: `python src/predict.py --image ../Dataset/s10/5.pgm`
   Then open `outputs/05_results/inference_demo_single.png`

---

## 11. Submission Packaging

```powershell
# Package source code only (exclude venv, outputs, models, Dataset)
python -c "
import zipfile, os
excl = ['venv', 'outputs', 'models', '__pycache__', '.git']
with zipfile.ZipFile('submission_v2.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in excl]
        for f in files:
            fp = os.path.join(root, f)
            if not any(e in fp for e in excl):
                zf.write(fp)
print('submission_v2.zip created.')
"
```

---

## 12. Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: insightface` | Activate venv and run `pip install insightface` |
| InsightFace download fails | Ensure internet access; model downloads to `~/.insightface/` |
| `ONNX Runtime error` | Run `pip install onnxruntime` (auto-installed with insightface) |
| `FileNotFoundError: Dataset` | Ensure `../Dataset/s1/` … `../Dataset/s40/` exist |
| `FileNotFoundError: classifier.pkl` | Run `python main.py` first before using `predict.py` |
| Slow embedding extraction | Expected ~1-3 min on CPU; GPU not required |
| PowerShell script blocked | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| t-SNE takes too long | Reduce `tsne.n_iter` in `config.yaml` (e.g., 500) |

---

## 13. Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Image I/O, all filters, CLAHE, Canny, FFT |
| `insightface` | ArcFace embedding extraction (RetinaFace + ArcFace) |
| `onnxruntime` | InsightFace ONNX model backend (CPU) |
| `scikit-learn` | KNN classifier, stratified split, metrics |
| `scikit-image` | HOG visualization |
| `matplotlib` | All visualizations, pipeline diagram |
| `seaborn` | Confusion matrix heatmap |
| `numpy` | Array operations, embedding math |
| `pandas` | CSV generation |
| `pyyaml` | Config loading |
| `joblib` | Model serialization |
| `Pillow` | Supplementary image I/O |

Install all: `pip install -r requirements.txt`
