# FBR-UDA — Field‑Adaptive Background Recomposition + Unsupervised Domain Adaptation (code‑only)

> Code and project structure for plant disease recognition under domain shift.  
> **No data or pretrained weights are included.** Bring your own datasets and follow the folder layout below.

---

## 🌱 Overview

Laboratory images (e.g., **PlantVillage**) often fail in real fields (e.g., **PlantPathology**) due to background/lighting/occlusion gaps.  
This repo provides two complementary components:

1. **FBR — Field‑Adaptive Background Recomposition**:  
   Segment foreground leaves with **SAM**, crop **real‑field** backgrounds, and composite them to create more field‑like training samples.
2. **UDA — Unsupervised Domain Adaptation**:  
   Train with source **labeled** (lab) and target **unlabeled** (field) data using methods such as **DDC**, **DCORAL**, **DANN**, **CDAN**, and **DALN**.

---

## 📂 Data (bring your own)

Use the **PlantVillage** and **PlantPathology** datasets. Download them yourself and place files to match the layout below.  


### Apple example layout
```
data/
└── apple/
    ├── PV/                         # lab‑collected (PlantVillage)
    │   ├── images/                 # raw lab images (flat or class subfolders)
    │   ├── bg_composed/            # FBR outputs (composited images)
    │   ├── pv_masks.pickle         # SAM masks index  (e.g., {filename: mask})
    │   └── pv_labels.pickle        # labels for PV
    └── plantpathology/             # real‑field (PlantPathology)
        ├── images/                 # raw field images
        ├── cropped_bg/             # background crops used for FBR
        └── apple_labels.pickle     # labels for field set
```

> You may adapt subfolders per crop/task, but **keep the same depth and names** so scripts can find your data (see `datasets/__init__.py`).

---

## ✂️ Create FBR data (Segmentation & Composition with SAM)

Run the notebook to generate FBR samples:

- **`01_FBR_field_adaptive_background_recomposition.ipynb`**
  1. Segment leaf foregrounds from lab images using **SAM**  
  2. Randomly crop **real-field** backgrounds  
  3. Composite foregrounds onto backgrounds → **FBR dataset**

**Weights**: large files (e.g., `sam_vit_h_4b8939.pth`) are **not tracked**. Download manually and set the local path in the notebook/config.

### Outputs produced by the FBR pipeline
- **Background crops** → `data/apple/plantpathology/cropped_bg/`  
  Files: `*.jpg` (e.g., `IMG_1234_crop00_s256_x12_y98.jpg`); size defaults to `256×256`.
- **SAM masks** → `data/apple/PV/pv_masks.pickle`  
  Python `dict` `{filename: mask}`, shape `(H, W)`, dtype `uint8` in `{0,255}` (post-processed for cleaner edges).
- **Composed (FBR) images** → `data/apple/PV/bg_composed/`  
  Same filenames as `PV/images/`; lab foreground pasted onto a random field crop.

> To train on composed images, set `src_dataset.kwargs.root = data/apple/PV/bg_composed/`.  
> For UDA, use real-field images under `data/apple/plantpathology/images/` as the target domain.

---

## 🚀 Quick Start

### 1) Install
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

### 2) Prepare data
- Download PlantVillage & PlantPathology
- Organize them to match the **data layout** above

### 3) Generate FBR in notebook
Open and run: `01_FBR_field_adaptive_background_recomposition.ipynb`

### 4) Train & evaluate
```bash
# default run
python main.py
```
Artifacts (checkpoints, logs) are saved under `exp/`.

---

## 🗂 Repository Structure

```
.
├── datasets/
├── models/
├── utils/
├── main.py
├── requirements.txt
└── README.md
```

---
