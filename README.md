# FBR-UDA — Field‑Adaptive Background Recomposition + Unsupervised Domain Adaptation

<img width="5805" height="2603" alt="Fig_graphical_abstract" src="https://github.com/user-attachments/assets/4848efa6-e0de-4b73-88eb-43b9b1163a48" />


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
python main.py
```
Artifacts (checkpoints, logs) are saved under `exp/`.

---

⚙️ Configuration (pointer)

Default options live in utils/train_config.py.

Dataset types supported (see datasets/__init__.py):
src_bg_augmented, src_lab, src_real, tgt, tst → each maps to the folders shown above.

Models supported: ddc, dcoral, dann, cdan, daln (with ResNet backbones; see models/).

---

## 🗂 Repository Structure

```
.
├── datasets/                         # dataset loaders & helpers
│   ├── __init__.py                   # maps dataset "type" -> folder paths
│   └── dataset.py
├── models/                           # model builders (UDA variants)
│   ├── layers/                       # extra layers if any
│   ├── uda_losses/                   # UDA loss modules
│   ├── ddc.py  dcoral.py  dann.py  cdan.py  daln.py  ...
│   └── resnet.py  vanilaresnet.py
├── utils/
│   ├── dataloader_utils.py           # class weights, distribution checks
│   ├── train_utils.py                # trainer & loops
│   ├── transforms_utils.py           # transforms/augmentations
│   └── train_config.py               # default training options
├── 01_FBR_field_adaptive_background_recomposition.ipynb
├── main.py
├── requirements.txt
└── README.md
```

---

### 🌶️ Practical Case Study: P-Chili Pepper Dataset

This dataset was used to validate our framework on the challenging task of distinguishing **Viral Infection (BBWV2)** from **Abiotic Stress (Flooding)**, which share visually similar symptoms.

* **Download:** [P-Chili Pepper Dataset (v1.0)](https://github.com/LABA-SNU/FBR-UDA/releases/download/v1.0/P_Chili_Pepper_Dataset.zip)
* **Specs:** 4,102 images (Healthy / Abiotic Stress / Viral Infection) collected in a controlled biosafe facility.
* **Contributors:** Data collection and annotation were conducted by **Seong-Yeop Kim** & **Mungyeong Song** *(Seoul National University)*.


---

## 📜 Citation

If you use this repository, please cite the following manuscript (under review):

```
@misc{jeon2025lab2field,
  title        = {Bridging the Lab-to-Field Gap in Plant Disease Diagnosis through Unsupervised Domain Adaptation Enhanced by Background Recomposition},
  author       = {Jeon, Woosang and Kim, Taehyeong and Choi, Sanghyeok and Yang, Kyuseok},
  year         = {2025},
  note         = {Manuscript under review. Corresponding author: taehyeong.kim@snu.ac.kr},
  howpublished = {\url{https://github.com/LABA-SNU/FBR-UDA}}
}
```
