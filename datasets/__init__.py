# datasets/__init__.py
from pathlib import Path
import os, pickle, glob
from typing import Dict, List, Tuple
from datasets.dataset import CustomDataset
from utils.transforms_utils import transform as default_transform, augmentation as default_augmentation

VALID_TYPES = {"src_bg_augmented", "src_lab", "src_real", "tgt", "tst"}

def _load_structure_or_scan(crop: str) -> Dict[str, List[str]]:
    """Try to load data/{crop}/exp_structure.pickle.
    If missing, build a naive split by scanning PV/images for source, and plantpathology/images for target/test.
    """
    root = Path(f"data/{crop}")
    pkl = root / "exp_structure.pickle"
    if pkl.exists():
        with open(pkl, "rb") as f:
            return pickle.load(f)

    # fallback: scan
    pv_imgs_dir = root / "PV/images"
    tgt_imgs_dir = root / "plantpathology/images"

    def list_images(d: Path) -> List[str]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        if not d.exists():
            return []
        return [p.name for p in d.iterdir() if p.suffix.lower() in exts]

    src_names = list_images(pv_imgs_dir)
    tgt_names = list_images(tgt_imgs_dir)

    # naive: 80%/10%/10% by name sort
    src_names.sort(); tgt_names.sort()
    n_src = len(src_names); n_tgt = len(tgt_names)
    s_train = src_names
    t_train = tgt_names[: int(n_tgt * 0.9)]
    t_test  = tgt_names[int(n_tgt * 0.9):]

    return {"source": s_train, "target": t_train, "test": t_test}

def get_dataset(crop: str, dataset_opts: dict, aug_type=None):
    """Factory that respects the public data layout under data/{crop}/..."""
    dtype = dataset_opts.get("type", None)
    if dtype not in VALID_TYPES:
        raise ValueError(f"Unknown dataset type '{dtype}'. Valid: {sorted(VALID_TYPES)}")

    tfm = dataset_opts.get("transform", default_transform)
    aug = dataset_opts.get("augmentation", None)

    # load structure or scan
    ds_struct = _load_structure_or_scan(crop)

    # roots relative to repo
    pv_root   = Path(f"data/{crop}/PV")
    pp_root   = Path(f"data/{crop}/plantpathology")

    # label pickles are optional
    pv_labels_pkl = pv_root / "pv_labels.pickle"
    pp_labels_pkl = pp_root / "apple_labels.pickle"

    if dtype in {"src_bg_augmented", "src_augmented"}:
        image_root = pv_root / "bg_composed"
        labels     = pv_labels_pkl if pv_labels_pkl.exists() else None
        flag       = "labeled"
        img_names  = ds_struct["source"]

    elif dtype == "src_lab":
        image_root = pv_root / "images"
        labels     = pv_labels_pkl if pv_labels_pkl.exists() else None
        flag       = "labeled"
        img_names  = ds_struct["source"]

    elif dtype == "src_real":
        image_root = pp_root / "images"
        labels     = pp_labels_pkl if pp_labels_pkl.exists() else None
        flag       = "labeled"
        img_names  = ds_struct["source"]

    elif dtype == "tgt":
        image_root = pp_root / "images"
        labels     = pp_labels_pkl if pp_labels_pkl.exists() else None
        flag       = "unlabeled"
        img_names  = ds_struct["target"]

    elif dtype == "tst":
        image_root = pp_root / "images"
        labels     = pp_labels_pkl if pp_labels_pkl.exists() else None
        flag       = "unlabeled"
        img_names  = ds_struct["test"]

    else:
        raise RuntimeError(f"Dataset type not handled: {dtype}")

    return CustomDataset(
        img_folder   = str(image_root),
        json_pickle  = str(labels) if labels else None,
        flag         = flag,
        img_names    = img_names,
        transform    = tfm,
        augmentation = aug,
    )
