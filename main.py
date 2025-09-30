# -*- coding: utf-8 -*-

import os
import gc
import cv2
import sys
import csv
import time
import pickle
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from pytz import timezone

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")

from datasets import get_dataset
from models import get_model
from utils.transforms_utils import transform, augmentation
from utils.train_utils import *  # get_trainer, etc.
from utils.dataloader_utils import distribution_check, calculate_class_weights
import utils.train_config as train_config

# -----------------------------
# Device
# -----------------------------
def pick_device(prefer_idx=0):
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        idx = min(max(prefer_idx, 0), n-1)
        dev = torch.device(f"cuda:{idx}")
        print(f"[info] Using CUDA:{idx} - {torch.cuda.get_device_name(idx)}")
        return dev
    print("[info] Using CPU")
    return torch.device("cpu")

device = pick_device(prefer_idx=1)  # prefer cuda:1 as in the original setup

# -----------------------------
# Utils
# -----------------------------
def now_kr():
    return datetime.now(timezone('Asia/Seoul')).strftime('%Y%m%d_%H%M')

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
seed_all(42)

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def csv_logger_init(csv_path, headers):
    ensure_dir(Path(csv_path).parent)
    new_file = not Path(csv_path).exists()
    f = open(csv_path, "a", newline="")
    w = csv.writer(f)
    if new_file: w.writerow(headers)
    return f, w

def model_supports_nwd(model_name: str) -> bool:
    # Return True only for models that use lambda_nwd (e.g., DALN)
    return model_name.lower() in {"daln", "daln_nwd"}

# -----------------------------
# Core experiment
# -----------------------------
def experiment(args, logging=True):
    gc.collect()

    # ==== Dataset / Loader ====
    src_dataset = get_dataset(args['crop'], args['src_dataset']['kwargs'])
    data_size   = len(src_dataset)
    train_size  = int(data_size * 0.75)
    valid_size  = data_size - train_size
    print(f"[info] source dataset: train={train_size} valid={valid_size} total={data_size}")

    generator = torch.Generator().manual_seed(42)
    src_train_set, src_valid_set = random_split(src_dataset, [train_size, valid_size], generator=generator)

    src_train_loader = DataLoader(src_train_set, batch_size=args['src_dataset']['batch_size'],
                                  shuffle=True, num_workers=args['src_dataset']['num_workers'])
    src_valid_loader = DataLoader(src_valid_set, batch_size=args['src_dataset']['batch_size'],
                                  shuffle=False, num_workers=args['src_dataset']['num_workers'])

    # class weights
    class_weights = calculate_class_weights(src_train_loader).to(device)

    # target / test
    tgt_dataset = get_dataset(args['crop'], args['tgt_dataset']['kwargs'])
    print(f"[info] target dataset: {len(tgt_dataset)}")
    tgt_loader  = DataLoader(tgt_dataset, batch_size=args['tgt_dataset']['batch_size'],
                             shuffle=True, num_workers=args['tgt_dataset']['num_workers'])

    test_dataset = get_dataset(args['crop'], args['test_dataset']['kwargs'])
    print(f"[info] test dataset:   {len(test_dataset)}")
    test_loader  = DataLoader(test_dataset, batch_size=args['test_dataset']['batch_size'],
                              shuffle=False, num_workers=args['test_dataset']['num_workers'])

    # ==== Model / Trainer ====
    arch = args['model']['name']
    crop = args['crop']
    model = get_model(arch, args['model']['backbone'], args['model']['n_class']).to(device)
    print(f"[info] model: {arch} / backbone: {args['model']['backbone']} / n_class={args['model']['n_class']}")

    # Pass only the necessary kwargs to the trainer
    trainer_kwargs = dict(class_weights=class_weights)
    if model_supports_nwd(arch) and 'lambda_nwd' in args:
        trainer_kwargs['lambda_nwd'] = args['lambda_nwd']
    trainer = get_trainer(arch, model, device, **trainer_kwargs)

    # ==== Train config ====
    n_epoch    = args.get('n_epochs', 300)
    patience   = args.get('early_stop', 20)  # number of epochs without improvement before early stop
    monitor    = args.get('monitor', 'val_loss')  # 'val_loss' or 'val_f1'
    optimize_min = (monitor == 'val_loss')

    run_tag   = args.get('save_name', f"{arch}_run")
    save_dir  = args.get('save_path', f"exp/{crop}")
    ensure_dir(save_dir)
    best_ckpt = os.path.join(save_dir, f"{crop}_{arch}_best.pth")
    last_ckpt = os.path.join(save_dir, f"{crop}_{arch}_last.pth")

    # Optim / Sched
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # ==== Logging ====
    csv_path = os.path.join(save_dir, f"{run_tag}_{now_kr()}.csv")
    csv_f, csv_w = csv_logger_init(csv_path, headers=[
        "epoch","train_loss","train_f1","val_loss","val_acc","val_f1","lr"
    ])

    # ==== Train loop ====
    best_score = np.inf if optimize_min else -np.inf
    wait = 0
    print("=========== Start Training ===========")
    st = time.time()

    for epoch in tqdm(range(1, n_epoch+1)):
        # Train
        model, train_loss, train_f1 = trainer.train(src_train_loader, tgt_loader, optimizer, scheduler, epoch-1, n_epoch)
        # Validate
        val_loss, val_acc, val_f1 = trainer.evaluate(src_valid_loader)

        # Step scheduler (per epoch)
        scheduler.step()

        # Save last checkpoint each epoch (optional but handy)
        torch.save(model.state_dict(), last_ckpt)

        # Monitor
        current = val_loss if optimize_min else val_f1
        improved = (current < best_score) if optimize_min else (current > best_score)

        # CSV log
        if logging:
            lr = optimizer.param_groups[0]["lr"]
            csv_w.writerow([epoch, float(train_loss), float(train_f1), float(val_loss), float(val_acc), float(val_f1), lr])
            csv_f.flush()

        # Early stop & best save
        if improved:
            best_score = current
            wait = 0
            torch.save(model.state_dict(), best_ckpt)
            print(f"[best] epoch={epoch} {monitor}={current:.5f} -> saved: {best_ckpt}")
        else:
            wait += 1
            if wait >= patience:
                print(f"[early stop] no improvement in {patience} epochs.")
                break

    et = time.time() - st
    print(f"=========== Finished ({int(et//60)}m {int(et%60)}s) ===========")
    csv_f.close()
    gc.collect()

    # ==== Test ====
    # load best
    try:
        trainer.model.load_state_dict(torch.load(best_ckpt, map_location=device))
        print(f"[info] loaded best checkpoint: {best_ckpt}")
    except Exception as e:
        print(f"[warn] failed to load best checkpoint ({e}), using last.")
        trainer.model.load_state_dict(torch.load(last_ckpt, map_location=device))

    print("Test :", end=" ")
    test_loss, test_acc, test_f1 = trainer.evaluate(test_loader)
    print(f"loss={test_loss:.4f} acc={test_acc:.4f} f1={test_f1:.4f}")

    # Final save (optional)
    if args.get('save', True):
        run_date = now_kr()
        final_path = os.path.join(save_dir, f"{run_tag}_{run_date}.pth")
        torch.save(trainer.model.state_dict(), final_path)
        print(f"[saved] {final_path}")

# -----------------------------
# Batch runner
# -----------------------------
def run_all():
    # Load base args
    base_args = train_config.get_args()

    # Common settings (adjust as needed)
    base_args['n_epochs']   = 5
    base_args['early_stop'] = 25          # patience
    base_args['monitor']    = 'val_loss'  # 'val_loss' or 'val_f1'
    base_args['save']       = True
    base_args['save_path']  = "exp"       # per-model subfolders will be created

    crops  = ['apple']                     # support multiple crops if needed
    models = ['ddc', 'dcoral', 'dann', 'cdan', 'daln']  # try all models

    # Provide lambda_nwd only for models that need it (e.g., DALN)
    lambda_grid = [0.005, 0.01]           # ignored if not applicable

    for crop in crops:
        for model_name in models:
            # Per-model runner
            if model_supports_nwd(model_name):
                lams = lambda_grid
            else:
                lams = [None]  # placeholder

            for i, lam in enumerate(lams, start=1):
                args = dict(base_args)  # shallow copy
                args['crop'] = crop
                args['model'] = {'name': model_name, 'backbone': 'resnet18', 'n_class': 3}
                # Per-model save directory
                args['save_path'] = os.path.join(base_args['save_path'], f"{crop}_{model_name}")
                ensure_dir(args['save_path'])

                # Set lambda_nwd only if required
                if lam is not None:
                    args['lambda_nwd'] = lam

                args['save_name'] = f"{model_name}_iter{i}"

                print(f"""
#################################################################################
* dataset : {args['crop']} ( n_classes : {args['model']['n_class']} ) 
* architecture : ResNet18 + {args['model']['name'].upper()}
* bg augmentation : {args['src_dataset']['kwargs']['type'].split('/')[-1]}
* lambda_nwd : {args.get('lambda_nwd', 'N/A')}
#################################################################################
""")
                experiment(args, logging=True)

if __name__ == "__main__":
    run_all()
