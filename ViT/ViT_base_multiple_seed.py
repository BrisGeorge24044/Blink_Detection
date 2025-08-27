# ViT final training and validation script
import os, time, random, csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor, ViTForImageClassification
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score, f1_score
)
from tqdm import tqdm

# Config
data_dir = "../A_dataset_split"        # expects subfolders: train/, val/ (ImageFolder layout)
batch_size = 32
epochs = 5
lr = 1e-4
SEEDS = [42, 123, 777]

# Saving
model_save_path_tpl = "vit_base_seed{seed}.pth"
csv_overall_path    = Path("vit_results_overall_per_seed.csv")
csv_perclass_path   = Path("vit_results_perclass_per_seed.csv")
csv_summary_overall = Path("vit_results_summary_overall.csv")
csv_summary_percls  = Path("vit_results_summary_perclass.csv")

# Timing config
ENABLE_WARMUP = True
WARMUP_STEPS  = 3

# Device (prefer CUDA, then MPS, else CPU)
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

def torch_sync():
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except AttributeError:
            pass

# Reproducibility helpers
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic flags (may be slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_worker_init_fn(seed: int):
    def _init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return _init_fn

# Preprocessing
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

# Prepare CSV writers
def init_csvs():
    # Overall per-seed
    if not csv_overall_path.exists():
        with csv_overall_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seed", "accuracy", "precision_macro", "recall_macro", "f1_macro", "ms_per_img", "fps"])
    # Per-class per-seed
    if not csv_perclass_path.exists():
        with csv_perclass_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seed", "class_name", "precision", "recall", "f1"])
    # Summaries overwritten each run

init_csvs()

# Main multi-seed loop
all_results = []   # list of dicts. Each contains overall + per-class arrays

for seed in SEEDS:
    print(f"\n===== Running seed {seed} =====")
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    # Datasets & Loaders
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    val_dataset   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=transform)
    class_names   = train_dataset.classes  # ImageFolder alphabetical order
    print(f"Classes: {class_names}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=0,
        worker_init_fn=make_worker_init_fn(seed),
        pin_memory=(device.type == "cuda"),
    )
    val_loader   = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=make_worker_init_fn(seed),
        pin_memory=(device.type == "cuda"),
    )

    # Model (re-initialize per seed)
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    # Replace classification head with 2-class head (new layer is randomly init -> seed matters)
    model.classifier = nn.Linear(model.classifier.in_features, 2)

    # Freeze backbone, train head
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    model.to(device)

    # Loss & Optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    #  Training
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"[seed {seed}] Epoch {epoch+1}/{epochs}")
        for i, (images, labels) in enumerate(loop, start=1):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=f"{(total_loss / i):.4f}")

        print(f"[seed {seed}] Epoch {epoch+1} Avg Loss: {total_loss / max(1, len(train_loader)):.4f}")

    # Validation (with inference timing)
    model.eval()

    # Optional warm-up to stabilize timings
    if ENABLE_WARMUP and len(val_loader) > 0:
        with torch.no_grad():
            for j, (images, _) in enumerate(val_loader):
                images = images.to(device)
                torch_sync()
                _ = model(images).logits
                torch_sync()
                if j + 1 >= WARMUP_STEPS:
                    break

    all_preds, all_labels = [], []
    total_images = 0
    total_ms = 0.0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"[seed {seed}] Validation"):
            images = images.to(device)

            # Time the forward pass
            torch_sync()
            t0 = time.perf_counter()
            outputs = model(images).logits
            torch_sync()
            t1 = time.perf_counter()

            batch_ms = (t1 - t0) * 1000.0
            total_ms += batch_ms
            total_images += images.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Metrics
    # Overall (macro = equal weight to each class)
    acc        = accuracy_score(all_labels, all_preds)
    prec_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec_macro  = recall_score(all_labels, all_preds, average="macro",  zero_division=0)
    f1_macro   = f1_score(all_labels, all_preds, average="macro",      zero_division=0)

    # Per-class (same class order as ImageFolder)
    prec_pc = precision_score(all_labels, all_preds, average=None, zero_division=0)
    rec_pc  = recall_score(all_labels, all_preds,  average=None, zero_division=0)
    f1_pc   = f1_score(all_labels, all_preds,      average=None, zero_division=0)

    print(f"\n[seed {seed}] Validation Performance (overall):")
    print(f"Accuracy: {acc:.4f} | Precision(macro): {prec_macro:.4f} | Recall(macro): {rec_macro:.4f} | F1(macro): {f1_macro:.4f}")

    print(f"\n[seed {seed}] Per-class metrics:")
    for cname, p, r, f in zip(class_names, prec_pc, rec_pc, f1_pc):
        print(f"  {cname:>12s} -> Precision: {p:.4f} | Recall: {r:.4f} | F1: {f:.4f}")

    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # Inference timing summary
    mean_ms_per_image = (total_ms / total_images) if total_images > 0 else float("nan")
    fps = 1000.0 / mean_ms_per_image if mean_ms_per_image > 0 else float("nan")
    print(f"[seed {seed}] Inference timing (validation): mean_per_image_ms={mean_ms_per_image:.4f} | ~FPS={fps:.4f}")

    # Save per-seed model (optional)
    save_path = model_save_path_tpl.format(seed=seed)
    torch.save(model.state_dict(), save_path)
    print(f"[seed {seed}] Model saved to '{save_path}'")

    # Save per-seed results to CSVs
    with csv_overall_path.open("a", newline="") as f:
        w = csv.writer(f)
        w.writerow([seed, acc, prec_macro, rec_macro, f1_macro, mean_ms_per_image, fps])

    with csv_perclass_path.open("a", newline="") as f:
        w = csv.writer(f)
        for cname, p, r, f1c in zip(class_names, prec_pc, rec_pc, f1_pc):
            w.writerow([seed, cname, p, r, f1c])

    # Store in memory for summary
    all_results.append({
        "seed": seed,
        "acc": acc,
        "prec_macro": prec_macro,
        "rec_macro": rec_macro,
        "f1_macro": f1_macro,
        "prec_pc": np.array(prec_pc, dtype=float),
        "rec_pc":  np.array(rec_pc,  dtype=float),
        "f1_pc":   np.array(f1_pc,   dtype=float),
        "ms_per_img": mean_ms_per_image,
        "fps": fps,
        "class_names": class_names,
    })

#  Summary across seeds
def summarize_scalar(key):
    vals = np.array([r[key] for r in all_results], dtype=float)
    mean = float(vals.mean())
    std  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return mean, std

acc_mean,  acc_std   = summarize_scalar("acc")
precM_mean, precM_std = summarize_scalar("prec_macro")
recM_mean,  recM_std  = summarize_scalar("rec_macro")
f1M_mean,   f1M_std   = summarize_scalar("f1_macro")
ms_mean,    ms_std    = summarize_scalar("ms_per_img")
fps_mean,   fps_std   = summarize_scalar("fps")

print("\n===== Summary over seeds (overall) =====")
print(f"Seeds: {SEEDS}")
print(f"Accuracy:         {acc_mean:.4f} ± {acc_std:.4f}")
print(f"Precision(macro): {precM_mean:.4f} ± {precM_std:.4f}")
print(f"Recall(macro):    {recM_mean:.4f} ± {recM_std:.4f}")
print(f"F1(macro):        {f1M_mean:.4f} ± {f1M_std:.4f}")
print(f"ms/img:           {ms_mean:.4f} ± {ms_std:.4f}  |  FPS: {fps_mean:.4f} ± {fps_std:.4f}")

# Per-class summaries
class_names = all_results[0]["class_names"]
prec_stack = np.stack([r["prec_pc"] for r in all_results], axis=0)  # [num_seeds, num_classes]
rec_stack  = np.stack([r["rec_pc"]  for r in all_results], axis=0)
f1_stack   = np.stack([r["f1_pc"]   for r in all_results], axis=0)

print("\n===== Summary over seeds (per class) =====")
perclass_summary_rows = []
for i, cname in enumerate(class_names):
    p_mean, p_std = float(prec_stack[:, i].mean()), float(prec_stack[:, i].std(ddof=1)) if len(SEEDS) > 1 else 0.0
    r_mean, r_std = float(rec_stack[:,  i].mean()), float(rec_stack[:,  i].std(ddof=1)) if len(SEEDS) > 1 else 0.0
    f_mean, f_std = float(f1_stack[:,   i].mean()), float(f1_stack[:,   i].std(ddof=1)) if len(SEEDS) > 1 else 0.0
    print(f"{cname:>12s} -> Precision: {p_mean:.4f} ± {p_std:.4f} | Recall: {r_mean:.4f} ± {r_std:.4f} | F1: {f_mean:.4f} ± {f_std:.4f}")
    perclass_summary_rows.append([cname, p_mean, p_std, r_mean, r_std, f_mean, f_std])

# Write summaries to CSV
with csv_summary_overall.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["seeds", "accuracy_mean", "accuracy_std", "precision_macro_mean", "precision_macro_std",
                "recall_macro_mean", "recall_macro_std", "f1_macro_mean", "f1_macro_std",
                "ms_per_img_mean", "ms_per_img_std", "fps_mean", "fps_std"])
    w.writerow([str(SEEDS), acc_mean, acc_std, precM_mean, precM_std, recM_mean, recM_std,
                f1M_mean, f1M_std, ms_mean, ms_std, fps_mean, fps_std])

with csv_summary_percls.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["class_name", "precision_mean", "precision_std", "recall_mean", "recall_std", "f1_mean", "f1_std"])
    w.writerows(perclass_summary_rows)

print(f"\nSaved per-seed overall metrics -> {csv_overall_path}")
print(f"Saved per-seed per-class metrics -> {csv_perclass_path}")
print(f"Saved overall summary -> {csv_summary_overall}")
print(f"Saved per-class summary -> {csv_summary_percls}")
