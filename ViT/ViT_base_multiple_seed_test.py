# ViT final test script
import os, time, csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor, ViTForImageClassification
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score, f1_score
)

# Config
data_dir = "../A_dataset_split"           # expects test/ with class subfolders
batch_size = 32

# Models to test (must exist on disk)
SEEDS = [42, 123, 777]
MODEL_PATH_TPL = "vit_base_seed{seed}.pth"   # e.g., vit_base_seed42.pth

# Binary metrics: which class is considered "positive"?
POSITIVE_CLASS_NAME = "blink"  # set to "blink" or "no_blink"

# Optional CSV outputs (set to None to skip)
CSV_PER_SEED_OVERALL = Path("vit_test_overall_per_seed.csv")
CSV_PER_SEED_PERCLS  = Path("vit_test_perclass_per_seed.csv")
CSV_SUMMARY_OVERALL  = Path("vit_test_summary_overall.csv")
CSV_SUMMARY_PERCLS   = Path("vit_test_summary_perclass.csv")

# Timing config
ENABLE_WARMUP = True
WARMUP_STEPS  = 3   # number of batches to warm up (not timed)

# Device (prefer CUDA, then MPS, else CPU)
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
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

#  Preprocessing
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

#  Dataset & DataLoader
test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=test_transform)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
class_names  = test_dataset.classes  # alphabetical by folder name
print("Classes:", class_names)

# Map POSITIVE_CLASS_NAME to label index for binary metrics
if POSITIVE_CLASS_NAME not in class_names:
    raise ValueError(f"POSITIVE_CLASS_NAME='{POSITIVE_CLASS_NAME}' not found in classes {class_names}")
pos_label = class_names.index(POSITIVE_CLASS_NAME)

# Prepare CSVs
def init_csv(path: Path, header: list[str]):
    if path is None:
        return
    if not path.exists():
        with path.open("w", newline="") as f:
            csv.writer(f).writerow(header)

init_csv(CSV_PER_SEED_OVERALL, ["seed","accuracy","precision_macro","recall_macro","f1_macro","precision_bin","recall_bin","f1_bin","ms_per_img","fps"])
init_csv(CSV_PER_SEED_PERCLS,  ["seed","class_name","precision","recall","f1"])
# (summaries written at end)

#  Evaluation helper
def load_model(weights_path: str) -> nn.Module:
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    # Ensure 2-class head (same shape as training)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def evaluate_model(model: nn.Module):
    # warm-up to stabilize timings
    if ENABLE_WARMUP and len(test_loader) > 0:
        with torch.no_grad():
            for j, (images, _) in enumerate(test_loader):
                images = images.to(device)
                torch_sync(); _ = model(images).logits; torch_sync()
                if j + 1 >= WARMUP_STEPS:
                    break

    all_preds, all_labels = [], []
    total_images = 0
    total_ms = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            torch_sync()
            t0 = time.perf_counter()
            outputs = model(images).logits
            torch_sync()
            t1 = time.perf_counter()

            total_ms += (t1 - t0) * 1000.0
            total_images += images.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Overall metrics
    acc        = accuracy_score(all_labels, all_preds)
    prec_macro = precision_score(all_labels, all_preds, average="macro",  zero_division=0)
    rec_macro  = recall_score(all_labels,  all_preds, average="macro",   zero_division=0)
    f1_macro   = f1_score(all_labels,      all_preds, average="macro",   zero_division=0)

    # Binary metrics for chosen positive class
    prec_bin = precision_score(all_labels, all_preds, average="binary", pos_label=pos_label, zero_division=0)
    rec_bin  = recall_score(all_labels,    all_preds, average="binary", pos_label=pos_label, zero_division=0)
    f1_bin   = f1_score(all_labels,        all_preds, average="binary", pos_label=pos_label, zero_division=0)

    # Per-class
    prec_pc = precision_score(all_labels, all_preds, average=None, zero_division=0)
    rec_pc  = recall_score(all_labels,    all_preds, average=None, zero_division=0)
    f1_pc   = f1_score(all_labels,        all_preds, average=None, zero_division=0)

    # Timing
    mean_ms_per_image = (total_ms / total_images) if total_images > 0 else float("nan")
    fps = 1000.0 / mean_ms_per_image if mean_ms_per_image > 0 else float("nan")

    return {
        "acc": acc,
        "prec_macro": prec_macro, "rec_macro": rec_macro, "f1_macro": f1_macro,
        "prec_bin": prec_bin, "rec_bin": rec_bin, "f1_bin": f1_bin,
        "prec_pc": np.array(prec_pc, dtype=float),
        "rec_pc":  np.array(rec_pc,  dtype=float),
        "f1_pc":   np.array(f1_pc,   dtype=float),
        "ms_per_img": mean_ms_per_image, "fps": fps,
        "all_labels": np.array(all_labels), "all_preds": np.array(all_preds)
    }

# Run all seed models
results = []
for seed in SEEDS:
    weights_path = MODEL_PATH_TPL.format(seed=seed)
    if not os.path.exists(weights_path):
        print(f"[seed {seed}] WARNING: weights not found at '{weights_path}', skipping.")
        continue

    print(f"\n===== Testing seed {seed} | weights: {weights_path} =====")
    model = load_model(weights_path)
    metrics = evaluate_model(model)

    # Print overall
    print(f"[seed {seed}] Overall (macro): "
          f"Acc={metrics['acc']:.4f} | Prec={metrics['prec_macro']:.4f} | Rec={metrics['rec_macro']:.4f} | F1={metrics['f1_macro']:.4f}")
    print(f"[seed {seed}] Binary ({POSITIVE_CLASS_NAME} as positive): "
          f"Prec={metrics['prec_bin']:.4f} | Rec={metrics['rec_bin']:.4f} | F1={metrics['f1_bin']:.4f}")

    # Per-class
    print(f"\n[seed {seed}] Per-class metrics:")
    for cname, p, r, f1c in zip(class_names, metrics["prec_pc"], metrics["rec_pc"], metrics["f1_pc"]):
        print(f"  {cname:>12s} -> Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1c:.4f}")

    # Full report (optional)
    print("\nClassification report:")
    print(classification_report(metrics["all_labels"], metrics["all_preds"], target_names=class_names, digits=4))

    # Timing
    print(f"[seed {seed}] Inference timing (test): mean_per_image_ms={metrics['ms_per_img']:.4f} | ~FPS={metrics['fps']:.4f}")

    # Save per-seed rows
    if CSV_PER_SEED_OVERALL:
        with CSV_PER_SEED_OVERALL.open("a", newline="") as f:
            csv.writer(f).writerow([
                seed, metrics["acc"], metrics["prec_macro"], metrics["rec_macro"], metrics["f1_macro"],
                metrics["prec_bin"], metrics["rec_bin"], metrics["f1_bin"],
                metrics["ms_per_img"], metrics["fps"]
            ])
    if CSV_PER_SEED_PERCLS:
        with CSV_PER_SEED_PERCLS.open("a", newline="") as f:
            w = csv.writer(f)
            for cname, p, r, f1c in zip(class_names, metrics["prec_pc"], metrics["rec_pc"], metrics["f1_pc"]):
                w.writerow([seed, cname, p, r, f1c])

    # Keep for summary
    metrics["seed"] = seed
    results.append(metrics)

if not results:
    raise SystemExit("No seed models were found/evaluated.")

#  Summary across seeds
def summarize(vals: np.ndarray):
    mean = float(vals.mean())
    std  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return mean, std

acc_mean, acc_std           = summarize(np.array([r["acc"] for r in results]))
precM_mean, precM_std       = summarize(np.array([r["prec_macro"] for r in results]))
recM_mean, recM_std         = summarize(np.array([r["rec_macro"] for r in results]))
f1M_mean, f1M_std           = summarize(np.array([r["f1_macro"] for r in results]))
precB_mean, precB_std       = summarize(np.array([r["prec_bin"] for r in results]))
recB_mean, recB_std         = summarize(np.array([r["rec_bin"] for r in results]))
f1B_mean, f1B_std           = summarize(np.array([r["f1_bin"] for r in results]))
ms_mean, ms_std             = summarize(np.array([r["ms_per_img"] for r in results]))
fps_mean, fps_std           = summarize(np.array([r["fps"] for r in results]))

print("\n===== Test Summary over seeds (overall) =====")
print(f"Seeds: { [r['seed'] for r in results] }")
print(f"Accuracy:                 {acc_mean:.4f} ± {acc_std:.4f}")
print(f"Precision (macro):        {precM_mean:.4f} ± {precM_std:.4f}")
print(f"Recall (macro):           {recM_mean:.4f} ± {recM_std:.4f}")
print(f"F1 (macro):               {f1M_mean:.4f} ± {f1M_std:.4f}")
print(f"Precision (binary {POSITIVE_CLASS_NAME}): {precB_mean:.4f} ± {precB_std:.4f}")
print(f"Recall (binary {POSITIVE_CLASS_NAME}):    {recB_mean:.4f} ± {recB_std:.4f}")
print(f"F1 (binary {POSITIVE_CLASS_NAME}):        {f1B_mean:.4f} ± {f1B_std:.4f}")
print(f"ms/img:                   {ms_mean:.4f} ± {ms_std:.4f}  |  FPS: {fps_mean:.4f} ± {fps_std:.4f}")

# Per-class summaries
prec_stack = np.stack([r["prec_pc"] for r in results], axis=0)
rec_stack  = np.stack([r["rec_pc"]  for r in results], axis=0)
f1_stack   = np.stack([r["f1_pc"]   for r in results], axis=0)

print("\n===== Test Summary over seeds (per class) =====")
percls_rows = []
for i, cname in enumerate(class_names):
    p_mean, p_std = summarize(prec_stack[:, i])
    r_mean, r_std = summarize(rec_stack[:,  i])
    f_mean, f_std = summarize(f1_stack[:,   i])
    print(f"{cname:>12s} -> Precision: {p_mean:.4f} ± {p_std:.4f} | Recall: {r_mean:.4f} ± {r_std:.4f} | F1: {f_mean:.4f} ± {f_std:.4f}")
    percls_rows.append([cname, p_mean, p_std, r_mean, r_std, f_mean, f_std])

# Write summaries
if CSV_SUMMARY_OVERALL:
    with CSV_SUMMARY_OVERALL.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seeds","acc_mean","acc_std","prec_macro_mean","prec_macro_std","rec_macro_mean","rec_macro_std",
                    "f1_macro_mean","f1_macro_std","prec_bin_mean","prec_bin_std","rec_bin_mean","rec_bin_std",
                    "f1_bin_mean","f1_bin_std","ms_per_img_mean","ms_per_img_std","fps_mean","fps_std"])
        w.writerow([str([r['seed'] for r in results]), acc_mean, acc_std, precM_mean, precM_std, recM_mean, recM_std,
                    f1M_mean, f1M_std, precB_mean, precB_std, recB_mean, recB_std, f1B_mean, f1B_std, ms_mean, ms_std, fps_mean, fps_std])

if CSV_SUMMARY_PERCLS:
    with CSV_SUMMARY_PERCLS.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_name","precision_mean","precision_std","recall_mean","recall_std","f1_mean","f1_std"])
        w.writerows(percls_rows)
