# Final non-data augmented MobileNetV2 model - test script
import os, time
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Configuration
data_dir = "../A_dataset_split"   # expects test/
batch_size = 32
SEEDS = [42, 123, 777]
MODEL_PATH_TPL = "mobilenetv2_base_seed{seed}.pth"

# Warm-up for inference time computation
ENABLE_WARMUP = True
WARMUP_STEPS  = 3

# Device
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
        try: torch.mps.synchronize()
        except AttributeError: pass

# Transforms
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# Dataset
test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=test_transform)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
class_names  = test_dataset.classes
print("Classes:", class_names)

# Helpers
def load_model(path: str) -> nn.Module:
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, 2)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device); model.eval()
    return model

def evaluate(model: nn.Module):
    if ENABLE_WARMUP and len(test_loader) > 0:
        with torch.no_grad():
            for j, (images, _) in enumerate(test_loader):
                images = images.to(device)
                torch_sync(); _ = model(images); torch_sync()
                if j+1 >= WARMUP_STEPS: break

    all_preds, all_labels = [], []
    total_ms, total_images = 0.0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            torch_sync(); t0 = time.perf_counter()
            logits = model(images)
            torch_sync(); t1 = time.perf_counter()
            total_ms += (t1 - t0) * 1000.0
            total_images += images.size(0)

            preds = torch.argmax(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc        = accuracy_score(all_labels, all_preds)
    prec_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec_macro  = recall_score(all_labels,  all_preds, average="macro",  zero_division=0)
    f1_macro   = f1_score(all_labels,      all_preds, average="macro",  zero_division=0)
    prec_pc    = precision_score(all_labels, all_preds, average=None, zero_division=0)
    rec_pc     = recall_score(all_labels,    all_preds, average=None, zero_division=0)
    f1_pc      = f1_score(all_labels,        all_preds, average=None, zero_division=0)

    ms_per_img = (total_ms / total_images) if total_images > 0 else float("nan")
    fps = 1000.0 / ms_per_img if ms_per_img > 0 else float("nan")

    return {"acc":acc,"prec_macro":prec_macro,"rec_macro":rec_macro,"f1_macro":f1_macro,
            "prec_pc":prec_pc,"rec_pc":rec_pc,"f1_pc":f1_pc,
            "ms":ms_per_img,"fps":fps,"labels":all_labels,"preds":all_preds}

# Run all seeds
results = []
for seed in SEEDS:
    path = MODEL_PATH_TPL.format(seed=seed)
    if not os.path.exists(path):
        print(f"[seed {seed}] Missing {path}, skipping."); continue
    print(f"\n===== Testing seed {seed} =====")
    m = evaluate(load_model(path))

    print(f"[seed {seed}] Overall: Acc={m['acc']:.4f} | Prec(macro)={m['prec_macro']:.4f} | Rec(macro)={m['rec_macro']:.4f} | F1(macro)={m['f1_macro']:.4f}")
    print(f"\n[seed {seed}] Per-class metrics:")
    for cname, p, r, f1c in zip(class_names, m["prec_pc"], m["rec_pc"], m["f1_pc"]):
        print(f"  {cname:>10s} -> P={p:.4f} | R={r:.4f} | F1={f1c:.4f}")
    print("\nClassification report:")
    print(classification_report(m["labels"], m["preds"], target_names=class_names, digits=4))
    print(f"[seed {seed}] Inference timing (test): ms/img={m['ms']:.4f} | ~FPS={m['fps']:.4f}")

    results.append(m)

if not results:
    raise SystemExit("No seed models evaluated. Check filenames and MODEL_PATH_TPL.")

# Summary
def summarize(vals):
    arr = np.array(vals, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

print("\n===== Test Summary over seeds (overall) =====")
for k, label in [("acc","Acc"),("prec_macro","Prec(macro)"),("rec_macro","Rec(macro)"),("f1_macro","F1(macro)")]:
    mean, std = summarize([r[k] for r in results])
    print(f"{label}: {mean:.4f} ± {std:.4f}")
ms_mean, ms_std = summarize([r["ms"] for r in results])
fps_mean, fps_std = summarize([r["fps"] for r in results])
print(f"ms/img: {ms_mean:.4f} ± {ms_std:.4f} | FPS: {fps_mean:.4f} ± {fps_std:.4f}")

print("\n===== Test Summary over seeds (per class) =====")
prec_stack = np.stack([r["prec_pc"] for r in results], axis=0)
rec_stack  = np.stack([r["rec_pc"]  for r in results], axis=0)
f1_stack   = np.stack([r["f1_pc"]   for r in results], axis=0)
for i, cname in enumerate(class_names):
    p_mean, p_std = summarize(prec_stack[:, i])
    r_mean, r_std = summarize(rec_stack[:,  i])
    f_mean, f_std = summarize(f1_stack[:,   i])
    print(f"{cname:>10s} -> P={f'{p_mean:.4f} ± {p_std:.4f}'} | R={f'{r_mean:.4f} ± {r_std:.4f}'} | F1={f'{f_mean:.4f} ± {f_std:.4f}'}")
