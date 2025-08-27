# ViT with data augmentation final train and validation script
import os, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor, ViTForImageClassification
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

#  Config
data_dir = "../A_dataset_split"
batch_size = 32
epochs = 5
lr = 1e-4
SEEDS = [42, 123, 777]
MODEL_SAVE_TPL = "vit_base_da_seed{seed}.pth"

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

# Seeding helpers
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_worker_init_fn(seed: int):
    def _init(worker_id):
        s = seed + worker_id
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
    return _init

#  Preprocessing
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Your original augmentations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

# Multi-seed training
for seed in SEEDS:
    print(f" Running seed {seed}")
    set_seed(seed); g = torch.Generator(); g.manual_seed(seed)

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    val_dataset   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=val_transform)
    class_names   = train_dataset.classes
    print("Classes:", class_names)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              generator=g, num_workers=0, worker_init_fn=make_worker_init_fn(seed),
                              pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=0, worker_init_fn=make_worker_init_fn(seed),
                              pin_memory=(device.type == "cuda"))

    # Model
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    for p in model.parameters(): p.requires_grad = False
    for p in model.classifier.parameters(): p.requires_grad = True
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    # Training
    for epoch in range(epochs):
        model.train(); total_loss = 0.0
        loop = tqdm(train_loader, desc=f"[seed {seed}] Epoch {epoch+1}/{epochs}")
        for i, (images, labels) in enumerate(loop, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(); logits = model(images).logits
            loss = criterion(logits, labels); loss.backward(); optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=f"{(total_loss/i):.4f}")
        print(f"[seed {seed}] Epoch {epoch+1} Avg Loss: {total_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    if ENABLE_WARMUP:
        with torch.no_grad():
            for j, (images, _) in enumerate(val_loader):
                images = images.to(device)
                torch_sync(); _ = model(images).logits; torch_sync()
                if j+1 >= WARMUP_STEPS: break

    all_preds, all_labels, total_ms, total_images = [], [], 0.0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"[seed {seed}] Validation"):
            images = images.to(device)
            torch_sync(); t0 = time.perf_counter()
            logits = model(images).logits
            torch_sync(); t1 = time.perf_counter()
            total_ms += (t1-t0)*1000; total_images += images.size(0)
            preds = torch.argmax(logits, 1)
            all_preds.extend(preds.cpu().numpy()); all_labels.extend(labels.numpy())

    # Metrics
    acc        = accuracy_score(all_labels, all_preds)
    prec_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec_macro  = recall_score(all_labels,  all_preds, average="macro",  zero_division=0)
    f1_macro   = f1_score(all_labels,      all_preds, average="macro",  zero_division=0)
    prec_pc    = precision_score(all_labels, all_preds, average=None, zero_division=0)
    rec_pc     = recall_score(all_labels,    all_preds, average=None, zero_division=0)
    f1_pc      = f1_score(all_labels,        all_preds, average=None, zero_division=0)

    print(f"\n[seed {seed}] Overall: Acc={acc:.4f} | Prec(macro)={prec_macro:.4f} | Rec(macro)={rec_macro:.4f} | F1(macro)={f1_macro:.4f}")
    for cname, p, r, f1c in zip(class_names, prec_pc, rec_pc, f1_pc):
        print(f"  {cname:>10s} -> P={p:.4f} | R={r:.4f} | F1={f1c:.4f}")
    print("\nClassification report:"); print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    ms_per_img = total_ms/total_images; fps = 1000.0/ms_per_img
    print(f"[seed {seed}] Inference timing (val): ms/img={ms_per_img:.4f} | ~FPS={fps:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_TPL.format(seed=seed))
    print(f"[seed {seed}] Saved -> {MODEL_SAVE_TPL.format(seed=seed)}")
