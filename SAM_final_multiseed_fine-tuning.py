# SAM final fine-tuning script
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import random
import torch
from torch import nn
from segment_anything import sam_model_registry, SamPredictor
from sklearn.model_selection import train_test_split

# Device
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# Seeds
SEEDS = [42, 123, 777]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Early stopping helper
class EarlyStopper:
    def __init__(self, patience=3, min_delta=1e-4, mode="max"):
        """
        patience: epochs to wait after last improvement
        min_delta: minimal change to be considered an improvement
        mode: "max" (metric increases) or "min" (metric decreases)
        """
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best = None
        self.bad = 0

    def update(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            self.bad = 0
            return True
        improved = (value - self.best) > self.min_delta if self.mode == "max" else (self.best - value) > self.min_delta
        if improved:
            self.best = value
            self.bad = 0
        else:
            self.bad += 1
        return improved

    def should_stop(self) -> bool:
        return self.bad >= self.patience

# Hyperparams
NUM_EPOCHS = 15
MIN_DELTA_FOR_BEST = 1e-4   # minimal Dice improvement to mark "best"
WEIGHT_DECAY = 1e-5
LR_DECODER = 2e-4
LR_PROMPT  = 5e-5
MAX_GRAD_NORM = 1.0

EARLY_PATIENCE = 3          # stop if no val Dice improvement for this many epochs
EARLY_MIN_DELTA = 1e-4      # same threshold as "best"

# Paths
BLINK_FOLDER = Path("../osfstorage-archive_blinks")
OPEN_EYE_FOLDER = Path("../osfstorage-archive_open_eyes")
CSV_PATH = Path("../combined_cleaned.csv")
EYE_CENTERS_CSV = Path("../combined_DL_eye_centres.csv")
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"

MODEL_SAVE_TPL = "sam_V12_seed{seed}.pth"            # last epoch
MODEL_SAVE_BEST_TPL = "sam_V12_seed{seed}_best.pth"  # best by val Dice

ANNOTATED_IMAGE_DIR = Path("annotation_images")
MASK_DIR = Path("SegmentationClass")
BASE_DEBUG_MASK_DIR = Path("debug_masks_V12")  # overlays saved per seed (final epoch only)
BASE_DEBUG_MASK_DIR.mkdir(exist_ok=True)

# Load metadata CSV
df_all = pd.read_csv(CSV_PATH)
df_all["filename"] = df_all["filename"].astype(str).apply(lambda x: Path(x).as_posix().strip().lower())

# Load eye centre coordinates
eye_centers_all = pd.read_csv(EYE_CENTERS_CSV)
eye_centers_all["filename"] = eye_centers_all["filename"].astype(str).str.replace(":", "_")
eye_centers_all = eye_centers_all.dropna(subset=["abs_lx", "abs_ly", "abs_rx", "abs_ry"]).copy()
eye_centers_all.set_index("filename", inplace=True)

# Filter for annotated files that exist
annotated_filenames = {f.name for f in ANNOTATED_IMAGE_DIR.glob("*.jpg")}
df_all = df_all[df_all["filename"].apply(lambda x: Path(x).name in annotated_filenames)]
df_all = df_all[df_all["filename"].isin(eye_centers_all.index)]

# Keep only files that actually exist on disk (blink/open-eye trees)
valid_files = []
for fname in df_all["filename"]:
    fname_base = Path(fname).name
    if list(BLINK_FOLDER.rglob(fname_base)) or list(OPEN_EYE_FOLDER.rglob(fname_base)):
        valid_files.append(fname)
df_all = df_all[df_all["filename"].isin(valid_files)]

print(f"Total valid annotated images: {len(df_all)}")

# Train/Val split (fixed once for fairness)
train_df_master, val_df_master = train_test_split(
    df_all, test_size=0.2, random_state=42, stratify=df_all["blink"]
)

# Metrics
bce_loss = nn.BCEWithLogitsLoss()

def dice_coef(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    pred_bin = (pred > 0.5).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum()
    return (2 * intersection + smooth) / (union + smooth)

def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    pred_bin = (pred > 0.5).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

# One pass over a DataFrame split
def process_batch(
    batch_df: pd.DataFrame,
    eye_centers: pd.DataFrame,
    predictor: SamPredictor,
    sam,
    optimizer: torch.optim.Optimizer | None,
    mode: str = "train",
    save_debug: bool = False,
    debug_dir: Path | None = None,
    shuffle_seed: int | None = None,
    epoch: int = 0,
):
    sam.train(mode == "train")

    if mode == "train":
        if shuffle_seed is None:
            df_iter = batch_df
        else:
            df_iter = batch_df.sample(frac=1.0, random_state=(shuffle_seed * 1000 + epoch)).reset_index(drop=True)
    else:
        df_iter = batch_df.reset_index(drop=True)

    running_loss = running_dice = running_iou = 0.0
    count = 0

    for _, row in tqdm(df_iter.iterrows(), total=len(df_iter), desc=f"{mode.upper()}"):
        filename = row["filename"]
        label = row["blink"]  # 1=blink, 0=open_eye in CSV

        # Eye centres
        try:
            eye = eye_centers.loc[filename]
        except KeyError:
            continue
        x_l, y_l, x_r, y_r = eye["abs_lx"], eye["abs_ly"], eye["abs_rx"], eye["abs_ry"]

        # File path (blink or open-eye folders)
        fname_base = Path(filename).name
        matches = list(BLINK_FOLDER.rglob(fname_base)) if label == 1 else list(OPEN_EYE_FOLDER.rglob(fname_base))
        if not matches:
            continue
        image_path = matches[0]

        # Load and prep image
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Prompt with eye centres
        predictor.set_image(image_rgb, image_format="RGB")
        point_coords = np.array([[x_l, y_l], [x_r, y_r]])
        point_labels = np.ones(len(point_coords), dtype=int)
        coords_transformed = predictor.transform.apply_coords(point_coords, predictor.original_size)
        coords_torch = torch.as_tensor(coords_transformed[None, :, :], dtype=torch.float32, device=device)
        labels_torch = torch.as_tensor(point_labels[None], dtype=torch.int, device=device)

        # Ground truth mask (256×256 for decoder)
        mask_filename = Path(filename).stem + ".png"
        mask_path = MASK_DIR / mask_filename
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(np.float32) / 255.0
            target_mask = torch.tensor(mask[None, :, :], dtype=torch.float32, device=device)
        else:
            target_mask = torch.zeros((1, 256, 256), dtype=torch.float32, device=device)

        try:
            with torch.no_grad():
                image_embedding = predictor.get_image_embedding()

            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=(coords_torch, labels_torch), boxes=None, masks=None
            )
            low_res_masks, _ = sam.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            pred_mask = low_res_masks.squeeze(0)                 # (1, 256, 256)
            loss = bce_loss(pred_mask, target_mask)
            pred_prob = torch.sigmoid(pred_mask)
            dice = dice_coef(pred_prob, target_mask)
            iou = iou_score(pred_prob, target_mask)

            if mode == "train":
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # clip grads for stability on small datasets
                torch.nn.utils.clip_grad_norm_(sam.parameters(), max_norm=MAX_GRAD_NORM)
                optimizer.step()

            # Save overlays (typically only final epoch)
            if mode == "val" and save_debug and debug_dir is not None:
                mask_bin = (pred_prob[0].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
                h, w = image_bgr.shape[:2]
                mask_resized = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)
                colored_mask = np.zeros_like(image_bgr)
                colored_mask[:, :, 2] = mask_resized  # red channel
                overlay = cv2.addWeighted(image_bgr, 0.7, colored_mask, 0.3, 0)
                for (px, py) in [(x_l, y_l), (x_r, y_r)]:
                    cv2.circle(overlay, (int(px), int(py)), 5, (0, 255, 0), -1)
                debug_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(debug_dir / fname_base), overlay)

            running_loss += loss.item()
            running_dice += dice.item()
            running_iou  += iou.item()
            count += 1

        except Exception as e:
            print(f"Error on {filename}: {e}")
            continue

    avg_loss = running_loss / max(count, 1)
    avg_dice = running_dice / max(count, 1)
    avg_iou  = running_iou  / max(count, 1)
    return avg_loss, avg_dice, avg_iou

# Train (multi-seed, keep best by Dice, with early stopping)
per_seed_results = []

for seed in SEEDS:
    print(f" Seed {seed}")
    set_seed(seed)

    train_df = train_df_master.copy()
    val_df   = val_df_master.copy()
    eye_centers = eye_centers_all

    # Fresh SAM per seed
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
    sam.to(device)

    # Freeze all except prompt encoder and mask decoder
    for _, p in sam.named_parameters():
        p.requires_grad = False
    for p in sam.prompt_encoder.parameters():
        p.requires_grad = True
    for p in sam.mask_decoder.parameters():
        p.requires_grad = True

    predictor = SamPredictor(sam)

    # Param groups & optimizer
    dec_params = list(sam.mask_decoder.parameters())
    pe_params  = list(sam.prompt_encoder.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": dec_params, "lr": LR_DECODER},
            {"params": pe_params,  "lr": LR_PROMPT},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    # Scheduler: step on validation Dice
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-6
    )

    DEBUG_MASK_DIR = BASE_DEBUG_MASK_DIR / f"seed_{seed}"

    best_val = {"epoch": -1, "loss": float("inf"), "dice": -1.0, "iou": -1.0}
    best_path = MODEL_SAVE_BEST_TPL.format(seed=seed)
    last_path = MODEL_SAVE_TPL.format(seed=seed)

    # Early stopping controller (monitor val Dice)
    early = EarlyStopper(patience=EARLY_PATIENCE, min_delta=EARLY_MIN_DELTA, mode="max")

    for epoch in range(NUM_EPOCHS):
        tr_loss, tr_dice, tr_iou = process_batch(
            train_df, eye_centers, predictor, sam, optimizer,
            mode="train", save_debug=False, debug_dir=None,
            shuffle_seed=seed, epoch=epoch
        )
        print(f"[seed {seed}] Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss={tr_loss:.4f} | Dice={tr_dice:.4f} | IoU={tr_iou:.4f}")

        save_debug = (epoch == NUM_EPOCHS - 1)  # overlays only on final epoch (set True to dump every epoch)
        val_loss, val_dice, val_iou = process_batch(
            val_df, eye_centers, predictor, sam, optimizer,
            mode="val", save_debug=save_debug, debug_dir=DEBUG_MASK_DIR,
            shuffle_seed=None, epoch=epoch
        )
        print(f"[seed {seed}] Validation        | Loss={val_loss:.4f} | Dice={val_dice:.4f} | IoU={val_iou:.4f}")

        # Step LR scheduler on the metric we're monitoring (Dice)
        scheduler.step(val_dice)

        # Track & save best (by Dice)
        if val_dice > best_val["dice"] + MIN_DELTA_FOR_BEST:
            best_val = {"epoch": epoch, "loss": val_loss, "dice": val_dice, "iou": val_iou}
            torch.save(sam.state_dict(), best_path)
            print(f"[seed {seed}] New best Dice={val_dice:.4f} at epoch {epoch+1}. Saved -> {best_path}")

        # Early stopping update
        improved = early.update(val_dice)
        if not improved:
            print(f"[seed {seed}] No val Dice improvement ({early.bad}/{EARLY_PATIENCE})")
        if early.should_stop():
            print(f"[seed {seed}] ⏹ Early stopping at epoch {epoch+1}. Best epoch: {best_val['epoch']+1}")
            break

    # Save last-epoch weights for reference
    torch.save(sam.state_dict(), last_path)
    print(f"[seed {seed}] Saved last-epoch model -> {last_path}")

    # Reload best weights so `sam` in memory is best (useful if you eval right away)
    if os.path.exists(best_path):
        sam.load_state_dict(torch.load(best_path, map_location=device))
        print(f"[seed {seed}] Reloaded best model from epoch {best_val['epoch']+1}")

    per_seed_results.append({
        "seed": seed,
        "val_loss": best_val["loss"],
        "val_dice": best_val["dice"],
        "val_iou":  best_val["iou"],
        "best_epoch": best_val["epoch"] + 1
    })

# Summary across seeds (best checkpoints)
def summarize(vals):
    arr = np.array(vals, dtype=float)
    mean = float(arr.mean())
    std  = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean, std

if per_seed_results:
    loss_mean, loss_std = summarize([r["val_loss"] for r in per_seed_results])
    dice_mean, dice_std = summarize([r["val_dice"] for r in per_seed_results])
    iou_mean,  iou_std  = summarize([r["val_iou"]  for r in per_seed_results])

    print("\n===== Validation Summary over seeds (BEST checkpoints via early stop/keep-best) =====")
    print(f"Seeds: {[r['seed'] for r in per_seed_results]}")
    print("Best epochs:", {r['seed']: r['best_epoch'] for r in per_seed_results})
    print(f"Loss: {loss_mean:.4f} ± {loss_std:.4f}")
    print(f"Dice: {dice_mean:.4f} ± {dice_std:.4f}")
    print(f"IoU : {iou_mean:.4f} ± {iou_std:.4f}")
else:
    print("No results collected over seeds.")

print("\nDone.")