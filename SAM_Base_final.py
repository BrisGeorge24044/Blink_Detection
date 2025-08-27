# Base SAM final script
import os
import re
import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from segment_anything import sam_model_registry, SamPredictor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Configuration
SPLIT_ROOT = Path("../A_dataset_split")   # train/val/test with flat blink/no_blink
EYE_CENTERS_CSV = Path("../combined_DL_eye_centres.csv")  # cols: filename, abs_lx, abs_ly, abs_rx, abs_ry, open_eyes, person_id
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"  # base SAM ViT-B checkpoint (no fine-tuned weights)

MASK_THRESH = 0.5
BLINK_RATIO_THRESH = 0.5  # <50% of person baseline => blink

LABELS = ("blink", "no_blink")  # folder names; blink->1, no_blink->0

# Overlays
VAL_OVERLAY_DIR = Path("debug_masks_V8.4_val")
TEST_OVERLAY_DIR = Path("debug_masks_V8.4_test")
VAL_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
TEST_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

# Outputs
OUT_DIR = Path("eval_outputs_untrained_SAM")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Warm-up for inference time computation
ENABLE_WARMUP = True
WARMUP_STEPS  = 3

# Device
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
print(f"Using device: {device}")

def torch_sync():
    """Synchronize device for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except AttributeError:
            pass

# Helpers
PID_PREFIX_RE = re.compile(r"^([0-9]{3})[_-]?")  # '001_', '010-', etc.

def gather_images_flat(root: Path, split: str):
    """Return list[(label, path)] for split in {train,val,test} with flat blink/no_blink dirs."""
    base = root / split
    items = []
    for label_name in LABELS:
        folder = base / label_name
        if folder.exists():
            for p in folder.iterdir():
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    items.append((label_name, p))
    return items

def parse_pid_and_strip(basename: str):
    """Return ('001', 'original_without_pid_prefix.jpg') from '001_foo.jpg'."""
    b = basename.lower()
    m = PID_PREFIX_RE.match(b)
    if not m:
        pid = (b[:3] if len(b) >= 3 else b).zfill(3)
        stripped = b
    else:
        pid = m.group(1)
        stripped = b[m.end():]  # remove pid prefix + separator
    return pid, stripped

def mean_of_top_k(arr, k=10):
    if not arr:
        return np.nan
    a = np.asarray(arr, dtype=float)
    a.sort()
    top = a[-k:] if len(a) >= k else a
    return float(np.mean(top))

def save_overlay(image_bgr, mask_bin, eye_pts, out_path, text=None):
    colored = np.zeros_like(image_bgr)
    colored[:, :, 2] = mask_bin  # red channel
    overlay = cv2.addWeighted(image_bgr, 0.7, colored, 0.3, 0)
    for (px, py) in eye_pts:
        cv2.circle(overlay, (int(px), int(py)), 5, (0, 255, 0), -1)
    if text:
        cv2.putText(overlay, text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)

# Load eye centres (MultiIndex by person_id + stripped filename)
ec = pd.read_csv(EYE_CENTERS_CSV)
ec["filename"] = ec["filename"].astype(str).str.replace(":", "_").str.strip().str.lower()
ec["person_id"] = ec["person_id"].astype(str).str.zfill(3)
ec["stripped"] = ec["filename"].apply(lambda x: Path(x).name)  # ensure basename only
ec = ec.set_index(["person_id", "stripped"])

def lookup_eye_centres(pid: str, stripped_name: str):
    key = (pid, stripped_name)
    if key in ec.index:
        row = ec.loc[key]
        return float(row["abs_lx"]), float(row["abs_ly"]), float(row["abs_rx"]), float(row["abs_ry"])
    return None

# Load base SAM (no fine-tuned weights)
t_init0 = time.perf_counter()
sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
sam.to(device)
sam.eval()
predictor = SamPredictor(sam)
torch_sync()
t_init1 = time.perf_counter()
print(f"Model init time: {(t_init1 - t_init0)*1000:.4f} ms")

# Warm-up
if ENABLE_WARMUP:
    dummy = np.zeros((256, 256, 3), dtype=np.uint8)
    predictor.set_image(dummy, image_format="RGB")
    pts = np.array([[128, 128], [130, 130]], dtype=np.float32)
    pts_t = predictor.transform.apply_coords(pts, predictor.original_size)
    coords_t = torch.as_tensor(pts_t[None, :, :], dtype=torch.float32, device=device)
    labels_t = torch.as_tensor(np.ones(len(pts))[None], dtype=torch.int, device=device)
    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            torch_sync()
            _ = predictor.get_image_embedding()
            sparse_emb, dense_emb = sam.prompt_encoder(points=(coords_t, labels_t), boxes=None, masks=None)
            _m, _ = sam.mask_decoder(
                image_embeddings=predictor.get_image_embedding(),
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            torch_sync()

# Inference (with timing)
def predict_area_for_image(img_path: Path, abs_lx, abs_ly, abs_rx, abs_ry):
    """
    Returns (area_px, mask_bin_fullres, image_bgr, timing_dict)
    timing_dict: {'embed_ms', 'decode_ms', 'total_ms'}
    """
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        return None, None, None, None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # EMBEDDING timing
    t0 = time.perf_counter()
    predictor.set_image(image_rgb, image_format="RGB")
    pts = np.array([[abs_lx, abs_ly], [abs_rx, abs_ry]], dtype=np.float32)
    labs = np.ones(len(pts), dtype=int)
    pts_t = predictor.transform.apply_coords(pts, predictor.original_size)

    coords_t = torch.as_tensor(pts_t[None, :, :], dtype=torch.float32, device=device)
    labels_t = torch.as_tensor(labs[None], dtype=torch.int, device=device)

    torch_sync()
    t1 = time.perf_counter()
    with torch.no_grad():
        img_emb = predictor.get_image_embedding()
    torch_sync()
    t2 = time.perf_counter()
    embed_ms = (t2 - t1) * 1000.0

    # DECODER timing (prompt encoder + mask decoder + sigmoid)
    with torch.no_grad():
        t3 = time.perf_counter()
        sparse_emb, dense_emb = sam.prompt_encoder(points=(coords_t, labels_t), boxes=None, masks=None)
        low_res_masks, _ = sam.mask_decoder(
            image_embeddings=img_emb,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        pred = torch.sigmoid(low_res_masks.squeeze(0))  # (1, 256, 256)
        torch_sync()
        t4 = time.perf_counter()
    decode_ms = (t4 - t3) * 1000.0

    # POST
    mask_bin_small = (pred[0].detach().cpu().numpy() > MASK_THRESH).astype(np.uint8) * 255
    h, w = image_bgr.shape[:2]
    mask_bin = cv2.resize(mask_bin_small, (w, h), interpolation=cv2.INTER_NEAREST)
    area_px = int((mask_bin > 0).sum())

    total_ms = (t4 - t0) * 1000.0
    timing = {"embed_ms": embed_ms, "decode_ms": decode_ms, "total_ms": total_ms}
    return area_px, mask_bin, image_bgr, timing

# Build per-person baselines from TRAIN only
train_items = gather_images_flat(SPLIT_ROOT, "train")
print(f"Train images found: {len(train_items)}")
areas_by_person = {}

for label, img_path in tqdm(train_items, desc="TRAIN baseline pass"):
    basename = img_path.name
    pid, stripped = parse_pid_and_strip(basename)
    coords = lookup_eye_centres(pid, stripped)
    if coords is None:
        continue
    abs_lx, abs_ly, abs_rx, abs_ry = coords

    try:
        area_px, _, _, _ = predict_area_for_image(img_path, abs_lx, abs_ly, abs_rx, abs_ry)
        if area_px is None:
            continue
        areas_by_person.setdefault(pid, []).append(area_px)
    except Exception as e:
        print(f"Baseline error on {img_path}: {e}")
        continue

baselines = {pid: mean_of_top_k(vals, k=10) for pid, vals in areas_by_person.items()}
print(f"Computed baselines for {len(baselines)} participants from TRAIN.")

# Timing summary
def summarize_latency(df_timings: pd.DataFrame, split: str):
    if df_timings.empty:
        return
    def fmt(x): return f"{x:.4f}"
    mean_total = df_timings["total_ms"].mean()
    p50_total  = float(np.percentile(df_timings["total_ms"], 50))
    p90_total  = float(np.percentile(df_timings["total_ms"], 90))
    p95_total  = float(np.percentile(df_timings["total_ms"], 95))
    mean_embed = df_timings["embed_ms"].mean()
    mean_decode= df_timings["decode_ms"].mean()
    fps = 1000.0 / mean_total if mean_total > 0 else float("nan")
    print(f"\n[{split.upper()}] Inference timing (ms): "
          f"mean_total={fmt(mean_total)} | median={fmt(p50_total)} | p90={fmt(p90_total)} | p95={fmt(p95_total)} "
          f"| mean_embed={fmt(mean_embed)} | mean_decode={fmt(mean_decode)} | ~FPS={fps:.4f}")

# Common eval pass (VAL or TEST)
def run_pass(split: str, overlay_dir: Path):
    items = gather_images_flat(SPLIT_ROOT, split)
    records = []

    for idx, (label, img_path) in enumerate(tqdm(items, desc=f"{split.upper()} pass")):
        basename = img_path.name
        pid, stripped = parse_pid_and_strip(basename)
        coords = lookup_eye_centres(pid, stripped)
        if coords is None:
            continue
        abs_lx, abs_ly, abs_rx, abs_ry = coords

        try:
            area_px, mask_bin, image_bgr, timing = predict_area_for_image(img_path, abs_lx, abs_ly, abs_rx, abs_ry)
            if area_px is None:
                continue

            baseline = baselines.get(pid, np.nan)
            area_ratio = area_px / baseline if np.isfinite(baseline) and baseline > 0 else np.nan
            pred_label = int((area_ratio < BLINK_RATIO_THRESH)) if np.isfinite(area_ratio) else 1  # conservative default

            # GT from folder name
            true_label = 1 if label == "blink" else 0

            # overlays (all images for val/test)
            out_path = overlay_dir / label / basename
            txt = (f"pid={pid} area={area_px} base="
                   f"{0 if np.isnan(baseline) else int(baseline)} "
                   f"ratio={area_ratio:.4f}" if np.isfinite(area_ratio) else f"pid={pid} base=NaN")
            save_overlay(image_bgr, mask_bin, [(abs_lx, abs_ly), (abs_rx, abs_ry)], out_path, text=txt)

            row = {
                "split": split,
                "filename": img_path.as_posix(),
                "basename": basename.lower(),
                "person_id": pid,
                "area_px": area_px,
                "open_eye_area_px": baseline,
                "area_ratio": area_ratio,
                "pred_label": pred_label,   # 1=blink, 0=no_blink
                "true_label": true_label,
                "source_folder": label
            }
            if timing is not None:
                row.update(timing)

            records.append(row)

        except Exception as e:
            print(f"Error on {img_path}: {e}")
            continue

    df = pd.DataFrame(records)
    if len(df):
        # Print classification metrics (4 dp)
        y_true = df["true_label"].astype(int).values
        y_pred = df["pred_label"].astype(int).values
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec  = recall_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred)
        print(f"\n[{split.upper()}] Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        print(f"\n[{split.upper()}] Classification report:")
        print(classification_report(y_true, y_pred, digits=4))

        # Save predictions with timings (rounded to 4 dp for reporting columns)
        out_csv = OUT_DIR / f"predictions_{split}.csv"
        for c in ["embed_ms","decode_ms","total_ms"]:
            if c not in df.columns:
                df[c] = np.nan
        df["area_ratio"] = df["area_ratio"].astype(float).round(4)
        df["embed_ms"]   = df["embed_ms"].astype(float).round(4)
        df["decode_ms"]  = df["decode_ms"].astype(float).round(4)
        df["total_ms"]   = df["total_ms"].astype(float).round(4)

        cols = ["split","filename","basename","person_id","area_px","open_eye_area_px",
                "area_ratio","pred_label","true_label","source_folder",
                "embed_ms","decode_ms","total_ms"]
        df[cols].to_csv(out_csv, index=False)
        print(f"[{split.upper()}] Saved per-image predictions + timings to {out_csv}")

        # Timing summary (4 dp)
        tdf = df[["embed_ms","decode_ms","total_ms"]].dropna()
        if not tdf.empty:
            summarize_latency(tdf, split)
    else:
        print(f"\n[{split.upper()}] No images evaluated.")
    return df

# Run VAL then TEST
val_df  = run_pass("val",  VAL_OVERLAY_DIR)
test_df = run_pass("test", TEST_OVERLAY_DIR)