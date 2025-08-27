# Final fine-tuned SAM validation and test script
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

# Config
SPLIT_ROOT = Path("../A_dataset_split")   # train/val/test with flat blink/no_blink
EYE_CENTERS_CSV = Path("../combined_DL_eye_centres.csv")  # filename, abs_lx, abs_ly, abs_rx, abs_ry, open_eyes, person_id
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"  # base SAM checkpoint
SEEDS = [42, 123, 777]

# Load BEST checkpoints (fallback to last-epoch if best missing)
SAM_WEIGHTS_BEST_TPL = "sam_V12_seed{seed}_best.pth"
SAM_WEIGHTS_LAST_TPL = "sam_V12_seed{seed}.pth"

MASK_THRESH = 0.5
BLINK_RATIO_THRESH = 0.5  # <50% of person baseline => blink

# Overlays (seed/split-specific dirs)
SAVE_OVERLAYS = True
OVERLAY_BASE_DIR = Path("debug_eval_split_flat_V12")
OVERLAY_BASE_DIR.mkdir(exist_ok=True)
OVERLAY_SAMPLE_EVERY = 1  # save every Nth image

# Outputs (seed/split-specific)
OUT_BASE_DIR = Path("eval_outputs_flat_V12")
OUT_BASE_DIR.mkdir(exist_ok=True)

# Class convention aligned with ViT/MobileNet:
# 0 = blink, 1 = no_blink
LABELS = ("blink", "no_blink")
print("Label mapping (evaluation): 0=blink, 1=no_blink")

# Warm up for inference time computation
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
PID_PREFIX_RE = re.compile(r"^([0-9]{3})[_-]?")  # first 3 digits + optional "_" or "-"

def gather_images_flat(root: Path, split: str):
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
    b = basename.lower()
    m = PID_PREFIX_RE.match(b)
    if not m:
        pid = b[:3].zfill(3) if len(b) >= 3 else b.zfill(3)
        stripped = b
    else:
        pid = m.group(1)
        stripped = b[m.end():]
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
        cv2.putText(overlay, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
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

# Inference helper
def predict_area_for_image(img_path: Path, abs_lx, abs_ly, abs_rx, abs_ry, predictor, sam):
    """
    Returns (area_px, mask_bin_fullres, image_bgr, timing_dict).
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
def build_train_baselines(predictor, sam):
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
            area_px, _, _, _ = predict_area_for_image(img_path, abs_lx, abs_ly, abs_rx, abs_ry, predictor, sam)
            if area_px is None:
                continue
            areas_by_person.setdefault(pid, []).append(area_px)
        except Exception as e:
            print(f"Baseline error on {img_path}: {e}")
            continue

    baselines = {pid: mean_of_top_k(vals, k=10) for pid, vals in areas_by_person.items()}
    print(f"Computed baselines for {len(baselines)} participants from TRAIN.")
    return baselines

# Timing summary (print to 4 dp)
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

# Eval pass (VAL or TEST)
def run_pass(split: str, baselines, predictor, sam, out_dir: Path, overlay_dir: Path):
    items = gather_images_flat(SPLIT_ROOT, split)
    records = []

    total_images = 0
    total_ms_all = 0.0

    for idx, (label, img_path) in enumerate(tqdm(items, desc=f"{split.upper()} pass")):
        basename = img_path.name
        pid, stripped = parse_pid_and_strip(basename)
        coords = lookup_eye_centres(pid, stripped)
        if coords is None:
            continue
        abs_lx, abs_ly, abs_rx, abs_ry = coords

        try:
            area_px, mask_bin, image_bgr, timing = predict_area_for_image(img_path, abs_lx, abs_ly, abs_rx, abs_ry, predictor, sam)
            if area_px is None:
                continue

            baseline = baselines.get(pid, np.nan)
            area_ratio = area_px / baseline if np.isfinite(baseline) and baseline > 0 else np.nan

            # 0=blink, 1=no_blink
            if np.isfinite(area_ratio):
                pred_label = 0 if (area_ratio < BLINK_RATIO_THRESH) else 1
            else:
                pred_label = 1  # conservative default: no_blink
            true_label = 0 if label == "blink" else 1

            # overlays
            if SAVE_OVERLAYS and (idx % OVERLAY_SAMPLE_EVERY == 0):
                out_path = overlay_dir / split / label / basename
                txt = (
                    f"pid={pid} area={area_px} base={0 if np.isnan(baseline) else int(baseline)} ratio={area_ratio:.4f}"
                    if np.isfinite(area_ratio) else f"pid={pid} base=NaN"
                )
                save_overlay(image_bgr, mask_bin, [(abs_lx, abs_ly), (abs_rx, abs_ry)], out_path, text=txt)

            row = {
                "split": split,
                "filename": img_path.as_posix(),
                "basename": basename.lower(),
                "person_id": pid,
                "area_px": area_px,
                "open_eye_area_px": baseline,
                "area_ratio": area_ratio,
                "pred_label": pred_label,   # 0=blink, 1=no_blink
                "true_label": true_label,
                "source_folder": label
            }
            if timing is not None:
                row.update(timing)
                total_ms_all += timing["total_ms"]
                total_images += 1

            records.append(row)

        except Exception as e:
            print(f"Error on {img_path}: {e}")
            continue

    df = pd.DataFrame(records)
    if len(df):
        # Overall metrics (macro) + per-class report
        y_true = df["true_label"].astype(int).values
        y_pred = df["pred_label"].astype(int).values

        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
        print(f"\n[{split.upper()}] Accuracy: {acc:.4f} | Precision(macro): {prec:.4f} | Recall(macro): {rec:.4f} | F1(macro): {f1:.4f}")

        print(f"\n[{split.upper()}] Classification report:")
        print(classification_report(
            y_true, y_pred,
            target_names=["blink", "no_blink"],  # 0 -> blink, 1 -> no_blink
            digits=4
        ))

        # Inference timing summary (mean ms/img and ~FPS)
        if total_images > 0:
            mean_ms_per_image = total_ms_all / total_images
            fps = 1000.0 / mean_ms_per_image if mean_ms_per_image > 0 else float("nan")
            print(f"[{split.upper()}] Inference timing: ms/img={mean_ms_per_image:.4f} | ~FPS={fps:.4f}")

        # Save predictions with timings (4 dp)
        out_csv = out_dir / f"predictions_{split}.csv"
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

        # More latency stats
        tdf = df[["embed_ms","decode_ms","total_ms"]].dropna()
        if not tdf.empty:
            summarize_latency(tdf, split)

        # return metrics for aggregation
        return {
            "acc": acc, "prec_macro": prec, "rec_macro": rec, "f1_macro": f1,
            "ms_per_img": mean_ms_per_image if total_images > 0 else float("nan"),
            "fps": fps if total_images > 0 else float("nan"),
            "df": df
        }
    else:
        print(f"\n[{split.upper()}] No images evaluated.")
        return None

# Evaluate a single seed (loads best)
def evaluate_seed(seed: int):
    best_path = SAM_WEIGHTS_BEST_TPL.format(seed=seed)
    last_path = SAM_WEIGHTS_LAST_TPL.format(seed=seed)

    weights_path = None
    if os.path.exists(best_path):
        weights_path = best_path
        which = "best"
    elif os.path.exists(last_path):
        weights_path = last_path
        which = "last"
    else:
        print(f"[seed {seed}] WARNING: no weights found ({best_path} or {last_path}). Skipping.")
        return None

    # Fresh model per seed
    t_init0 = time.perf_counter()
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
    state = torch.load(weights_path, map_location=device)
    sam.load_state_dict(state, strict=False)
    sam.to(device)
    sam.eval()
    predictor = SamPredictor(sam)
    torch_sync()
    t_init1 = time.perf_counter()
    print(f"[seed {seed}] Loaded '{which}' checkpoint: {weights_path}")
    print(f"[seed {seed}] Model init+load time: {(t_init1 - t_init0)*1000:.4f} ms")

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

    # Baselines from TRAIN for this (sam, predictor) pair
    baselines = build_train_baselines(predictor, sam)

    # Seed-specific dirs
    out_dir = OUT_BASE_DIR / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = OVERLAY_BASE_DIR / f"seed_{seed}"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # Run VAL and TEST
    val_metrics  = run_pass("val",  baselines, predictor, sam, out_dir, overlay_dir)
    test_metrics = run_pass("test", baselines, predictor, sam, out_dir, overlay_dir)

    return {
        "seed": seed,
        "val": val_metrics,
        "test": test_metrics
    }

# Run all seeds
results = []
for s in SEEDS:
    print(f"\n========== Evaluating seed {s} ==========")
    r = evaluate_seed(s)
    if r is not None:
        results.append(r)

# Summaries across seeds
def summarize(vals):
    arr = np.array(vals, dtype=float)
    mean = float(arr.mean()) if len(arr) else float("nan")
    std  = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean, std

def summarize_split(split_key: str):
    accs, precs, recs, f1s, ms, fps = [], [], [], [], [], []
    for r in results:
        m = r.get(split_key)
        if m:
            accs.append(m["acc"])
            precs.append(m["prec_macro"])
            recs.append(m["rec_macro"])
            f1s.append(m["f1_macro"])
            ms.append(m["ms_per_img"])
            fps.append(m["fps"])

    a_m, a_s = summarize(accs)
    p_m, p_s = summarize(precs)
    r_m, r_s = summarize(recs)
    f_m, f_s = summarize(f1s)
    ms_m, ms_s = summarize(ms)
    fps_m, fps_s = summarize(fps)

    print(f"\n===== {split_key.upper()} Summary over seeds (BEST checkpoints) =====")
    print(f"Seeds: {[r['seed'] for r in results if r.get(split_key)]}")
    print(f"Accuracy:         {a_m:.4f} ± {a_s:.4f}")
    print(f"Precision(macro): {p_m:.4f} ± {p_s:.4f}")
    print(f"Recall(macro):    {r_m:.4f} ± {r_s:.4f}")
    print(f"F1(macro):        {f_m:.4f} ± {f_s:.4f}")
    print(f"ms/img:           {ms_m:.4f} ± {ms_s:.4f}")
    print(f"~FPS:             {fps_m:.4f} ± {fps_s:.4f}")

if results:
    summarize_split("val")
    summarize_split("test")
else:
    print("No seeds evaluated.")

print("\nDone.")