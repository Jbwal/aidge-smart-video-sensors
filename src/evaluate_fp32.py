import random as rd
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches

import aidge_core
import aidge_backend_cpu
import aidge_onnx

from config import IMAGE_HEIGHT, IMAGE_WIDTH
from utils import decode_predictions, evaluate_detections

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONNX_FP32_PATH = PROJECT_ROOT / "models" / "person_detector_aidge_v2_fp32.onnx"
VAL_IMG_DIR = PROJECT_ROOT / "data" / "processed" / "images" / "val"
VAL_LBL_DIR = PROJECT_ROOT / "data" / "processed" / "labels" / "val"
RESULTS_DIR = PROJECT_ROOT / "results"
BS = 16


def read_gt(lbl_p):
    boxes = []
    with open(lbl_p, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 5:
                continue
            _, xc, yc, w, h = map(float, p)
            boxes.append([(xc - w / 2) * IMAGE_WIDTH, (yc - h / 2) * IMAGE_HEIGHT, w * IMAGE_WIDTH, h * IMAGE_HEIGHT])
    return boxes


def load_model():
    model = aidge_onnx.load_onnx(str(ONNX_FP32_PATH))
    aidge_core.remove_flatten(model)
    aidge_core.expand_metaops(model)
    model.forward_dims(dims=[[BS, 1, IMAGE_HEIGHT, IMAGE_WIDTH]], allow_data_dependency=True)
    model.compile("cpu", aidge_core.dtype.float32, dims=[[BS, 1, IMAGE_HEIGHT, IMAGE_WIDTH]])
    return model, aidge_core.SequentialScheduler(model)


def run(model, scheduler, imgs):
    """imgs: liste d'images uint8 -> sortie brute [N,5,gh,gw] (batch complété si besoin)."""
    outs = []
    for s in range(0, len(imgs), BS):
        chunk = imgs[s:s + BS]
        n = len(chunk)
        chunk = chunk + [chunk[-1]] * (BS - n)
        x = (np.stack(chunk).astype(np.float32) / 127.5 - 1.0)[:, None]
        scheduler.forward(True, [aidge_core.Tensor(x)])
        outs.append(np.asarray(model.get_ordered_outputs()[0][0].get_operator().get_output(0)).copy()[:n])
    return np.concatenate(outs, 0)


def evaluate_visual_fp32(threshold=0.35, full_ap=True):
    aidge_core.Log.set_console_level(aidge_core.Level.Notice)
    model, scheduler = load_model()

    if full_ap:
        lbls = sorted(VAL_LBL_DIR.glob("*.txt"))
        imgs = [cv2.imread(str(VAL_IMG_DIR / f"{p.stem}.png"), cv2.IMREAD_GRAYSCALE) for p in lbls]
        gts = [read_gt(p) if p.stat().st_size > 0 else [] for p in lbls]
        raw = run(model, scheduler, imgs)
        dets = decode_predictions(raw, confidence_threshold=0.05, max_detections=30)
        m = evaluate_detections(dets, gts, 0.5, threshold)
        print(f"Val complète : AP50={m['ap']:.3f} | précision@{threshold}={m['precision']:.2f} | rappel@{threshold}={m['recall']:.2f}")

    lbl_files = [p for p in sorted(VAL_LBL_DIR.glob("*.txt")) if p.stat().st_size > 0]
    rd.seed(42)
    sel = rd.sample(lbl_files, 16)
    imgs_raw = [cv2.imread(str(VAL_IMG_DIR / f"{p.stem}.png"), cv2.IMREAD_GRAYSCALE) for p in sel]
    gt_all = [read_gt(p) for p in sel]
    preds_all = decode_predictions(run(model, scheduler, imgs_raw), confidence_threshold=threshold)

    fig, axes = plt.subplots(4, 4, figsize=(15, 11))
    axes = axes.reshape(-1)
    for i in range(16):
        axes[i].imshow(imgs_raw[i], cmap="gray", vmin=0, vmax=255)
        for gx, gy, gw, gh in gt_all[i]:
            axes[i].add_patch(patches.Rectangle((gx, gy), gw, gh, linewidth=1.2, edgecolor="lime", linestyle="--", fill=False))
        for px, py, pw, ph, score in preds_all[i]:
            axes[i].add_patch(patches.Rectangle((px, py), pw, ph, linewidth=1.8, edgecolor="red", fill=False))
            axes[i].text(px, max(0, py - 2), f"{score:.2f}", fontsize=8, color="red",
                         bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1})
        axes[i].axis("off")
    fig.suptitle("Vert = Vraies personnes (COCO) | Rouge = Prédictions AIDGE FP32 v2")
    plt.tight_layout()
    out_file = RESULTS_DIR / "eval_fp32_v2_vs_gt.png"
    plt.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Planche comparative sauvegardée dans : {out_file}")


if __name__ == "__main__":
    evaluate_visual_fp32(threshold=0.35)
