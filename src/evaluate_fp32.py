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

from utils import decode_predictions, sigmoid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONNX_FP32_PATH = PROJECT_ROOT / "models" / "person_detector_aidge_fp32.onnx"
VAL_IMG_DIR = PROJECT_ROOT / "data" / "processed" / "images" / "val"
VAL_LBL_DIR = PROJECT_ROOT / "data" / "processed" / "labels" / "val"
RESULTS_DIR = PROJECT_ROOT / "results"


def evaluate_visual_fp32(threshold=0.40):
    aidge_core.Log.set_console_level(aidge_core.Level.Notice)

    # Sélectionne 16 images de validation contenant des personnes
    lbl_files = [p for p in sorted(VAL_LBL_DIR.glob("*.txt")) if p.stat().st_size > 0]
    rd.seed(42)
    selected_lbls = rd.sample(lbl_files, 16)

    imgs_raw = []
    gt_boxes_all = []
    for lbl_p in selected_lbls:
        img_p = VAL_IMG_DIR / f"{lbl_p.stem}.png"
        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        imgs_raw.append(img)
        boxes = []
        with open(lbl_p, "r", encoding="utf-8") as f:
            for line in f:
                _, xc, yc, w, h = map(float, line.strip().split())
                x = (xc - w / 2.0) * 160.0
                y = (yc - h / 2.0) * 120.0
                boxes.append([x, y, w * 160.0, h * 120.0])
        gt_boxes_all.append(boxes)

    batch_np = np.stack(imgs_raw).astype(np.float32)
    batch_norm = (batch_np / 127.5 - 1.0)[:, None, :, :]

    model = aidge_onnx.load_onnx(str(ONNX_FP32_PATH))
    aidge_core.remove_flatten(model)
    aidge_core.expand_metaops(model)
    model.forward_dims(dims=[[16, 1, 120, 160]], allow_data_dependency=True)
    model.compile("cpu", aidge_core.dtype.float32, dims=[[16, 1, 120, 160]])

    scheduler = aidge_core.SequentialScheduler(model)
    scheduler.forward(True, [aidge_core.Tensor(batch_norm)])

    raw_out = np.asarray(model.get_ordered_outputs()[0][0].get_operator().get_output(0))
    preds_all = decode_predictions(raw_out, confidence_threshold=threshold)

    fig, axes = plt.subplots(4, 4, figsize=(15, 11))
    axes = axes.reshape(-1)

    for i in range(16):
        axes[i].imshow(imgs_raw[i], cmap="gray", vmin=0, vmax=255)
        # Vérité terrain en VERT pointillé
        for gx, gy, gw, gh in gt_boxes_all[i]:
            axes[i].add_patch(
                patches.Rectangle((gx, gy), gw, gh, linewidth=1.2, edgecolor="lime", linestyle="--", fill=False)
            )
        # Prédictions AIDGE en ROUGE plein
        for px, py, pw, ph, score in preds_all[i]:
            axes[i].add_patch(
                patches.Rectangle((px, py), pw, ph, linewidth=1.8, edgecolor="red", fill=False)
            )
            axes[i].text(px, max(0, py - 2), f"{score:.2f}", fontsize=8, color="red",
                         bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1})
        axes[i].axis("off")

    fig.suptitle("Vert = Vraies personnes (COCO) | Rouge = Prédictions AIDGE FP32")
    plt.tight_layout()
    out_file = RESULTS_DIR / "eval_fp32_vs_gt.png"
    plt.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Planche comparative sauvegardée dans : {out_file}")


if __name__ == "__main__":
    evaluate_visual_fp32(threshold=0.50)
