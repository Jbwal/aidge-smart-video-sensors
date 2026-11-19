import csv
import math
import os
import random
from pathlib import Path
import cv2
import numpy as np

import aidge_core
import aidge_backend_cpu
import aidge_learning
import aidge_onnx

from config import (STRIDE, IMAGE_HEIGHT, IMAGE_WIDTH, GRID_H, GRID_W, MIN_BOX_W, MIN_BOX_H)
from model_aidge import create_person_detector_aidge
from utils import decode_predictions, sigmoid, evaluate_detections

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_IMG_DIR = PROJECT_ROOT / "data" / "processed" / "images" / "train"
TRAIN_LBL_DIR = PROJECT_ROOT / "data" / "processed" / "labels" / "train"
VAL_IMG_DIR = PROJECT_ROOT / "data" / "processed" / "images" / "val"
VAL_LBL_DIR = PROJECT_ROOT / "data" / "processed" / "labels" / "val"

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ONNX_FP32_PATH = MODEL_DIR / "person_detector_aidge_v2_fp32.onnx"
HISTORY_PATH = RESULTS_DIR / "training_history_aidge_v2.csv"

BATCH_SIZE = 16
EPOCHS = 60
LEARNING_RATE = 0.001
MIN_LR = 0.00002
BOX_WEIGHT = 2.0
RESUME = os.environ.get("RESUME", "0") == "1"   # reprise depuis le même ONNX v2 (même architecture)


# ----------------------------------------------------------------------------- données
def load_dataset_list(img_dir, lbl_dir, keep_neg_ratio=0.25):
    """
    Retourne [(chemin, boxes_pos, boxes_ignore)] avec boîtes en pixels xyxy.
    Les petites boîtes (< MIN_BOX) ne sont plus des 'négatifs' : elles deviennent des zones ignorées.
    """
    pos, neg = [], []
    for img_path in sorted(Path(img_dir).glob("*.png")):
        lbl_path = Path(lbl_dir) / f"{img_path.stem}.txt"
        boxes, ign = [], []
        if lbl_path.is_file() and lbl_path.stat().st_size > 0:
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    _, xc, yc, w, h = map(float, parts)
                    bw, bh = w * IMAGE_WIDTH, h * IMAGE_HEIGHT
                    box = [xc * IMAGE_WIDTH - bw / 2, yc * IMAGE_HEIGHT - bh / 2,
                           xc * IMAGE_WIDTH + bw / 2, yc * IMAGE_HEIGHT + bh / 2]
                    (boxes if (bw >= MIN_BOX_W and bh >= MIN_BOX_H) else ign).append(box)
        (pos if boxes else neg).append((str(img_path), boxes, ign))
    rng = random.Random(42)
    n_keep = int(len(pos) * keep_neg_ratio)
    neg = rng.sample(neg, min(len(neg), n_keep))
    allv = pos + neg
    rng.shuffle(allv)
    return allv


def random_zoom(img, boxes, ign):
    """Crop aléatoire (centré sur une personne 70 % du temps) agrandi à 160x120 -> gros plans."""
    s = random.uniform(0.45, 1.0)
    cw, ch = int(round(IMAGE_WIDTH * s)), int(round(IMAGE_HEIGHT * s))
    if boxes and random.random() < 0.7:
        b = random.choice(boxes)
        xc = (b[0] + b[2]) / 2 + random.uniform(-0.15, 0.15) * cw
        yc = (b[1] + b[3]) / 2 + random.uniform(-0.15, 0.15) * ch
        x0 = int(np.clip(xc - cw / 2, 0, IMAGE_WIDTH - cw))
        y0 = int(np.clip(yc - ch / 2, 0, IMAGE_HEIGHT - ch))
    else:
        x0 = random.randint(0, IMAGE_WIDTH - cw)
        y0 = random.randint(0, IMAGE_HEIGHT - ch)
    img = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    sx, sy = IMAGE_WIDTH / cw, IMAGE_HEIGHT / ch

    def tf(b):
        x1, y1, x2, y2 = (b[0] - x0) * sx, (b[1] - y0) * sy, (b[2] - x0) * sx, (b[3] - y0) * sy
        full = max(1e-6, (x2 - x1) * (y2 - y1))
        c = [float(np.clip(x1, 0, IMAGE_WIDTH)), float(np.clip(y1, 0, IMAGE_HEIGHT)),
             float(np.clip(x2, 0, IMAGE_WIDTH)), float(np.clip(y2, 0, IMAGE_HEIGHT))]
        area = max(0.0, c[2] - c[0]) * max(0.0, c[3] - c[1])
        return c, area / full

    new_boxes, new_ign = [], []
    for b in boxes:
        c, vis = tf(b)
        if c[2] - c[0] <= 1 or c[3] - c[1] <= 1:
            continue
        if vis >= 0.45 and (c[2] - c[0]) >= MIN_BOX_W and (c[3] - c[1]) >= MIN_BOX_H:
            new_boxes.append(c)
        else:
            new_ign.append(c)
    for b in ign:
        c, _ = tf(b)
        if c[2] - c[0] > 1 and c[3] - c[1] > 1:
            new_ign.append(c)
    return img, new_boxes, new_ign


GX = (np.arange(GRID_W, dtype=np.float32) + 0.5)[None, :] * STRIDE   # [1,GW]
GY = (np.arange(GRID_H, dtype=np.float32) + 0.5)[:, None] * STRIDE   # [GH,1]


def make_targets(boxes, ign):
    """
    Canaux : 0..3 = l,t,r,b (unités de stride) | 4 = cible de qualité gaussienne (soft) |
             5 = masque de positifs pour la régression | 6 = poids de la perte de confiance (0 = ignoré)
    """
    tgt = np.zeros((7, GRID_H, GRID_W), dtype=np.float32)
    tgt[6] = 1.0
    for x1, y1, x2, y2 in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        w, h = x2 - x1, y2 - y1
        xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
        sx = max(w / 6.0, STRIDE * 0.75)
        sy = max(h / 6.0, STRIDE * 0.75)
        g = np.exp(-0.5 * (((GX - xc) / sx) ** 2 + ((GY - yc) / sy) ** 2)).astype(np.float32)
        inside = (GX > x1) & (GX < x2) & (GY > y1) & (GY < y2)
        near = (np.abs(GX - xc) <= max(STRIDE, 0.2 * w)) & (np.abs(GY - yc) <= max(STRIDE, 0.2 * h))
        # cellule contenant le centre : toujours positive
        cgx = min(max(int(xc // STRIDE), 0), GRID_W - 1)
        cgy = min(max(int(yc // STRIDE), 0), GRID_H - 1)
        force = np.zeros((GRID_H, GRID_W), dtype=bool)
        force[cgy, cgx] = True
        pos = (inside & near) | force
        heat_region = inside | force

        tgt[4] = np.where(heat_region, np.maximum(tgt[4], g), tgt[4])
        tgt[0] = np.where(pos, np.maximum(0.25, (GX - x1) / STRIDE), tgt[0])
        tgt[1] = np.where(pos, np.maximum(0.25, (GY - y1) / STRIDE), tgt[1])
        tgt[2] = np.where(pos, np.maximum(0.25, (x2 - GX) / STRIDE), tgt[2])
        tgt[3] = np.where(pos, np.maximum(0.25, (y2 - GY) / STRIDE), tgt[3])
        tgt[5] = np.where(pos, 1.0, tgt[5])
    for x1, y1, x2, y2 in ign:
        inside = (GX > x1) & (GX < x2) & (GY > y1) & (GY < y2)
        tgt[6] = np.where(inside & (tgt[4] == 0.0), 0.0, tgt[6])
    return tgt


def build_batch_arrays(batch_samples, is_train=True):
    b_size = len(batch_samples)
    images_np = np.zeros((b_size, 1, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
    targets_np = np.zeros((b_size, 7, GRID_H, GRID_W), dtype=np.float32)
    for i, (img_path, orig_boxes, orig_ign) in enumerate(batch_samples):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        boxes = [list(b) for b in orig_boxes]
        ign = [list(b) for b in orig_ign]

        if is_train:
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
                boxes = [[IMAGE_WIDTH - b[2], b[1], IMAGE_WIDTH - b[0], b[3]] for b in boxes]
                ign = [[IMAGE_WIDTH - b[2], b[1], IMAGE_WIDTH - b[0], b[3]] for b in ign]
            if random.random() < 0.5:
                img, boxes, ign = random_zoom(img, boxes, ign)
            f = img.astype(np.float32)
            if random.random() < 0.5:
                f = f * random.uniform(0.7, 1.3) + random.uniform(-20, 20)
            if random.random() < 0.3:
                f = f + np.random.normal(0.0, 3.0, f.shape)
            img = np.clip(f, 0, 255)
        images_np[i, 0] = img.astype(np.float32) / 127.5 - 1.0
        targets_np[i] = make_targets(boxes, ign)
    return images_np, targets_np


# ----------------------------------------------------------------------------- perte
def compute_loss_and_grad(pred_np, tgt, box_weight=BOX_WEIGHT):
    """
    - Confiance : Quality Focal Loss (cible soft = qualité gaussienne), gradient analytique.
    - Boîtes : -log(IoU) des distances LTRB sur les cellules positives, pondéré par la qualité.
    Les deux sont normalisées par le nombre de positifs (Adam est de toute façon invariant à l'échelle).
    """
    eps = 1e-6
    pred = np.nan_to_num(pred_np, nan=0.0, posinf=50.0, neginf=-50.0)
    grad = np.zeros_like(pred, dtype=np.float32)

    # ---- confiance (QFL, beta=2)
    p = sigmoid(np.clip(pred[:, 4], -20.0, 20.0))
    y, pos, cw = tgt[:, 4], tgt[:, 5] > 0.5, tgt[:, 6]
    n_pos = max(1.0, float(pos.sum()))
    d = p - y
    ce = -(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
    cls_loss = float((ce * d ** 2 * cw).sum() / n_pos)
    grad[:, 4] = d * (d ** 2 + 2.0 * p * (1.0 - p) * ce) * cw / n_pos

    # ---- boîtes (-log IoU)
    wgt = np.where(pos, np.maximum(y, 0.25), 0.0)
    denom = max(1e-6, float(wgt.sum()))
    box_loss = 0.0
    if wgt.sum() > 0:
        l, t, r, b = [np.maximum(pred[:, k], 1e-2) for k in range(4)]
        gl, gt, gr, gb = [tgt[:, k] for k in range(4)]
        wi = np.minimum(l, gl) + np.minimum(r, gr)
        hi = np.minimum(t, gt) + np.minimum(b, gb)
        inter = wi * hi
        union = (l + r) * (t + b) + (gl + gr) * (gt + gb) - inter
        iou = inter / (union + eps)
        box_loss = float((wgt * -np.log(iou + eps)).sum() / denom)

        dI = [hi * (l < gl), wi * (t < gt), hi * (r < gr), wi * (b < gb)]
        dU = [(t + b) - dI[0], (l + r) - dI[1], (t + b) - dI[2], (l + r) - dI[3]]
        for k in range(4):
            g_k = -dI[k] / (inter + eps) + dU[k] / (union + eps)
            grad[:, k] = box_weight * wgt / denom * np.clip(g_k, -5.0, 5.0)

    total = cls_loss + box_weight * box_loss
    return total, cls_loss, box_loss, np.clip(grad, -5.0, 5.0).astype(np.float32)


# ----------------------------------------------------------------------------- entraînement
def load_weights_from_onnx(model, path):
    onnx_m = aidge_onnx.load_onnx(str(path))
    src = [np.asarray(p.get_operator().get_output(0)).copy() for p in aidge_core.producers(onnx_m, False)]
    dst = list(aidge_core.producers(model, False))
    if len(src) != len(dst):
        print("Reprise ignorée : architecture différente.")
        return
    for prod, arr in zip(dst, src):
        if tuple(prod.get_operator().get_output(0).dims) != arr.shape:
            print("Reprise ignorée : formes différentes.")
            return
    for prod, arr in zip(dst, src):
        prod.get_operator().set_output(0, aidge_core.Tensor(arr))
    print(f"Poids rechargés depuis {path.name}")


def train_with_aidge():
    aidge_core.Log.set_console_level(aidge_core.Level.Notice)
    random.seed(0)
    np.random.seed(0)

    train_samples = load_dataset_list(TRAIN_IMG_DIR, TRAIN_LBL_DIR)
    val_samples = load_dataset_list(VAL_IMG_DIR, VAL_LBL_DIR)
    print(f"{len(train_samples)} train | {len(val_samples)} val | stride={STRIDE} grille={GRID_H}x{GRID_W}")

    model = create_person_detector_aidge(batch_size=BATCH_SIZE)
    if RESUME and ONNX_FP32_PATH.is_file():
        load_weights_from_onnx(model, ONNX_FP32_PATH)

    model.set_backend("cpu")
    model.set_datatype(aidge_core.dtype.float32)
    model.forward_dims(dims=[[BATCH_SIZE, 1, IMAGE_HEIGHT, IMAGE_WIDTH]], allow_data_dependency=True)
    scheduler = aidge_core.SequentialScheduler(model)
    output_node = model.get_ordered_outputs()[0][0]

    optimizer = aidge_learning.Adam()
    optimizer.set_parameters(list(aidge_core.producers(model, False)))

    best_ap = -1.0
    history = []
    n_train = len(train_samples) // BATCH_SIZE
    n_val = len(val_samples) // BATCH_SIZE

    for epoch in range(EPOCHS):
        lr = MIN_LR + 0.5 * (LEARNING_RATE - MIN_LR) * (1 + math.cos(math.pi * epoch / EPOCHS))
        if epoch < 2:
            lr *= 0.3 * (epoch + 1)  # petit warm-up
        optimizer.set_learning_rate_scheduler(aidge_learning.constant_lr(lr))

        random.shuffle(train_samples)
        tr_sum = 0.0
        for b_idx in range(n_train):
            batch = train_samples[b_idx * BATCH_SIZE:(b_idx + 1) * BATCH_SIZE]
            imgs_np, tgt_np = build_batch_arrays(batch, is_train=True)
            optimizer.reset_grad()
            scheduler.forward(True, [aidge_core.Tensor(imgs_np)])
            out_tensor = output_node.get_operator().get_output(0)
            loss, _, _, grad_np = compute_loss_and_grad(np.asarray(out_tensor), tgt_np)
            tr_sum += loss
            out_tensor.grad = aidge_core.Tensor(grad_np)
            scheduler.backward()
            optimizer.update()

        v_loss = v_cls = v_box = 0.0
        dets_all, gts_all = [], []
        for b_idx in range(n_val):
            batch = val_samples[b_idx * BATCH_SIZE:(b_idx + 1) * BATCH_SIZE]
            imgs_np, tgt_np = build_batch_arrays(batch, is_train=False)
            scheduler.forward(True, [aidge_core.Tensor(imgs_np)])
            pred_np = np.asarray(output_node.get_operator().get_output(0))
            l, c, bx, _ = compute_loss_and_grad(pred_np, tgt_np)
            v_loss += l; v_cls += c; v_box += bx
            dets_all += decode_predictions(pred_np, confidence_threshold=0.05, max_detections=30)
            gts_all += [[[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in s[1]] for s in batch]

        m = evaluate_detections(dets_all, gts_all, iou_thr=0.5, pr_score_thr=0.35)
        v_loss /= max(1, n_val); v_cls /= max(1, n_val); v_box /= max(1, n_val)
        history.append({"epoch": epoch + 1, "train_loss": tr_sum / max(1, n_train), "val_loss": v_loss,
                        "ap50": m["ap"], "precision": m["precision"], "recall": m["recall"]})
        print(f"Époque {epoch + 1:02d}/{EPOCHS} lr={lr:.5f} | train={tr_sum / max(1, n_train):.3f} | "
              f"val={v_loss:.3f} (cls={v_cls:.3f}, box={v_box:.3f}) | "
              f"AP50={m['ap']:.3f} P@0.35={m['precision']:.2f} R@0.35={m['recall']:.2f}")

        if m["ap"] > best_ap:
            best_ap = m["ap"]
            aidge_onnx.export_onnx(model, str(ONNX_FP32_PATH))
            print(f"  -> Meilleur AP50, modèle sauvegardé ({ONNX_FP32_PATH.name})")

    with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)


if __name__ == "__main__":
    train_with_aidge()
