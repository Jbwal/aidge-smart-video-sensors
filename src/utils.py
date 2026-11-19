from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches

from config import STRIDE, IMAGE_HEIGHT, IMAGE_WIDTH

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def sigmoid(x):
    x_clip = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def nms(boxes, scores, iou_threshold=0.50, iomin_threshold=0.85):
    """NMS classique (IoU) + suppression des boîtes quasi totalement incluses (IoMin)."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(1.0, (x2 - x1) * (y2 - y1))
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest]); yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest]); yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / np.maximum(1e-6, areas[i] + areas[rest] - inter)
        iomin = inter / np.maximum(1e-6, np.minimum(areas[i], areas[rest]))
        order = rest[(iou <= iou_threshold) & (iomin <= iomin_threshold)]
    return keep


def local_max_mask(p):
    """Pics locaux 3x3 d'une carte [H,W] (équivalent d'un MaxPool 3x3 stride 1)."""
    pp = np.pad(p, 1, constant_values=-1.0)
    h, w = p.shape
    neigh = np.full_like(p, -1.0)
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            neigh = np.maximum(neigh, pp[dy:dy + h, dx:dx + w])
    return p >= neigh


def decode_predictions(raw_output, confidence_threshold=0.35, nms_iou_threshold=0.50,
                       max_detections=12, stride=STRIDE):
    """[B,5,gh,gw] -> liste (par image) de [x, y, w, h, score]."""
    batch_size, _, gh, gw = raw_output.shape
    gxs, gys = np.meshgrid(np.arange(gw), np.arange(gh))
    cx = (gxs + 0.5) * stride
    cy = (gys + 0.5) * stride
    out = []
    for b in range(batch_size):
        probs = sigmoid(raw_output[b, 4])
        cand = (probs >= confidence_threshold) & local_max_mask(probs)
        ys, xs = np.where(cand)
        if len(xs) == 0:
            out.append([])
            continue
        l = np.maximum(0.0, raw_output[b, 0][ys, xs]) * stride
        t = np.maximum(0.0, raw_output[b, 1][ys, xs]) * stride
        r = np.maximum(0.0, raw_output[b, 2][ys, xs]) * stride
        bt = np.maximum(0.0, raw_output[b, 3][ys, xs]) * stride
        x1 = np.clip(cx[ys, xs] - l, 0, IMAGE_WIDTH); x2 = np.clip(cx[ys, xs] + r, 0, IMAGE_WIDTH)
        y1 = np.clip(cy[ys, xs] - t, 0, IMAGE_HEIGHT); y2 = np.clip(cy[ys, xs] + bt, 0, IMAGE_HEIGHT)
        scores = probs[ys, xs].astype(np.float32)
        valid = ((x2 - x1) >= 6.0) & ((y2 - y1) >= 10.0)
        if not valid.any():
            out.append([])
            continue
        boxes = np.stack([x1, y1, x2, y2], 1)[valid].astype(np.float32)
        scores = scores[valid]
        keep = nms(boxes, scores, iou_threshold=nms_iou_threshold)[:max_detections]
        out.append([[float(boxes[k, 0]), float(boxes[k, 1]),
                     float(boxes[k, 2] - boxes[k, 0]), float(boxes[k, 3] - boxes[k, 1]),
                     float(scores[k])] for k in keep])
    return out


def _iou_xywh(a, g):
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    gx2, gy2 = g[0] + g[2], g[1] + g[3]
    iw = max(0.0, min(ax2, gx2) - max(a[0], g[0]))
    ih = max(0.0, min(ay2, gy2) - max(a[1], g[1]))
    inter = iw * ih
    return inter / max(1e-6, a[2] * a[3] + g[2] * g[3] - inter)


def evaluate_detections(dets_all, gts_all, iou_thr=0.5, pr_score_thr=0.35):
    """AP@IoU et précision/rappel au seuil pr_score_thr. dets: [x,y,w,h,s], gts: [x,y,w,h]."""
    records = []
    n_gt = sum(len(g) for g in gts_all)
    for dets, gts in zip(dets_all, gts_all):
        used = [False] * len(gts)
        for d in sorted(dets, key=lambda d: -d[4]):
            best, bj = 0.0, -1
            for j, g in enumerate(gts):
                if used[j]:
                    continue
                iou = _iou_xywh(d, g)
                if iou > best:
                    best, bj = iou, j
            if bj >= 0 and best >= iou_thr:
                used[bj] = True
                records.append((d[4], 1))
            else:
                records.append((d[4], 0))
    if not records or n_gt == 0:
        return {"ap": 0.0, "precision": 0.0, "recall": 0.0}
    records.sort(key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in records]).astype(np.float64)
    fp = np.cumsum([1 - r[1] for r in records]).astype(np.float64)
    rec = tp / n_gt
    prec = tp / np.maximum(1.0, tp + fp)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    sel = [r for r in records if r[0] >= pr_score_thr]
    tpn = sum(r[1] for r in sel)
    return {"ap": ap, "precision": tpn / max(1, len(sel)), "recall": tpn / n_gt}


def plot_batch(images, predictions, cols=4, figsize=(14, 10), title="Prédictions AIDGE",
               save_name="predictions.png"):
    n = len(images)
    rows = int(np.ceil(n / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for i in range(len(axes)):
        axes[i].axis("off")
        if i >= n:
            continue
        axes[i].imshow(images[i], cmap="gray", vmin=0, vmax=255)
        for x, y, w, h, score in predictions[i]:
            axes[i].add_patch(patches.Rectangle((x, y), w, h, linewidth=1.8, edgecolor="red", fill=False))
            axes[i].text(x, max(0, y - 2), f"{score:.2f}", fontsize=8, color="red",
                         bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1})
    fig.suptitle(title)
    plt.tight_layout()
    out_path = RESULTS_DIR / save_name
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Image sauvegardée : {out_path}")
