from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STRIDE = 4
IMAGE_HEIGHT = 120
IMAGE_WIDTH = 160


def sigmoid(x):
    x_clip = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def nms(boxes, scores, iou_threshold=0.30, iomin_threshold=0.55):
    """
    NMS vectorisé combinant IoU classique et IoMin (pour supprimer les petites boîtes
    incluses à l'intérieur d'une grande boîte sur la même personne).
    boxes : array [N, 4] au format (x1, y1, x2, y2)
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(1.0, (x2 - x1) * (y2 - y1))
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(1e-6, union)
        min_area = np.minimum(areas[i], areas[order[1:]])
        iomin = inter / np.maximum(1e-6, min_area)

        inds = np.where((iou <= iou_threshold) & (iomin <= iomin_threshold))[0]
        order = order[inds + 1]

    return keep


def decode_predictions(
    raw_output,
    confidence_threshold=0.43,
    nms_iou_threshold=0.30,
    max_detections=12,
    stride=STRIDE,
):
    """
    Décode la sortie brute AIDGE [B, 5, 30, 40] en liste de boîtes [x, y, w, h, score].
    """
    batch_size, _, gh, gw = raw_output.shape
    batch_detections = []

    for b in range(batch_size):
        logits = raw_output[b, 4, :, :]
        probs = sigmoid(logits)

        ys, xs = np.where(probs >= confidence_threshold)
        if len(xs) == 0:
            batch_detections.append([])
            continue

        boxes_xyxy = []
        scores = []

        for gy, gx in zip(ys, xs):
            score = float(probs[gy, gx])
            l = max(0.0, float(raw_output[b, 0, gy, gx])) * stride
            t = max(0.0, float(raw_output[b, 1, gy, gx])) * stride
            r = max(0.0, float(raw_output[b, 2, gy, gx])) * stride
            btm = max(0.0, float(raw_output[b, 3, gy, gx])) * stride

            cx = (gx + 0.5) * stride
            cy = (gy + 0.5) * stride

            x1 = float(np.clip(cx - l, 0.0, IMAGE_WIDTH))
            y1 = float(np.clip(cy - t, 0.0, IMAGE_HEIGHT))
            x2 = float(np.clip(cx + r, 0.0, IMAGE_WIDTH))
            y2 = float(np.clip(cy + btm, 0.0, IMAGE_HEIGHT))

            bw = x2 - x1
            bh = y2 - y1

            if bw >= 6.0 and bh >= 10.0:
                boxes_xyxy.append([x1, y1, x2, y2])
                scores.append(score)

        if len(boxes_xyxy) == 0:
            batch_detections.append([])
            continue

        boxes_xyxy = np.array(boxes_xyxy, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)

        keep = nms(boxes_xyxy, scores, iou_threshold=nms_iou_threshold)[:max_detections]

        img_dets = []
        for idx in keep:
            x1, y1, x2, y2 = boxes_xyxy[idx]
            img_dets.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1), float(scores[idx])])

        batch_detections.append(img_dets)

    return batch_detections


def plot_batch(
    images,
    predictions,
    cols=4,
    figsize=(14, 10),
    title="Prédictions AIDGE",
    save_name="predictions.png",
):
    n = len(images)
    rows = int(np.ceil(n / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for i in range(len(axes)):
        axes[i].axis("off")
        if i >= n:
            continue
        axes[i].imshow(images[i], cmap="gray", vmin=0, vmax=255)
        for det in predictions[i]:
            x, y, w, h, score = det
            rect = patches.Rectangle((x, y), w, h, linewidth=1.8, edgecolor="red", fill=False)
            axes[i].add_patch(rect)
            axes[i].text(
                x,
                max(0, y - 2),
                f"{score:.2f}",
                fontsize=8,
                color="red",
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1},
            )

    fig.suptitle(title)
    plt.tight_layout()
    out_path = RESULTS_DIR / save_name
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Image sauvegardée : {out_path}")
