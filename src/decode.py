import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def nms(boxes, scores, iou_threshold=0.3):
    if len(boxes) == 0:
        return []

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)

        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection

        iou = intersection / np.maximum(union, 1e-8)

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return keep


def decode_predictions(
    raw_output,
    confidence_threshold=0.3,
    nms_iou_threshold=0.5,
    stride=4,
    image_height=120,
    image_width=160,
    max_detections=100,
):
    raw_output = np.asarray(raw_output, dtype=np.float32)

    if raw_output.ndim == 3:
        raw_output = raw_output[None, ...]

    if raw_output.ndim != 4 or raw_output.shape[1] != 5:
        raise ValueError(
            f"Expected shape [5, H, W] or [B, 5, H, W], got {raw_output.shape}"
        )

    batch_size, _, grid_height, grid_width = raw_output.shape

    grid_y, grid_x = np.meshgrid(
        np.arange(grid_height, dtype=np.float32),
        np.arange(grid_width, dtype=np.float32),
        indexing="ij",
    )

    center_x = (grid_x + 0.5) * float(stride)
    center_y = (grid_y + 0.5) * float(stride)

    results = []

    for batch_idx in range(batch_size):
        prediction = raw_output[batch_idx]

        raw_ltrb = prediction[:4]
        objectness_logits = prediction[4]

        distances = np.maximum(raw_ltrb, 0.0) * float(stride)
        scores = sigmoid(objectness_logits)

        left = distances[0]
        top = distances[1]
        right = distances[2]
        bottom = distances[3]

        x1 = np.clip(center_x - left, 0.0, float(image_width))
        y1 = np.clip(center_y - top, 0.0, float(image_height))
        x2 = np.clip(center_x + right, 0.0, float(image_width))
        y2 = np.clip(center_y + bottom, 0.0, float(image_height))

        boxes_xyxy = np.stack(
            (x1, y1, x2, y2),
            axis=-1,
        ).reshape(-1, 4)

        scores = scores.reshape(-1)

        valid = (
            (scores >= confidence_threshold)
            & (boxes_xyxy[:, 2] > boxes_xyxy[:, 0])
            & (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
        )

        boxes_xyxy = boxes_xyxy[valid]
        scores = scores[valid]

        if len(boxes_xyxy) == 0:
            results.append([])
            continue

        keep = nms(
            boxes_xyxy,
            scores,
            nms_iou_threshold,
        )

        if max_detections is not None:
            keep = keep[:max_detections]

        boxes_xyxy = boxes_xyxy[keep].astype(np.float32)
        scores = scores[keep].astype(np.float32)

        image_results = []

        for box, score in zip(boxes_xyxy, scores):
            x1, y1, x2, y2 = box

            image_results.append([
                float(x1),
                float(y1),
                float(x2 - x1),
                float(y2 - y1),
                float(score),
            ])

        results.append(image_results)

    return results


def plot_batch(images, predictions, cols=8, figsize=(16, 4), title=None):
    batch_size = len(images)
    rows = (batch_size + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for i in range(batch_size):
        image = images[i]

        if hasattr(image, "detach"):
            image = image.detach().cpu().numpy()

        image = np.squeeze(image)

        axes[i].imshow(image, cmap="gray")

        for prediction in predictions[i]:
            x, y, w, h = prediction[:4]

            rect = patches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=2,
                edgecolor="red",
                fill=False,
            )
            axes[i].add_patch(rect)

            if len(prediction) >= 5:
                score = prediction[4]

                axes[i].text(
                    x,
                    max(0, y - 2),
                    f"{score:.2f}",
                    fontsize=8,
                    color="red",
                    bbox=dict(
                        facecolor="white",
                        alpha=0.7,
                        edgecolor="none",
                        pad=1,
                    ),
                )

        axes[i].axis("off")

    for i in range(batch_size, len(axes)):
        axes[i].axis("off")

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    
    plt.tight_layout()
    plt.savefig("results/prediction_test.png", dpi=150)
    print("Image sauvegardée dans : results/prediction_test.png")