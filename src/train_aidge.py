import csv
import random
from pathlib import Path
import cv2
import numpy as np

import aidge_core
import aidge_backend_cpu
import aidge_learning
import aidge_onnx

from model_aidge import create_person_detector_aidge, STRIDE, IMAGE_HEIGHT, IMAGE_WIDTH
from utils import decode_predictions, sigmoid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_IMG_DIR = PROJECT_ROOT / "data" / "processed" / "images" / "train"
TRAIN_LBL_DIR = PROJECT_ROOT / "data" / "processed" / "labels" / "train"
VAL_IMG_DIR = PROJECT_ROOT / "data" / "processed" / "images" / "val"
VAL_LBL_DIR = PROJECT_ROOT / "data" / "processed" / "labels" / "val"

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ONNX_FP32_PATH = MODEL_DIR / "person_detector_aidge_fp32.onnx"
HISTORY_PATH = RESULTS_DIR / "training_history_aidge.csv"

GRID_H = IMAGE_HEIGHT // STRIDE  # 30
GRID_W = IMAGE_WIDTH // STRIDE   # 40

BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 0.0005


def load_dataset_list(img_dir, lbl_dir, keep_neg_ratio=0.20):
    """
    Charge les images avec personnes et limite la proportion d'images 100% vides
    pour éviter que le réseau ne prédise '0 personne' partout.
    """
    pos_samples = []
    neg_samples = []

    for img_path in sorted(Path(img_dir).glob("*.png")):
        lbl_path = Path(lbl_dir) / f"{img_path.stem}.txt"
        boxes = []
        if lbl_path.is_file() and lbl_path.stat().st_size > 0:
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, xc, yc, w, h = map(float, parts)
                        if (w * IMAGE_WIDTH) >= 8.0 and (h * IMAGE_HEIGHT) >= 14.0:
                            boxes.append([xc, yc, w, h])
        if len(boxes) > 0:
            pos_samples.append((str(img_path), boxes))
        else:
            neg_samples.append((str(img_path), []))

    random.seed(42)
    num_neg_to_keep = int(len(pos_samples) * keep_neg_ratio)
    neg_samples = random.sample(neg_samples, min(len(neg_samples), num_neg_to_keep))

    all_samples = pos_samples + neg_samples
    random.shuffle(all_samples)
    return all_samples


def build_batch_arrays(batch_samples, is_train=True):
    b_size = len(batch_samples)
    images_np = np.zeros((b_size, 1, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
    targets_np = np.zeros((b_size, 5, GRID_H, GRID_W), dtype=np.float32)

    for i, (img_path, orig_boxes) in enumerate(batch_samples):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

        boxes = [b.copy() for b in orig_boxes]

        if is_train:
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
                for b in boxes:
                    b[0] = 1.0 - b[0]
            if random.random() < 0.4:
                alpha = random.uniform(0.85, 1.15)
                img = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

        images_np[i, 0] = img.astype(np.float32) / 127.5 - 1.0

        # Tri des plus grandes boîtes aux plus petites pour que les petites personnes devant ne soient pas écrasées
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)

        for xc, yc, w, h in boxes:
            x1 = (xc - w / 2.0) * IMAGE_WIDTH
            y1 = (yc - h / 2.0) * IMAGE_HEIGHT
            x2 = (xc + w / 2.0) * IMAGE_WIDTH
            y2 = (yc + h / 2.0) * IMAGE_HEIGHT

            gx_c = min(max(int((xc * IMAGE_WIDTH) // STRIDE), 0), GRID_W - 1)
            gy_c = min(max(int((yc * IMAGE_HEIGHT) // STRIDE), 0), GRID_H - 1)

            # Zone centrale étendue (5x5 bornée à l'intérieur de la boîte)
            for dy in range(-2,3):
                for dx in range(-2,3):
                    gx, gy = gx_c + dx, gy_c + dy
                    if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                        cx = (gx + 0.5) * STRIDE
                        cy = (gy + 0.5) * STRIDE
                        # On vérifie que le point est bien dans les 60% centraux de la personne
                        margin_x = (x2 - x1) * 0.20
                        margin_y = (y2 - y1) * 0.20
                        if (
                            (x1 + margin_x <= cx <= x2 - margin_x)
                            and (y1 + margin_y <= cy <= y2 - margin_y)
                        ) or (dx == 0 and dy == 0):
                            targets_np[i, 0, gy, gx] = max(0.2, (cx - x1) / STRIDE)
                            targets_np[i, 1, gy, gx] = max(0.2, (cy - y1) / STRIDE)
                            targets_np[i, 2, gy, gx] = max(0.2, (x2 - cx) / STRIDE)
                            targets_np[i, 3, gy, gx] = max(0.2, (y2 - cy) / STRIDE)
                            targets_np[i, 4, gy, gx] = 1.0

    return images_np, targets_np


def compute_fcos_loss_and_grad(pred_np, target_np, pos_weight=7.5, box_weight=1.0):
    batch_size, _, gh, gw = pred_np.shape
    grad_output = np.zeros_like(pred_np, dtype=np.float32)

    pred_clean = np.nan_to_num(pred_np, nan=0.0, posinf=50.0, neginf=-50.0)

    # 1. Canal 4 : Objectness (Confiance)
    logits = np.clip(pred_clean[:, 4, :, :], -20.0, 20.0)
    obj_target = target_np[:, 4, :, :]
    probs = sigmoid(logits)

    eps = 1e-6
    neg_weight = 1.0 + 3.0 * (probs ** 2)
    weights_map = np.where(obj_target == 1.0, pos_weight, neg_weight)

    bce_map = -(
        pos_weight * obj_target * np.log(probs + eps)
        + (1.0 - obj_target) * neg_weight * np.log(1.0 - probs + eps)
    )
    conf_loss = float(np.mean(bce_map))

    # Échelle de gradient forte : divisé uniquement par batch_size * 100 (et non 19200)
    grad_output[:, 4, :, :] = (probs - obj_target) * weights_map / float(batch_size * 100.0)

    # 2. Canaux 0..3 : Distances LTRB
    mask = obj_target == 1.0
    num_pos = float(np.sum(mask))

    if num_pos > 0:
        mask_4d = np.broadcast_to(mask[:, None, :, :], (batch_size, 4, gh, gw))
        raw_ltrb = np.clip(pred_clean[:, :4, :, :], -50.0, 50.0)
        true_ltrb = target_np[:, :4, :, :]

        diff = (raw_ltrb - true_ltrb) * mask_4d
        abs_diff = np.abs(diff)

        smooth_l1 = np.where(abs_diff < 1.0, 0.5 * (diff ** 2), abs_diff - 0.5)
        box_loss = float(np.sum(smooth_l1 * mask_4d) / (num_pos * 4.0))

        grad_l1 = np.where(abs_diff < 1.0, diff, np.sign(diff)) * mask_4d
        grad_output[:, :4, :, :] = box_weight * grad_l1 / max(4.0, num_pos / 4.0)
    else:
        box_loss = 0.0

    total_loss = conf_loss + box_weight * box_loss
    grad_output = np.clip(grad_output, -0.5, 0.5)
    return total_loss, conf_loss, box_loss, grad_output


def train_with_aidge():
    aidge_core.Log.set_console_level(aidge_core.Level.Notice)

    train_samples = load_dataset_list(TRAIN_IMG_DIR, TRAIN_LBL_DIR, keep_neg_ratio=0.20)
    val_samples = load_dataset_list(VAL_IMG_DIR, VAL_LBL_DIR, keep_neg_ratio=0.20)

    print(f"Entraînement Haute Capacité AIDGE : {len(train_samples)} train | {len(val_samples)} val")

    model = create_person_detector_aidge(batch_size=BATCH_SIZE)

    if ONNX_FP32_PATH.is_file():
        print(f"Transfert propre des poids depuis {ONNX_FP32_PATH.name} vers PaddedConv2D...")
        onnx_m = aidge_onnx.load_onnx(str(ONNX_FP32_PATH))
        onnx_tensors = [
            np.asarray(p.get_operator().get_output(0)).copy()
            for p in aidge_core.producers(onnx_m, False)
        ]
        used = [False] * len(onnx_tensors)

        for prod in aidge_core.producers(model, False):
            t_target = prod.get_operator().get_output(0)
            target_shape = tuple(t_target.dims)
            for idx, arr in enumerate(onnx_tensors):
                if not used[idx] and arr.shape == target_shape:
                    # Si c'est le biais à 5 canaux de det_head, on remonte le logit de confiance (+2.2)
                    if target_shape == (5,):
                        arr[4] = max(arr[4] + 2.2, -1.2)
                    prod.get_operator().set_output(0, aidge_core.Tensor(arr))
                    used[idx] = True
                    break

    model.set_backend("cpu")
    model.set_datatype(aidge_core.dtype.float32)
    model.forward_dims(dims=[[BATCH_SIZE, 1, IMAGE_HEIGHT, IMAGE_WIDTH]], allow_data_dependency=True)
    scheduler = aidge_core.SequentialScheduler(model)
    output_node = model.get_ordered_outputs()[0][0]

    learnable_params = list(aidge_core.producers(model, False))
    optimizer = aidge_learning.Adam()
    optimizer.set_parameters(learnable_params)

    best_val_loss = float("inf")
    history = []

    num_train_batches = len(train_samples) // BATCH_SIZE
    num_val_batches = len(val_samples) // BATCH_SIZE

    for epoch in range(EPOCHS):
        current_lr = LEARNING_RATE
        if epoch >= 15:
            current_lr = LEARNING_RATE * 0.5
        if epoch >= 25:
            current_lr = LEARNING_RATE * 0.2
        optimizer.set_learning_rate_scheduler(aidge_learning.constant_lr(current_lr))

        random.shuffle(train_samples)
        train_loss_sum = 0.0

        for b_idx in range(num_train_batches):
            batch_data = train_samples[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
            imgs_np, targets_np = build_batch_arrays(batch_data, is_train=True)

            input_tensor = aidge_core.Tensor(imgs_np)
            optimizer.reset_grad()

            scheduler.forward(True, [input_tensor])
            out_tensor = output_node.get_operator().get_output(0)
            pred_np = np.asarray(out_tensor)

            loss_val, _, _, grad_np = compute_fcos_loss_and_grad(pred_np, targets_np)
            train_loss_sum += loss_val

            out_tensor.grad = aidge_core.Tensor(grad_np)

            scheduler.backward()
            optimizer.update()

        avg_train_loss = train_loss_sum / max(1, num_train_batches)

        val_loss_sum = 0.0
        val_conf_sum = 0.0
        val_box_sum = 0.0
        val_detections = 0

        for b_idx in range(num_val_batches):
            batch_data = val_samples[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
            imgs_np, targets_np = build_batch_arrays(batch_data, is_train=False)

            input_tensor = aidge_core.Tensor(imgs_np)
            scheduler.forward(True, [input_tensor])
            pred_np = np.asarray(output_node.get_operator().get_output(0))

            v_loss, v_conf, v_box, _ = compute_fcos_loss_and_grad(pred_np, targets_np)
            val_loss_sum += v_loss
            val_conf_sum += v_conf
            val_box_sum += v_box

            decoded = decode_predictions(pred_np, confidence_threshold=0.45)
            val_detections += sum(len(d) for d in decoded)

        avg_val_loss = val_loss_sum / max(1, num_val_batches)
        avg_conf = val_conf_sum / max(1, num_val_batches)
        avg_box = val_box_sum / max(1, num_val_batches)

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_detections": val_detections,
        })

        print(
            f"Époque {epoch + 1:02d}/{EPOCHS} | "
            f"train={avg_train_loss:.3f} | "
            f"val={avg_val_loss:.3f} (conf={avg_conf:.3f}, box={avg_box:.3f}) | "
            f"détections(>=0.45)={val_detections}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            aidge_onnx.export_onnx(model, str(ONNX_FP32_PATH))
            print(f"  -> Modèle sauvegardé ({ONNX_FP32_PATH.name})")

    with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_detections"])
        writer.writeheader()
        writer.writerows(history)


if __name__ == "__main__":
    train_with_aidge()
