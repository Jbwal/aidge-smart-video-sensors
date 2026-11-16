import csv
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance
from torchvision import transforms

from model import PersonDetector
from decode import decode_predictions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_IMAGE_DIR = PROJECT_ROOT / "data" / "processed" / "images" / "train"
TRAIN_LABEL_DIR = PROJECT_ROOT / "data" / "processed" / "labels" / "train"
VAL_IMAGE_DIR = PROJECT_ROOT / "data" / "processed" / "images" / "val"
VAL_LABEL_DIR = PROJECT_ROOT / "data" / "processed" / "labels" / "val"

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
BEST_MODEL_PATH = MODEL_DIR / "person_detector_fp32.pth"
ONNX_FP32_PATH = MODEL_DIR / "person_detector_fp32.onnx"
HISTORY_PATH = RESULTS_DIR / "training_history.csv"

IMAGE_HEIGHT = 120
IMAGE_WIDTH = 160
STRIDE = PersonDetector.stride  # 4
GRID_H = IMAGE_HEIGHT // STRIDE  # 30
GRID_W = IMAGE_WIDTH // STRIDE   # 40

BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 0.0015
RANDOM_SEED = 42


class PersonFCOSDataset(Dataset):
    """Dataset avec Center-Sampling 3x3, filtrage des micro-boîtes et Data Augmentation."""

    def __init__(self, img_dir, label_dir, is_train=True):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.is_train = is_train
        self.to_tensor = transforms.ToTensor()
        self.images = sorted(
            p.name for p in self.img_dir.iterdir() if p.suffix.lower() == ".png"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = self.img_dir / img_name
        label_path = self.label_dir / f"{Path(img_name).stem}.txt"

        with Image.open(img_path) as source:
            image = source.convert("L").resize((IMAGE_WIDTH, IMAGE_HEIGHT))

        boxes = []
        if label_path.is_file() and label_path.stat().st_size > 0:
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, xc, yc, w, h = map(float, parts)
                        # Filtrer les personnes trop minuscules (< 8px de large ou < 14px de haut)
                        if (w * IMAGE_WIDTH) >= 8.0 and (h * IMAGE_HEIGHT) >= 14.0:
                            boxes.append([xc, yc, w, h])

        # Data Augmentation en entraînement uniquement
        if self.is_train:
            # 1. Flip horizontal aléatoire
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                for b in boxes:
                    b[0] = 1.0 - b[0]  # Inversion de xc

            # 2. Variation aléatoire de luminosité et contraste (simule l'éclairage réel caméra)
            if random.random() < 0.5:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))

        tensor_img = self.to_tensor(image)

        # Cible [0:left, 1:top, 2:right, 3:bottom, 4:conf]
        target = torch.zeros((5, GRID_H, GRID_W), dtype=torch.float32)

        # Trier les boîtes des plus grandes aux plus petites (les plus petites prennent la priorité au centre)
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)

        for xc, yc, w, h in boxes:
            x1 = (xc - w / 2.0) * IMAGE_WIDTH
            y1 = (yc - h / 2.0) * IMAGE_HEIGHT
            x2 = (xc + w / 2.0) * IMAGE_WIDTH
            y2 = (yc + h / 2.0) * IMAGE_HEIGHT

            gx_center = min(max(int((xc * IMAGE_WIDTH) // STRIDE), 0), GRID_W - 1)
            gy_center = min(max(int((yc * IMAGE_HEIGHT) // STRIDE), 0), GRID_H - 1)

            # Center Sampling : on active un voisinage 3x3 autour du centre (si dans la boîte)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    gx = gx_center + dx
                    gy = gy_center + dy
                    if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                        cx = (gx + 0.5) * STRIDE
                        cy = (gy + 0.5) * STRIDE
                        if x1 < cx < x2 and y1 < cy < y2:
                            target[0, gy, gx] = (cx - x1) / STRIDE
                            target[1, gy, gx] = (cy - y1) / STRIDE
                            target[2, gy, gx] = (x2 - cx) / STRIDE
                            target[3, gy, gx] = (y2 - cy) / STRIDE
                            target[4, gy, gx] = 1.0

        return tensor_img, target


def iou_loss(pred_ltrb, true_ltrb):
    """Perte IoU normalisée entre boîtes prédites et réelles (indépendante de la taille)."""
    pred_l, pred_t, pred_r, pred_b = pred_ltrb[:, 0], pred_ltrb[:, 1], pred_ltrb[:, 2], pred_ltrb[:, 3]
    true_l, true_t, true_r, true_b = true_ltrb[:, 0], true_ltrb[:, 1], true_ltrb[:, 2], true_ltrb[:, 3]

    pred_area = (pred_l + pred_r) * (pred_t + pred_b)
    true_area = (true_l + true_r) * (true_t + true_b)

    inter_w = torch.min(pred_l, true_l) + torch.min(pred_r, true_r)
    inter_h = torch.min(pred_t, true_t) + torch.min(pred_b, true_b)
    inter_area = torch.clamp(inter_w, min=0.0) * torch.clamp(inter_h, min=0.0)

    union_area = pred_area + true_area - inter_area + 1e-6
    iou = inter_area / union_area
    return (1.0 - iou).mean()


def fcos_loss(predictions, targets):
    # Poids positif équilibré (5.0 au lieu de 15.0 car on a maintenant 9 cellules par personne)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0], device=predictions.device))

    conf_loss = bce(predictions[:, 4, :, :], targets[:, 4, :, :])

    mask = targets[:, 4, :, :] == 1.0
    if mask.sum() > 0:
        pred_ltrb = F.relu(predictions[:, 0:4, :, :]).permute(0, 2, 3, 1)[mask]
        true_ltrb = targets[:, 0:4, :, :].permute(0, 2, 3, 1)[mask]
        box_loss = iou_loss(pred_ltrb, true_ltrb)
    else:
        box_loss = 0.0

    return conf_loss + 1.5 * box_loss


def train():
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Matériel utilisé : {device}")

    train_loader = DataLoader(
        PersonFCOSDataset(TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR, is_train=True),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        PersonFCOSDataset(VAL_IMAGE_DIR, VAL_LABEL_DIR, is_train=False),
        batch_size=BATCH_SIZE, shuffle=False
    )

    model = PersonDetector().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = fcos_loss(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        total_detected = 0
        with torch.inference_mode():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                preds = model(images)
                val_loss += fcos_loss(preds, targets).item() * images.size(0)
                decoded = decode_predictions(preds.cpu().numpy(), confidence_threshold=0.45, nms_iou_threshold=0.35)
                total_detected += sum(len(dets) for dets in decoded)

        val_loss /= len(val_loader.dataset)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "detected": total_detected})

        print(f"Époque {epoch+1:02d}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | détections_val={total_detected}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    with HISTORY_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "detected"])
        writer.writeheader()
        writer.writerows(history)

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location="cpu"))
    model.cpu().eval()
    dummy_input = torch.randn(1, 1, IMAGE_HEIGHT, IMAGE_WIDTH)
    torch.onnx.export(
        model, dummy_input, ONNX_FP32_PATH,
        export_params=True, opset_version=13,
        input_names=["input"], output_names=["output"]
    )
    print(f"Modèle ONNX exporté : {ONNX_FP32_PATH}")


if __name__ == "__main__":
    train()