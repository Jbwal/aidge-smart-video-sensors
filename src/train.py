import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from model import PersonDetectorCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_IMAGE_DIR = PROJECT_ROOT / "data" / "processed" / "images" / "train"
TRAIN_LABEL_DIR = PROJECT_ROOT / "data" / "processed" / "labels" / "train"
VAL_IMAGE_DIR = PROJECT_ROOT / "data" / "processed" / "images" / "val"
VAL_LABEL_DIR = PROJECT_ROOT / "data" / "processed" / "labels" / "val"

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
BEST_MODEL_PATH = MODEL_DIR / "person_detector_fp32.pth"
HISTORY_PATH = RESULTS_DIR / "training_history.csv"

IMAGE_HEIGHT = 120
IMAGE_WIDTH = 160
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001
RANDOM_SEED = 42


class PersonDataset(Dataset):
    """Dataset de classification binaire : personne (1) ou absence (0)."""

    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform

        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"Dossier d'images introuvable : {self.img_dir}")
        if not self.label_dir.is_dir():
            raise FileNotFoundError(f"Dossier de labels introuvable : {self.label_dir}")

        self.images = sorted(
            path.name for path in self.img_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        )

        if not self.images:
            raise RuntimeError(f"Aucune image PNG trouvée dans {self.img_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = self.img_dir / img_name
        label_path = self.label_dir / f"{Path(img_name).stem}.txt"

        # Même format d'entrée pour train, validation et prédiction.
        with Image.open(img_path) as source:
            image = source.convert("L")

        has_person = (
            label_path.is_file() and label_path.stat().st_size > 0
        )
        label = torch.tensor([1.0 if has_person else 0.0], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


def calculate_metrics(tp, tn, fp, fn):
    """Calcule accuracy, precision, recall et F1 avec gestion des divisions par zéro."""
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )
    return accuracy, precision, recall, f1


def run_validation(model, loader, criterion, device):
    """Évalue le modèle sans calcul de gradients."""
    model.eval()

    total_loss = 0.0
    total_samples = 0
    tp = tn = fp = fn = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Avec BCEWithLogitsLoss, logit >= 0 équivaut à sigmoid(logit) >= 0.5.
            predictions = (logits >= 0.0)
            targets = (labels >= 0.5)

            tp += (predictions & targets).sum().item()
            tn += ((~predictions) & (~targets)).sum().item()
            fp += (predictions & (~targets)).sum().item()
            fn += ((~predictions) & targets).sum().item()

    val_loss = total_loss / total_samples
    accuracy, precision, recall, f1 = calculate_metrics(tp, tn, fp, fn)

    metrics = {
        "val_loss": val_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    return metrics


def train():
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Matériel utilisé : {device}")

    # Train et validation utilisent exactement le même prétraitement.
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
    ])

    train_dataset = PersonDataset(
        TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR, transform=transform
    )
    val_dataset = PersonDataset(
        VAL_IMAGE_DIR, VAL_LABEL_DIR, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    print(f"Images entraînement : {len(train_dataset)}")
    print(f"Images validation   : {len(val_dataset)}")

    model = PersonDetectorCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history = []

    print(f"\nDébut de l'entraînement : {EPOCHS} époques\n")

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            train_loss_sum += loss.item() * batch_size
            train_samples += batch_size

        train_loss = train_loss_sum / train_samples

        # Validation à la fin de chaque époque.
        val_metrics = run_validation(model, val_loader, criterion, device)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **val_metrics,
        }
        history.append(row)

        print(
            f"Époque {epoch + 1:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"accuracy={val_metrics['accuracy']:.3f} | "
            f"precision={val_metrics['precision']:.3f} | "
            f"recall={val_metrics['recall']:.3f} | "
            f"F1={val_metrics['f1']:.3f}"
        )

        # On conserve les poids ayant la meilleure perte de validation.
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> meilleur modèle sauvegardé : {BEST_MODEL_PATH}")

    # Historique CSV pour tracer les courbes et comparer les expériences.
    fieldnames = [
        "epoch", "train_loss", "val_loss", "accuracy",
        "precision", "recall", "f1", "tp", "tn", "fp", "fn"
    ]
    with HISTORY_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    print("\nEntraînement terminé.")
    print(f"Meilleur modèle : {BEST_MODEL_PATH}")
    print(f"Historique      : {HISTORY_PATH}")
    print(f"Meilleure val_loss : {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
