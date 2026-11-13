import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import PersonDetectorCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "data" / "processed" / "images" / "val"
LABEL_DIR = PROJECT_ROOT / "data" / "processed" / "labels" / "val"
MODEL_PATH = PROJECT_ROOT / "models" / "person_detector_fp32.pth"
RESULTS_DIR = PROJECT_ROOT / "results"

IMAGE_HEIGHT = 120
IMAGE_WIDTH = 160
BATCH_SIZE = 32


class PersonDataset(Dataset):
    """Dataset binaire : 1 si le fichier YOLO contient au moins une bbox, sinon 0."""

    def __init__(self, image_dir, label_dir):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Dossier images absent : {self.image_dir}")
        if not self.label_dir.is_dir():
            raise FileNotFoundError(f"Dossier labels absent : {self.label_dir}")

        self.image_names = sorted(
            p.name for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".png"
        )
        if not self.image_names:
            raise RuntimeError(f"Aucune image PNG dans {self.image_dir}")

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = self.image_dir / image_name
        label_path = self.label_dir / f"{Path(image_name).stem}.txt"

        with Image.open(image_path) as source:
            image = self.transform(source.convert("L"))

        # Un label non vide signifie qu'au moins une personne est annotée.
        target = 1 if label_path.is_file() and label_path.stat().st_size > 0 else 0
        return image, torch.tensor(target, dtype=torch.int64), image_name


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def compute_metrics(tp, tn, fp, fn, loss):
    total = tp + tn + fp + fn
    accuracy = safe_divide(tp + tn, total)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    balanced_accuracy = (recall + specificity) / 2

    return {
        "samples": total,
        "threshold": None,  # renseigné par l'appelant
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "balanced_accuracy": balanced_accuracy,
        "confusion_matrix": {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
        },
    }


def evaluate(model_path, image_dir, label_dir, threshold):
    device = torch.device("cpu")
    dataset = PersonDataset(image_dir, label_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = PersonDetectorCNN().to(device)
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_samples = 0
    tp = tn = fp = fn = 0
    rows = []

    with torch.inference_mode():
        for images, targets, names in loader:
            images = images.to(device)
            targets_float = targets.to(device).float().view(-1, 1)

            logits = model(images)
            if logits.ndim != 2 or logits.shape[1] != 1:
                raise RuntimeError(
                    f"Sortie modèle inattendue : {tuple(logits.shape)} ; attendu [batch, 1]"
                )

            batch_loss = criterion(logits, targets_float)
            total_loss += batch_loss.item()
            total_samples += targets.numel()

            probabilities = torch.sigmoid(logits).view(-1).cpu()
            predictions = probabilities >= threshold

            for name, target, probability, prediction in zip(
                names, targets.tolist(), probabilities.tolist(), predictions.tolist()
            ):
                pred = int(prediction)
                rows.append({
                    "image": name,
                    "label": int(target),
                    "probability_person": f"{probability:.8f}",
                    "prediction": pred,
                    "correct": int(pred == int(target)),
                })

                if pred == 1 and target == 1:
                    tp += 1
                elif pred == 0 and target == 0:
                    tn += 1
                elif pred == 1 and target == 0:
                    fp += 1
                else:
                    fn += 1

    average_loss = total_loss / total_samples
    metrics = compute_metrics(tp, tn, fp, fn, average_loss)
    metrics["threshold"] = threshold
    metrics["model"] = str(Path(model_path))
    metrics["dataset_images"] = str(Path(image_dir))
    metrics["dataset_labels"] = str(Path(label_dir))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / "evaluation_metrics.json"
    predictions_path = RESULTS_DIR / "evaluation_predictions.csv"
    confusion_path = RESULTS_DIR / "confusion_matrix.csv"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False)

    with predictions_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["image", "label", "probability_person", "prediction", "correct"]
        )
        writer.writeheader()
        writer.writerows(rows)

    with confusion_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["", "Prédit sans personne", "Prédit personne"])
        writer.writerow(["Réel sans personne", tn, fp])
        writer.writerow(["Réel personne", fn, tp])

    print("\n===== ÉVALUATION DU MODÈLE FP32 =====")
    print(f"Images évaluées      : {metrics['samples']}")
    print(f"Seuil                : {threshold:.2f}")
    print(f"Loss                 : {average_loss:.4f}")
    print(f"Accuracy             : {metrics['accuracy']:.4f}")
    print(f"Precision            : {metrics['precision']:.4f}")
    print(f"Recall               : {metrics['recall']:.4f}")
    print(f"Specificity          : {metrics['specificity']:.4f}")
    print(f"F1-score             : {metrics['f1_score']:.4f}")
    print(f"Balanced accuracy    : {metrics['balanced_accuracy']:.4f}")
    print("\nMatrice de confusion :")
    print("                         Prédit 0    Prédit 1")
    print(f"Réel 0 (sans personne)  {tn:8d}    {fp:8d}")
    print(f"Réel 1 (personne)       {fn:8d}    {tp:8d}")
    print("\nFichiers générés :")
    print(f"- {metrics_path}")
    print(f"- {predictions_path}")
    print(f"- {confusion_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Évalue le classifieur présence/absence de personne sur le split validation."
    )
    parser.add_argument("--model", default=str(MODEL_PATH), help="Chemin des poids FP32")
    parser.add_argument("--images", default=str(IMAGE_DIR), help="Dossier images de validation")
    parser.add_argument("--labels", default=str(LABEL_DIR), help="Dossier labels de validation")
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Seuil de classification entre 0 et 1 (défaut : 0.5)"
    )
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold doit être compris entre 0 et 1.")

    evaluate(args.model, args.images, args.labels, args.threshold)
