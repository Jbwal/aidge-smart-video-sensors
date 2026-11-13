import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import PersonDetectorCNN


# Le modèle a été entraîné avec des images 1 canal, 120 x 160 pixels.
IMAGE_HEIGHT = 120
IMAGE_WIDTH = 160
DEFAULT_THRESHOLD = 0.5

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "person_detector_fp32.pth"


def build_transform():
    """Reproduit le prétraitement utilisé pendant l'entraînement."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),  # valeurs des pixels dans [0, 1]
    ])


def predict(image_path, model_path=DEFAULT_MODEL_PATH, threshold=DEFAULT_THRESHOLD):
    image_path = Path(image_path)
    model_path = Path(model_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image introuvable : {image_path}")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Poids du modèle introuvables : {model_path}\n"
            "Lance d'abord l'entraînement avec : python src/train.py"
        )

    device = torch.device("cpu")

    model = PersonDetectorCNN().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with Image.open(image_path) as image:
        input_tensor = build_transform()(image).unsqueeze(0).to(device)

    # Vérification explicite du format transmis au CNN.
    expected_shape = (1, 1, IMAGE_HEIGHT, IMAGE_WIDTH)
    if tuple(input_tensor.shape) != expected_shape:
        raise RuntimeError(
            f"Format inattendu : {tuple(input_tensor.shape)}, "
            f"format attendu : {expected_shape}"
        )

    with torch.inference_mode():
        logit = model(input_tensor)
        probability = torch.sigmoid(logit).item()

    prediction = probability >= threshold

    print(f"Image : {image_path.name}")
    print(f"Entrée modèle : {tuple(input_tensor.shape)}")
    print(f"Score sigmoid : {probability:.4f} ({probability * 100:.2f} %)")
    print(f"Seuil : {threshold:.2f}")
    print("PERSONNE DÉTECTÉE" if prediction else "PAS DE PERSONNE")

    return probability, prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classification présence/absence d'une personne."
    )
    parser.add_argument("image", help="Chemin de l'image à analyser")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Chemin du fichier de poids .pth",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Seuil de décision entre 0 et 1 (défaut : 0.5)",
    )
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold doit être compris entre 0 et 1.")

    predict(args.image, args.model, args.threshold)
