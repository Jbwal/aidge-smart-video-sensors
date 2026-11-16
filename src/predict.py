import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms

from model import PersonDetector
from decode import decode_predictions, plot_batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "person_detector_fp32.pth"


def predict(image_path, model_path=DEFAULT_MODEL_PATH, threshold=0.4):
    model = PersonDetector()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((120, 160)),
        transforms.ToTensor(),
    ])

    with Image.open(image_path) as img:
        tensor = transform(img).unsqueeze(0)

    with torch.inference_mode():
        raw_output = model(tensor).numpy()

    detections = decode_predictions(raw_output, confidence_threshold=threshold)
    print(f"{len(detections[0])} personne(s) détectée(s) :")
    for det in detections[0]:
        x, y, w, h, score = det
        print(f"  - Boîte [x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}] | Confiance={score:.2f}")

    plot_batch(tensor, detections, cols=1, figsize=(5, 4), title=Path(image_path).name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Chemin de l'image à tester")
    parser.add_argument("--threshold", type=float, default=0.4)
    args = parser.parse_args()
    predict(args.image, threshold=args.threshold)