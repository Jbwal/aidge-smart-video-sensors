import argparse
from pathlib import Path
import cv2
import numpy as np
import aidge_core
import aidge_backend_cpu
import aidge_onnx

from utils import decode_predictions, plot_batch, sigmoid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONNX_FP32_PATH = PROJECT_ROOT / "models" / "person_detector_aidge_fp32.onnx"


def predict_single_image(img_path, threshold=0.35):
    aidge_core.Log.set_console_level(aidge_core.Level.Notice)

    # 1. Chargement de l'image en niveaux de gris QQVGA (160x120)
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Impossible de lire l'image : {img_path}")
    img_resized = cv2.resize(img, (160, 120))

    # Normalisation [-1, 1] utilisée pendant l'entraînement AIDGE
    img_norm = img_resized.astype(np.float32) / 127.5 - 1.0
    input_tensor = aidge_core.Tensor(img_norm[None, None, :, :])  # [1, 1, 120, 160]

    # 2. Chargement du modèle ONNX dans AIDGE et inférence CPU
    model = aidge_onnx.load_onnx(str(ONNX_FP32_PATH))
    aidge_core.remove_flatten(model)
    aidge_core.expand_metaops(model)
    model.forward_dims(dims=[[1, 1, 120, 160]], allow_data_dependency=True)
    model.compile("cpu", aidge_core.dtype.float32, dims=[[1, 1, 120, 160]])

    scheduler = aidge_core.SequentialScheduler(model)
    scheduler.forward(True, [input_tensor])

    out_node = model.get_ordered_outputs()[0][0]
    raw_out = np.asarray(out_node.get_operator().get_output(0))

    # Affichage du score de confiance maximal trouvé dans toute la grille 30x40
    max_conf = float(np.max(sigmoid(raw_out[0, 4, :, :])))
    print(f"Score de confiance maximal sur l'image : {max_conf:.4f}")

    # 3. Décodage des boîtes et sauvegarde dans results/single_prediction.png
    detections = decode_predictions(raw_out, confidence_threshold=threshold)[0]
    print(f"{len(detections)} personne(s) détectée(s) (seuil={threshold}) :")
    for det in detections:
        x, y, w, h, score = det
        print(f"  - Boîte [x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}] | Confiance={score:.2f}")

    plot_batch(
        [img_resized],
        [detections],
        cols=1,
        figsize=(6, 5),
        title=f"Test AIDGE (max conf: {max_conf:.2f})",
        save_name="single_prediction.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Chemin vers l'image .png")
    parser.add_argument("--threshold", type=float, default=0.35, help="Seuil de confiance")
    args = parser.parse_args()

    predict_single_image(args.image_path, args.threshold)