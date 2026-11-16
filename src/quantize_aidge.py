from pathlib import Path
import numpy as np
from PIL import Image
import aidge_core
import aidge_backend_cpu
import aidge_onnx
import aidge_quantization

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONNX_FP32_PATH = str(PROJECT_ROOT / "models" / "person_detector_fp32.onnx")
ONNX_UINT8_PATH = str(PROJECT_ROOT / "models" / "person_detector_uint8.onnx")
VAL_IMG_DIR = PROJECT_ROOT / "data" / "processed" / "images" / "val"


def run_aidge_quantization():
    # 1. Charger le graphe ONNX dans AIDGE
    graph = aidge_onnx.load_onnx(ONNX_FP32_PATH)
    graph.set_backend("cpu")

    # 2. Optimiser le graphe (fusion Conv2d + BatchNorm2d)
    aidge_core.fuse_batchnorm(graph)

    # 3. Calibration sur 100 images réelles QQVGA
    calibration_set = []
    for img_path in sorted(VAL_IMG_DIR.glob("*.png"))[:100]:
        with Image.open(img_path) as img:
            arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
            arr = arr.reshape((1, 1, 120, 160))
            calibration_set.append(aidge_core.Tensor(arr))

    # 4. Quantification uint8 exigée par ColibryNPU
    aidge_quantization.quantize_network(
        network=graph,
        nb_bits=8,
        calibration_set=calibration_set,
        target_type=aidge_core.dtype.uint8,
    )

    # 5. Export du modèle quantifié prêt pour le compilateur ColibryNPU
    aidge_onnx.export_onnx(graph, ONNX_UINT8_PATH)
    print(f"Modèle quantifié uint8 sauvegardé dans : {ONNX_UINT8_PATH}")


if __name__ == "__main__":
    run_aidge_quantization()