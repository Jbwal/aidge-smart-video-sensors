import random as rd
from pathlib import Path
import cv2
import numpy as np

import aidge_core
import aidge_backend_cpu
import aidge_onnx
import aidge_quantization
import aidge_export_colibry

from utils import decode_predictions, plot_batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONNX_FP32_PATH = PROJECT_ROOT / "models" / "person_detector_aidge_fp32.onnx"
VAL_IMG_DIR = PROJECT_ROOT / "data" / "processed" / "images" / "val"

BATCH_SIZE = 16
N_CALIBRATION_BATCH = 5
CONFIDENCE_THRESHOLD = 0.30


def load_val_images():
    """Charge toutes les images QQVGA de validation en mémoire."""
    paths = sorted(VAL_IMG_DIR.glob("*.png"))
    if len(paths) < BATCH_SIZE * N_CALIBRATION_BATCH:
        raise RuntimeError(f"Pas assez d'images dans {VAL_IMG_DIR}")
    return [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) for p in paths]


def run_ptq_and_export_colibry():
    aidge_core.Log.set_console_level(aidge_core.Level.Notice)

    if not ONNX_FP32_PATH.is_file():
        raise FileNotFoundError(
            f"Modèle introuvable : {ONNX_FP32_PATH}\n"
            "Lancez d'abord : python src/train_aidge.py"
        )

    val_images = load_val_images()
    rd.seed(42)

    # 1. Chargement et simplification du graphe AIDGE FP32 (comme dans bikeVision.ipynb)
    aidge_model = aidge_onnx.load_onnx(str(ONNX_FP32_PATH))
    aidge_core.remove_flatten(aidge_model)
    aidge_core.fuse_batchnorm(aidge_model)
    aidge_core.expand_metaops(aidge_model)
    aidge_model.forward_dims(dims=[[BATCH_SIZE, 1, 120, 160]], allow_data_dependency=True)
    aidge_model.compile("cpu", aidge_core.dtype.float32, dims=[[BATCH_SIZE, 1, 120, 160]])

    # 2. Évaluation et visualisation du modèle FP32 sur un batch de test
    test_indices = rd.sample(range(len(val_images)), BATCH_SIZE)
    image_batch = np.stack([val_images[i] for i in test_indices])

    # Normalisation obligatoire [-1, 1] pour la quantification AIDGE
    batch_normalized = image_batch.astype(np.float32) / 127.5 - 1.0
    input_tensor = aidge_core.Tensor(batch_normalized[:, None, :, :])
    input_tensor.to_dtype(aidge_core.dtype.float32)

    scheduler = aidge_core.SequentialScheduler(aidge_model)
    scheduler.forward(True, [input_tensor])

    output_node = aidge_model.get_ordered_outputs()[0][0]
    output_tensor = np.asarray(output_node.get_operator().get_output(0))

    fp32_predictions = decode_predictions(output_tensor, confidence_threshold=CONFIDENCE_THRESHOLD)
    plot_batch(
        image_batch,
        fp32_predictions,
        cols=4,
        figsize=(14, 10),
        title="Prédictions Modèle AIDGE FP32",
        save_name="predictions_fp32.png",
    )

    # 3. Construction du jeu de calibration (N_CALIBRATION_BATCH * BATCH_SIZE images)
    n_samples = BATCH_SIZE * N_CALIBRATION_BATCH
    calib_indices = rd.sample(range(len(val_images)), n_samples)
    calibration_set = []

    for batch_idx in range(N_CALIBRATION_BATCH):
        b_idx = calib_indices[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
        batch = np.stack([val_images[i] for i in b_idx])
        batch = batch[:, None, :, :] / 127.5 - 1.0
        calibration_set.append(aidge_core.Tensor(batch.astype(np.float32)))

    # 4. Quantification Post-Entraînement (PTQ) pour ColibryNPU
    ptq_model = aidge_model.clone()

    aidge_quantization.auto_assign_node_precision(
        graphview=ptq_model,
        act_precision=aidge_core.dtype.int8,
        weight_precision=aidge_core.dtype.int8,
        bias_precision=aidge_core.dtype.int32,
    )

    aidge_quantization.quantize_network(
        network=ptq_model,
        calibration_set=calibration_set,
        clipping_mode=aidge_quantization.Clipping.MAX,
        no_quant=False,
        optimize_signs=False,
        single_shift=False,
        fold_graph=False,
        fake_quantize=False,
        verbose=False,
        dequantize=True,
    )

    # 5. Vérification des prédictions après quantification PTQ
    quantized_batch = np.round((image_batch / 127.5 - 1.0) * 127).astype(np.int8)
    input_ptq = aidge_core.Tensor(quantized_batch[:, None, :, :])
    input_ptq.to_dtype(aidge_core.dtype.int8)

    scheduler_ptq = aidge_core.SequentialScheduler(ptq_model)
    scheduler_ptq.forward(True, [input_ptq])

    output_node_ptq = ptq_model.get_ordered_outputs()[0][0]
    output_tensor_ptq = np.asarray(output_node_ptq.get_operator().get_output(0))

    ptq_predictions = decode_predictions(output_tensor_ptq, confidence_threshold=CONFIDENCE_THRESHOLD)
    plot_batch(
        image_batch,
        ptq_predictions,
        cols=4,
        figsize=(14, 10),
        title="Prédictions Modèle AIDGE Quantifié (PTQ)",
        save_name="predictions_ptq.png",
    )

    # 6. Fusion matérielle @NNPA et Export vers ColibryNPU (.nnpa + rapport mémoire/cycles/énergie)
    input_np = quantized_batch[0][None, None, :, :]
    input_aidge = aidge_core.Tensor(input_np)
    input_aidge.to_dtype(aidge_core.dtype.int8)

    aidge_core.fuse_to_metaop(ptq_model, "@NNPA")
    export_folder = aidge_export_colibry.export(
        graph=ptq_model,
        input_tensors=[input_aidge],
        dtype="int8",
    )
    print(f"\nExport ColibryNPU (.nnpa) généré avec succès dans : {export_folder}")


if __name__ == "__main__":
    run_ptq_and_export_colibry()