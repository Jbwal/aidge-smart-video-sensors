import aidge_core
import aidge_backend_cpu
import aidge_learning
import aidge_onnx

# Option 1 : Construire le modèle directement avec les opérateurs AIDGE supportés par ColibryNPU
def build_aidge_model():
    model = aidge_core.Sequential([
        # Bloc 1 : 120x160 -> 60x80
        aidge_core.PaddedConv2D(1, 16, kernel_dims=[3, 3], stride_dims=[2, 2], padding_dims=[1, 1, 1, 1], name="conv1"),
        aidge_core.BatchNorm2D(16, name="bn1"),
        aidge_core.ReLU(name="relu1"),
        
        # Bloc 2 : 60x80 -> 30x40
        aidge_core.PaddedConvDepthWise2D(16, kernel_dims=[3, 3], stride_dims=[2, 2], padding_dims=[1, 1, 1, 1], name="dw2"),
        aidge_core.Conv2D(16, 32, kernel_dims=[1, 1], name="pw2"),
        aidge_core.BatchNorm2D(32, name="bn2"),
        aidge_core.ReLU(name="relu2"),
        
        # Bloc 3 : 30x40 -> 15x20
        aidge_core.PaddedConvDepthWise2D(32, kernel_dims=[3, 3], stride_dims=[2, 2], padding_dims=[1, 1, 1, 1], name="dw3"),
        aidge_core.Conv2D(32, 64, kernel_dims=[1, 1], name="pw3"),
        aidge_core.BatchNorm2D(64, name="bn3"),
        aidge_core.ReLU(name="relu3"),
        
        # Tête de détection : 5 canaux (conf, x, y, w, h)
        aidge_core.Conv2D(64, 5, kernel_dims=[1, 1], name="det_head")
    ])
    model.set_backend("cpu")
    model.set_datatype(aidge_core.dtype.float32)
    return model