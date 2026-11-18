import math
import numpy as np
import aidge_core
import aidge_backend_cpu

STRIDE = 4
IMAGE_HEIGHT = 120
IMAGE_WIDTH = 160


def create_person_detector_aidge(batch_size=16):
    """
    Architecture 6 couches 3x3 équilibrée (rapide sur CPU, gain de signal exact Kaiming=1.0).
    Entrée : [batch_size, 1, 120, 160]
    Sortie : [batch_size, 5, 30, 40] (stride global = 4)
    """
    model = aidge_core.sequential([
        # Bloc 1 : 120x160 -> 60x80 (stride 2)
        aidge_core.PaddedConv2D(1, 24, kernel_dims=[3, 3], stride_dims=[2, 2], padding_dims=[1, 1, 1, 1], name="conv1"),
        aidge_core.ReLU(name="relu1"),

        # Bloc 2 : 60x80 -> 30x40 (stride 2 -> grille 30x40)
        aidge_core.PaddedConv2D(24, 48, kernel_dims=[3, 3], stride_dims=[2, 2], padding_dims=[1, 1, 1, 1], name="conv2"),
        aidge_core.ReLU(name="relu2"),

        # Bloc 3 : 30x40 -> 30x40
        aidge_core.PaddedConv2D(48, 64, kernel_dims=[3, 3], stride_dims=[1, 1], padding_dims=[1, 1, 1, 1], name="conv3"),
        aidge_core.ReLU(name="relu3"),

        # Bloc 4 : 30x40 -> 30x40
        aidge_core.PaddedConv2D(64, 96, kernel_dims=[3, 3], stride_dims=[1, 1], padding_dims=[1, 1, 1, 1], name="conv4"),
        aidge_core.ReLU(name="relu4"),

        # Bloc 5 : 30x40 -> 30x40
        aidge_core.PaddedConv2D(96, 64, kernel_dims=[3, 3], stride_dims=[1, 1], padding_dims=[1, 1, 1, 1], name="conv5"),
        aidge_core.ReLU(name="relu5"),

        # Tête finale FCOS : 64 -> 5 canaux [left, top, right, bottom, objectness_logit]
        aidge_core.Conv2D(64, 5, kernel_dims=[1, 1], name="det_head"),
    ])

    model.set_backend("cpu")
    model.set_datatype(aidge_core.dtype.float32)
    model.forward_dims(dims=[[batch_size, 1, IMAGE_HEIGHT, IMAGE_WIDTH]], allow_data_dependency=True)

    # Initialisation Kaiming He EXACTE (sans facteur réducteur qui étouffe le signal)
    for node in model.get_nodes():
        op_type = node.type()
        op = node.get_operator()

        if op_type in ("Conv2D", "PaddedConv2D"):
            w_tensor = op.get_input(1)
            b_tensor = op.get_input(2)
            w_shape = w_tensor.dims
            fan_in = w_shape[1] * w_shape[2] * w_shape[3]
            std = math.sqrt(2.0 / fan_in)

            if "det_head" in node.name():
                std = 0.02

            aidge_core.normal_filler(w_tensor, 0.0, std)

            if b_tensor is not None:
                aidge_core.constant_filler(b_tensor, 0.0)
                if "det_head" in node.name():
                    b_np = np.zeros(b_tensor.dims, dtype=np.float32)
                    b_np[0:4] = 3.0
                    b_np[4] = math.log(0.10 / 0.90)
                    b_prod = node.get_parent(2)
                    if b_prod is not None:
                        b_prod.get_operator().set_output(0, aidge_core.Tensor(b_np))

    return model
