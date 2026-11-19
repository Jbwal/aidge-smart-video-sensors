import math
import numpy as np
import aidge_core
import aidge_backend_cpu

from config import (STRIDE, IMAGE_HEIGHT, IMAGE_WIDTH, MID_LAYERS, MID_CHANNELS,
                    LTRB_INIT, OBJ_PRIOR)


def _conv(cin, cout, stride, name):
    return [
        aidge_core.PaddedConv2D(cin, cout, kernel_dims=[3, 3], stride_dims=[stride, stride],
                                padding_dims=[1, 1, 1, 1], name=name),
        aidge_core.ReLU(name=name.replace("conv", "relu")),
    ]


def create_person_detector_aidge(batch_size=16):
    """
    Réseau séquentiel (Conv2D 3x3 + ReLU uniquement, sans BN, sans branche) :
      conv1 1->16  s2  (RF 3)
      conv2 16->32 s2  (RF 7)
      conv3 32->64 s2 (STRIDE=8) ou s1 (STRIDE=4)   (RF 15)
      conv4..conv(3+MID_LAYERS) 64->64 s1            (RF +16 par couche à stride 8 -> 111 px)
      det_head Conv2D 1x1 -> 5 canaux [l, t, r, b, objectness_logit]
    Entrée [B,1,120,160] -> sortie [B,5,120/STRIDE,160/STRIDE]
    ~245 k paramètres, activation max = 16x60x80 = 76.8 kB (INT8).
    """
    layers = []
    layers += _conv(1, 16, 2, "conv1")
    layers += _conv(16, 32, 2, "conv2")
    layers += _conv(32, MID_CHANNELS, 2 if STRIDE == 8 else 1, "conv3")
    for i in range(MID_LAYERS):
        layers += _conv(MID_CHANNELS, MID_CHANNELS, 1, f"conv{4 + i}")
    layers.append(aidge_core.Conv2D(MID_CHANNELS, 5, kernel_dims=[1, 1], name="det_head"))

    model = aidge_core.sequential(layers)
    model.set_backend("cpu")
    model.set_datatype(aidge_core.dtype.float32)
    model.forward_dims(dims=[[batch_size, 1, IMAGE_HEIGHT, IMAGE_WIDTH]], allow_data_dependency=True)

    for node in model.get_nodes():
        op_type = node.type()
        op = node.get_operator()
        if op_type in ("Conv2D", "PaddedConv2D"):
            w_tensor = op.get_input(1)
            b_tensor = op.get_input(2)
            w_shape = w_tensor.dims
            fan_in = w_shape[1] * w_shape[2] * w_shape[3]
            std = math.sqrt(2.0 / fan_in)
            is_head = "det_head" in node.name()
            if is_head:
                std = 0.01
            aidge_core.normal_filler(w_tensor, 0.0, std)

            if b_tensor is not None:
                aidge_core.constant_filler(b_tensor, 0.0)
                if is_head:
                    b_np = np.zeros(b_tensor.dims, dtype=np.float32)
                    b_np[0:4] = LTRB_INIT
                    b_np[4] = math.log(OBJ_PRIOR / (1.0 - OBJ_PRIOR))
                    b_prod = node.get_parent(2)
                    if b_prod is not None:
                        b_prod.get_operator().set_output(0, aidge_core.Tensor(b_np))
    return model
