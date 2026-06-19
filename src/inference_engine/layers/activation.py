import torch.nn.functional as F

ACT2FN = {
    "silu": F.silu,
    "gelu": F.gelu,
    "relu": F.relu,
}


def get_activation(name: str):
    return ACT2FN[name]
