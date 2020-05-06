import torch
import gin


def device(type_str: str, index: int = 0):
    return torch.device(type_str, index)


gin.external_configurable(device, module="torch")
