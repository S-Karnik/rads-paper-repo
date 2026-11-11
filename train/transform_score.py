import torch

def tanh_transform(x, scale, threshold):
    return torch.tanh(scale * (x - threshold))

def inverse_tanh_transform(y, scale, threshold):
    return (torch.atanh(y) / scale) + threshold