import torch
import torch.nn.functional as F


def cos_sim(a, b):
    a = F.normalize(a, p=2, dim=-1)
    b = F.normalize(b, p=2, dim=-1)
    return torch.matmul(a, b.T)
