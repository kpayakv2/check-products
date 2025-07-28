import torch

# Simple cosine similarity based on absolute difference for tests


def cos_sim(a, b):
    return 1 / (1 + torch.abs(b - a.unsqueeze(1)))
