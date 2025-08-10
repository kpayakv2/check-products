import torch

from sentence_transformers.util import cos_sim


def test_cos_sim_similarity_properties():
    emb = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    sims = cos_sim(emb, emb)

    diag = torch.diag(sims)
    assert torch.all(diag > 0.99)

    mask = ~torch.eye(sims.size(0), dtype=torch.bool)
    assert torch.all(sims[mask] < 0.999)

    assert sims[0, 1] > sims[0, 2]
