from . import util


class SentenceTransformer:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, sentences, convert_to_tensor=True):
        import torch

        lengths = [float(len(s)) for s in sentences]
        tensor = torch.tensor(lengths)
        return tensor
