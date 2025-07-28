import numpy as np


class Tensor:
    def __init__(self, data):
        self.data = np.array(data, dtype=float)

    def unsqueeze(self, dim):
        if dim == 1:
            return Tensor(self.data.reshape(-1, 1))
        elif dim == 0:
            return Tensor(self.data.reshape(1, -1))
        else:
            raise NotImplementedError

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        else:
            return Tensor(self.data - other)

    def __rsub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(other.data - self.data)
        else:
            return Tensor(other - self.data)

    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        else:
            return Tensor(self.data + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        else:
            return Tensor(self.data / other)

    def __rtruediv__(self, other):
        if isinstance(other, Tensor):
            return Tensor(other.data / self.data)
        else:
            return Tensor(other / self.data)

    def __abs__(self):
        return Tensor(np.abs(self.data))

    def topk(self, k):
        idx = np.argsort(-self.data.flatten())[:k]
        vals = self.data.flatten()[idx]
        return Tensor(vals), Tensor(idx)

    def __iter__(self):
        for x in self.data.flatten():
            yield float(x)

    def __len__(self):
        return len(self.data.flatten())

    def __getitem__(self, idx):
        return Tensor(self.data[idx])

    def __float__(self):
        return float(self.data)

    def tolist(self):
        return self.data.tolist()

    def __repr__(self):
        return f"Tensor({self.data})"


def tensor(data):
    return Tensor(data)


def abs(x):
    if isinstance(x, Tensor):
        return x.__abs__()
    return Tensor(np.abs(x))
