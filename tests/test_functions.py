import pandas as pd
import pytest

import main


def test_remove_duplicates_no_dupes(tmp_path):
    df = pd.DataFrame({"รายการ": ["A", "B", "C"]})
    dup_path = tmp_path / "dupes.csv"
    dedup, dupes = main.remove_duplicates(df, subset="รายการ", duplicates_path=dup_path)

    assert dupes.empty
    assert not dup_path.exists()
    assert dedup["รายการ"].tolist() == ["A", "B", "C"]


def test_remove_duplicates_with_dupes(tmp_path):
    df = pd.DataFrame({"รายการ": ["A", "B", "A", "C", "A"]})
    dup_path = tmp_path / "dupes.csv"
    dedup, dupes = main.remove_duplicates(df, subset="รายการ", duplicates_path=dup_path)

    assert dup_path.exists()
    assert dupes["รายการ"].tolist() == ["A", "A", "A"]
    assert dedup["รายการ"].tolist() == ["A", "B", "C"]


def test_check_product_similarity_ranking(monkeypatch):
    class DummyModel:
        def encode(self, sentences, convert_to_tensor=True):
            import torch

            return torch.tensor([float(len(s)) for s in sentences])

    def dummy_cos_sim(a, b):
        import torch

        return 1 / (1 + torch.abs(b - a.unsqueeze(1)))

    monkeypatch.setattr(main.util, "cos_sim", dummy_cos_sim)

    model = DummyModel()
    old_names = ["a", "ab", "abc", "abcd"]
    old_emb = model.encode(old_names)

    result = main.check_product_similarity("abcd", old_names, old_emb, model, top_k=2)
    assert result == [("abcd", 1.0), ("abc", 0.5)]


def test_check_product_similarity_negative_top_k(monkeypatch):
    class DummyModel:
        def encode(self, sentences, convert_to_tensor=True):
            import torch

            return torch.tensor([float(len(s)) for s in sentences])

    def dummy_cos_sim(a, b):
        import torch

        return 1 / (1 + torch.abs(b - a.unsqueeze(1)))

    monkeypatch.setattr(main.util, "cos_sim", dummy_cos_sim)

    model = DummyModel()
    old_names = ["a", "b"]
    old_emb = model.encode(old_names)

    with pytest.raises(ValueError):
        main.check_product_similarity("a", old_names, old_emb, model, top_k=-1)


def test_check_product_similarity_gpu_tensor(monkeypatch):
    class DummyModel:
        def encode(self, sentences, convert_to_tensor=True):
            # Embeddings are not used in the dummy cosine similarity below
            return [0] * len(sentences)

    class FakeGPUScore:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

        def cpu(self):
            return self

        def __float__(self):  # pragma: no cover - used to emulate CUDA restriction
            raise TypeError("Can't convert CUDA tensor to float")

    class FakeCosScores:
        def __init__(self, scores):
            self.scores = scores

        def topk(self, k):
            top_scores = [FakeGPUScore(v) for v in self.scores[:k]]
            top_indices = list(range(k))
            return top_scores, top_indices

    def dummy_cos_sim(a, b):
        # Return predefined scores in descending order
        return [FakeCosScores([1.0, 0.5, 0.2])]

    monkeypatch.setattr(main.util, "cos_sim", dummy_cos_sim)

    model = DummyModel()
    old_names = ["a", "b", "c"]
    old_emb = [0, 0, 0]

    result = main.check_product_similarity("new", old_names, old_emb, model, top_k=2)
    assert result == [("a", 1.0), ("b", 0.5)]
