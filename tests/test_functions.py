import pandas as pd

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
