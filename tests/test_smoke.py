# tests/test_smoke.py
import pandas as pd

import clean_csv_products
import filter_matched_products
import main


def test_run_creates_output(tmp_path, monkeypatch):
    # Prepare small datasets
    old_csv = tmp_path / "old.csv"
    pd.DataFrame({"name": ["apple", "banana", "cherry"]}).to_csv(old_csv, index=False)

    new_csv = tmp_path / "new.csv"
    pd.DataFrame({"รายการ": ["apple", "orange", "orange"]}).to_csv(new_csv, index=False)

    # Patch model and similarity to avoid heavy downloads
    class DummyModel:
        def encode(self, sentences, convert_to_tensor=True):
            import torch

            lengths = [float(len(s)) for s in sentences]
            return torch.tensor(lengths)

    def dummy_cos_sim(a, b):
        import torch

        return 1 / (1 + torch.abs(b - a.unsqueeze(1)))

    monkeypatch.setattr(main, "SentenceTransformer", lambda _: DummyModel())
    monkeypatch.setattr(main.util, "cos_sim", dummy_cos_sim)

    main.run(old_csv, new_csv, tmp_path)

    assert (tmp_path / "matched_products.csv").exists()


def test_clean_csv_products(tmp_path):
    raw = tmp_path / "raw.csv"
    with open(raw, "w", encoding="utf-8-sig") as f:
        f.write("a\n" * 3)
        f.write("รายการ\n")
        f.write("A\nA\nB\nวันที่ เวลา สร้างสินค้า\n")

    output = tmp_path / "clean.csv"
    clean_csv_products.run(raw, output)

    df = pd.read_csv(output)
    assert df["name"].tolist() == ["A", "B"]


def test_filter_matched_products(tmp_path):
    matched = tmp_path / "matched.csv"
    pd.DataFrame({"score": [0.95, 0.5]}).to_csv(matched, index=False)

    filter_matched_products.run(matched, tmp_path)

    assert (tmp_path / "matched_products_check.csv").exists()
    assert (tmp_path / "matched_products_unique.csv").exists()
