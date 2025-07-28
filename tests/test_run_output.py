import pandas as pd

import main


def test_run_produces_matched_csv(tmp_path, monkeypatch):
    old_csv = tmp_path / "old.csv"
    pd.DataFrame({"name": ["A", "B"]}).to_csv(old_csv, index=False)

    new_csv = tmp_path / "new.csv"
    pd.DataFrame({"รายการ": ["A", "C"]}).to_csv(new_csv, index=False)

    class DummyModel:
        def encode(self, sentences, convert_to_tensor=True):
            import torch

            return torch.tensor([float(len(s)) for s in sentences])

    def dummy_cos_sim(a, b):
        import torch

        return 1 / (1 + torch.abs(b - a.unsqueeze(1)))

    monkeypatch.setattr(main, "SentenceTransformer", lambda _: DummyModel())
    monkeypatch.setattr(main.util, "cos_sim", dummy_cos_sim)

    main.run(old_csv, new_csv, tmp_path)

    assert (tmp_path / "matched_products.csv").exists()
