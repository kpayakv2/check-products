import argparse
import os
from pathlib import Path

import pandas as pd


def run(matched_csv: Path | None = None, output_dir: Path | None = None) -> None:
    """Split matched_products.csv into items that need checking and unique items."""
    matched_csv = (

        (
            Path(os.getenv("MATCHED_CSV", "matched_products.csv"))
            if matched_csv is None
            else Path(matched_csv)
        )
        .expanduser()
        .resolve()
    )
    output_dir = (
        (Path(os.getenv("OUTPUT_DIR", "output")) if output_dir is None else Path(output_dir))
        .expanduser()
        .resolve()
    )
        Path(os.getenv("MATCHED_CSV", "matched_products.csv"))
        if matched_csv is None
        else Path(matched_csv)
    ).expanduser().resolve()
    output_dir = (
        Path(os.getenv("OUTPUT_DIR", "output")) if output_dir is None else Path(output_dir)
    ).expanduser().resolve()
main
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(matched_csv)
    check_df = df[df["score"] >= 0.90].sort_values(by="score", ascending=False)
    unique_df = df[df["score"] < 0.90].sort_values(by="score", ascending=False)

    check_path = output_dir / "matched_products_check.csv"
    unique_path = output_dir / "matched_products_unique.csv"
    check_df.to_csv(check_path, index=False, encoding="utf-8-sig")
    unique_df.to_csv(unique_path, index=False, encoding="utf-8-sig")
    print(f"บันทึกไฟล์ {check_path} และ {unique_path} เรียบร้อยแล้ว")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter matched products by score")
    parser.add_argument("--matched-csv", help="CSV produced by main.py")
    parser.add_argument("--output-dir", help="Directory for filtered CSV files")
    args = parser.parse_args()
    run(args.matched_csv, args.output_dir)
