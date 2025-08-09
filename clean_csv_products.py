import argparse
import os
from pathlib import Path

import pandas as pd


def run(input_csv: Path | None = None, output_csv: Path | None = None) -> None:
    """Clean a POS CSV file and output a single-column CSV of unique product names."""
    input_csv = (
        Path(os.getenv("POS_CSV", "pos_products.csv")) if input_csv is None else Path(input_csv)
    ).expanduser().resolve()
    output_csv = (
        Path(os.getenv("CLEANED_CSV", "cleaned_products.csv"))
        if output_csv is None
        else Path(output_csv)
    ).expanduser().resolve()

    df = pd.read_csv(input_csv, header=3)
    df = df.rename(columns={"รายการ": "name"})
    if "name" not in df.columns:
        print("Columns:", df.columns.tolist())
        print("กรุณาตรวจสอบชื่อคอลัมน์สินค้าในไฟล์ CSV แล้วแก้ไขให้ถูกต้อง")
        return

    df["name"] = df["name"].astype(str).str.strip()
    df = df[~df["name"].str.lower().str.contains("nan|^$|วันที่ เวลา สร้างสินค้า", na=True)]
    df = df.drop_duplicates(subset=["name"])
    df[["name"]].to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"บันทึกไฟล์ {output_csv} แล้ว")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean POS CSV to unique product list")
    parser.add_argument("--input-csv", help="POS CSV file to clean")
    parser.add_argument("--output-csv", help="Destination for cleaned CSV")
    args = parser.parse_args()
    run(args.input_csv, args.output_csv)
