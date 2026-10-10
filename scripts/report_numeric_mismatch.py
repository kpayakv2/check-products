"""รายงานผลของกฎอ่านตัวเลขบนคู่สินค้าที่ยังไม่มีคนตรวจ — ไม่เขียนฐานข้อมูล

ใช้ `src/core/product_identity.classify_pair` ตัดสินทุกคู่ใน `similarity_matches`
ที่ `reviewed=false` แล้วรายงานว่ากฎจัดการได้เท่าไร เหลือให้คนเท่าไร

นอกจากรายงาน ยังเขียนชุดที่คนต้องตรวจออกเป็น JSON ตามที่ตกลงไว้:
คู่ที่กฎวีโต้ซึ่ง embedding มั่นใจสูงสุด 25 คู่ (จุดที่กฎกับโมเดลเถียงกันแรงที่สุด)
บวกกลุ่มที่กฎประกาศว่าซ้ำทั้งหมด (เกณฑ์ตั้งไว้ 100% จึงต้องดูครบ ไม่ใช่สุ่ม)

ตัวเลขที่ได้จากชุดนี้เป็น**ขอบล่าง**ของความแม่น ไม่ใช่ค่าจริง เพราะเลือกคู่ที่ยากที่สุดมาตรวจ

ใช้งาน:
    .venv/Scripts/python.exe scripts/report_numeric_mismatch.py
    .venv/Scripts/python.exe scripts/report_numeric_mismatch.py --out review_set.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv  # noqa: E402
from supabase import create_client  # noqa: E402

from src.core.product_identity import PairVerdict, classify_pair  # noqa: E402

load_dotenv(BASE_DIR / "taxonomy-app" / ".env.local")

FETCH_PAGE_SIZE = 1000
HARDEST_VETOES_TO_REVIEW = 25


def fetch_unreviewed_pairs(supabase) -> List[Dict]:
    """ดึงคู่ที่ยังไม่มีคนตรวจพร้อมชื่อสินค้าทั้งสองฝั่ง

    ต้องวนหน้า — มี 1,738 คู่ ซึ่งเกินเพดาน 1,000 แถวของ Supabase
    ถ้าไม่วนจะได้มาไม่ครบแบบเงียบๆ แล้วรายงานตัวเลขที่ผิด
    """
    rows: List[Dict] = []
    page = 0
    while True:
        response = (
            supabase.table("similarity_matches")
            .select(
                "id, similarity_score, match_type, "
                "product_a:products!similarity_matches_product_a_id_fkey(name_th), "
                "product_b:products!similarity_matches_product_b_id_fkey(name_th)"
            )
            .eq("reviewed", False)
            .range(page * FETCH_PAGE_SIZE, (page + 1) * FETCH_PAGE_SIZE - 1)
            .execute()
        )
        batch = response.data or []
        rows.extend(batch)
        if len(batch) < FETCH_PAGE_SIZE:
            return rows
        page += 1


def judge_all(rows: List[Dict]) -> List[Dict]:
    judged = []
    for row in rows:
        name_a = (row.get("product_a") or {}).get("name_th")
        name_b = (row.get("product_b") or {}).get("name_th")
        if not (name_a and name_b):
            continue
        judged.append(
            {
                "id": row["id"],
                "name_a": name_a,
                "name_b": name_b,
                "score": round(float(row["similarity_score"] or 0), 3),
                "verdict": classify_pair(name_a, name_b).name,
            }
        )
    return judged


def build_review_set(judged: List[Dict]) -> List[Dict]:
    vetoes = sorted(
        (p for p in judged if p["verdict"] == PairVerdict.DIFFERENT.name),
        key=lambda p: p["score"],
        reverse=True,
    )[:HARDEST_VETOES_TO_REVIEW]
    duplicates = [p for p in judged if p["verdict"] == PairVerdict.DUPLICATE.name]
    return vetoes + duplicates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="เขียนชุดที่คนต้องตรวจเป็นไฟล์ JSON")
    args = parser.parse_args()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        print("❌ ไม่พบ SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY", file=sys.stderr)
        return 1

    rows = fetch_unreviewed_pairs(create_client(url, key))
    judged = judge_all(rows)
    if not judged:
        print("ไม่พบคู่ที่ยังไม่มีคนตรวจ")
        return 0

    tally = Counter(p["verdict"] for p in judged)
    total = len(judged)
    handled = tally[PairVerdict.DIFFERENT.name] + tally[PairVerdict.DUPLICATE.name]

    print(f"คู่ที่ยังไม่มีคนตรวจ {total} คู่\n")
    for verdict in PairVerdict:
        count = tally[verdict.name]
        print(f"   {verdict.value:<20} {count:>5} คู่  ({count/total:.1%})")
    print(f"\nกฎจัดการได้ {handled} คู่ ({handled/total:.1%}) เหลือให้คน {total-handled} คู่")

    review_set = build_review_set(judged)
    vetoes_in_set = sum(1 for p in review_set if p["verdict"] == PairVerdict.DIFFERENT.name)
    print(
        f"\nชุดที่คนต้องตรวจ {len(review_set)} คู่ "
        f"(วีโต้ที่ยากสุด {vetoes_in_set} + ประกาศซ้ำทั้งหมด {len(review_set)-vetoes_in_set})"
    )

    if args.out:
        Path(args.out).write_text(
            json.dumps(review_set, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"เขียนชุดตรวจลง {args.out}")
    else:
        print("(ใส่ --out <ไฟล์> เพื่อเขียนชุดตรวจออกเป็น JSON)")

    print("\nไม่ได้เขียนฐานข้อมูลใดๆ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
