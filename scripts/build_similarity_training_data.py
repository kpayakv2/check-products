#!/usr/bin/env python3
"""สร้างข้อมูลฝึกสอนคู่สินค้าซ้ำ/ไม่ซ้ำ ลงตาราง `similarity_matches`

ทำไมต้องมี: `retrain_model()` อ่านเฉพาะ `similarity_matches` ที่ `reviewed=true`
แต่ไม่มีโค้ดไหนในระบบเขียนลงตารางนี้เลย — `internal_match.py` มีแค่ read (บรรทัด 109)
ผลการสแกนเก็บใน dict ในหน่วยความจำ (`SCAN_TASKS`) จึงหายทุกครั้งที่รีสตาร์ท
ทำให้ ML ไม่มีทางมีข้อมูลเทรน

เกณฑ์แบ่งชั้นตาม `taxonomy-app/DEDUPLICATION_WORKFLOW.md`:
    ความคล้าย >= 0.95        → ซ้ำ (เฉลยอัตโนมัติ)
    0.75 <= ความคล้าย < 0.95 → ให้คนตัดสิน (เข้าคิวรีวิว)
    ความคล้าย < 0.75         → ไม่ซ้ำ (เฉลยอัตโนมัติ)

ข้อควรรู้เรื่องคุณภาพเฉลย: เฉลยอัตโนมัติสองปลายมาจากเกณฑ์ความคล้ายเอง โมเดลจึงเรียน
"ค่าความคล้ายสูง = ซ้ำ" เป็นพื้นฐานก่อน คุณค่าจริงจะมาเมื่อคนรีวิวคู่ในช่วงกลาง
ซึ่งเป็นช่วงที่ความคล้ายอย่างเดียวตัดสินไม่ได้ แล้วเทรนซ้ำ

รันซ้ำได้ — ลบคู่ที่สคริปต์นี้สร้างไว้ก่อนเสมอ

    .venv/Scripts/python.exe scripts/build_similarity_training_data.py --dry-run
    .venv/Scripts/python.exe scripts/build_similarity_training_data.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv  # noqa: E402
from supabase import create_client  # noqa: E402

load_dotenv(BASE_DIR / "taxonomy-app" / ".env.local")

ALGORITHM = "legacy_embedding_scan"
SOURCE = "legacy_labelled"

DUPLICATE_THRESHOLD = 0.95
REVIEW_THRESHOLD = 0.75

# เลือก negative จากช่วงที่ใกล้เส้นแบ่งด้วย เพราะคู่ที่ต่างกันชัดเจนอยู่แล้วสอนอะไรโมเดลไม่ได้มาก
HARD_NEGATIVE_FLOOR = 0.55


def fetch_products(supabase) -> List[dict]:
    products: List[dict] = []
    page = 0
    while True:
        rows = (
            supabase.table("products")
            .select("id, name_th, embedding")
            .eq("metadata->>source", SOURCE)
            .not_.is_("embedding", "null")
            .range(page * 1000, page * 1000 + 999)
            .execute()
            .data
        )
        products.extend(rows)
        if len(rows) < 1000:
            return products
        page += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-queue-size", type=int, default=400,
                        help="จำนวนคู่ช่วงกลางที่ส่งให้คนรีวิว (เรียงจากคล้ายมากสุด)")
    parser.add_argument("--negatives", type=int, default=700,
                        help="จำนวนคู่ 'ไม่ซ้ำ' ที่ใช้เป็นตัวอย่างฝั่งตรงข้าม")
    parser.add_argument("--dry-run", action="store_true", help="แสดงผลอย่างเดียว")
    args = parser.parse_args()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        print("❌ ไม่พบ SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY", file=sys.stderr)
        return 1

    supabase = create_client(url, key)
    products = fetch_products(supabase)
    if len(products) < 2:
        print("❌ สินค้าไม่พอ — รัน scripts/import_legacy_products.py ก่อน", file=sys.stderr)
        return 1

    ids = [p["id"] for p in products]
    names = [p["name_th"] for p in products]
    matrix = np.array(
        [json.loads(p["embedding"]) if isinstance(p["embedding"], str) else p["embedding"]
         for p in products],
        dtype=np.float32,
    )
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    print(f"📦 สินค้า {len(ids)} รายการ")

    similarity = matrix @ matrix.T
    rows_i, rows_j = np.triu_indices(len(ids), k=1)
    scores = similarity[rows_i, rows_j]
    print(f"🔗 คู่ทั้งหมด {len(scores):,}")

    duplicates = np.flatnonzero(scores >= DUPLICATE_THRESHOLD)
    review_band = np.flatnonzero((scores >= REVIEW_THRESHOLD) & (scores < DUPLICATE_THRESHOLD))
    hard_negatives = np.flatnonzero((scores >= HARD_NEGATIVE_FLOOR) & (scores < REVIEW_THRESHOLD))

    # ช่วงกลางมีเป็นหมื่นคู่ เกินกว่าคนจะรีวิวไหว จึงเอาเฉพาะคู่ที่คล้ายที่สุด
    # (เอกสาร workflow แนะนำให้ "ตรวจสอบสินค้าที่มี similarity สูงก่อน")
    review_band = review_band[np.argsort(-scores[review_band])][: args.review_queue_size]

    rng = random.Random(42)
    negatives = list(hard_negatives)
    rng.shuffle(negatives)
    negatives = negatives[: args.negatives]

    print(f"   ซ้ำ (>= {DUPLICATE_THRESHOLD}, เฉลยอัตโนมัติ) : {len(duplicates):,}")
    print(f"   ให้คนดู ({REVIEW_THRESHOLD}-{DUPLICATE_THRESHOLD})        : {len(review_band):,} "
          f"(จากทั้งหมด {int(((scores >= REVIEW_THRESHOLD) & (scores < DUPLICATE_THRESHOLD)).sum()):,})")
    print(f"   ไม่ซ้ำ (เฉลยอัตโนมัติ)              : {len(negatives):,}")

    def build(index_array, *, is_duplicate: bool | None, reviewed: bool, match_type: str):
        built = []
        for k in index_array:
            a, b = int(rows_i[k]), int(rows_j[k])
            built.append({
                "product_a_id": ids[a],
                "product_b_id": ids[b],
                "similarity_score": float(scores[k]),
                "match_type": match_type,
                "algorithm": ALGORITHM,
                "is_duplicate": is_duplicate,
                "reviewed": reviewed,
                "metadata": {
                    "product_a_name": names[a],
                    "product_b_name": names[b],
                    "labelled_by": "threshold" if reviewed else "pending_human",
                },
            })
        return built

    rows = (
        build(duplicates, is_duplicate=True, reviewed=True, match_type="auto_duplicate")
        + build(negatives, is_duplicate=False, reviewed=True, match_type="auto_different")
        + build(review_band, is_duplicate=None, reviewed=False, match_type="needs_review")
    )

    if args.dry_run:
        print(f"\n(dry-run — จะเขียน {len(rows):,} แถว)")
        for row in rows[:3]:
            print(f"   {row['similarity_score']:.3f} {row['metadata']['product_a_name'][:28]} "
                  f"| {row['metadata']['product_b_name'][:28]}")
        return 0

    existing = (
        supabase.table("similarity_matches").select("id").eq("algorithm", ALGORITHM).execute().data
    )
    if existing:
        for start in range(0, len(existing), 100):
            ids_chunk = [row["id"] for row in existing[start:start + 100]]
            supabase.table("similarity_matches").delete().in_("id", ids_chunk).execute()
        print(f"🗑️  ลบคู่ที่สร้างรอบก่อน {len(existing):,} แถว")

    inserted = 0
    for start in range(0, len(rows), 200):
        supabase.table("similarity_matches").insert(rows[start:start + 200]).execute()
        inserted += len(rows[start:start + 200])
        print(f"   บันทึก {inserted:,}/{len(rows):,}", end="\r")
    print()

    print(f"✅ เขียน similarity_matches {inserted:,} แถว "
          f"(reviewed={len(duplicates) + len(negatives):,} พร้อมเทรน, "
          f"รอคนรีวิว {len(review_band):,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
