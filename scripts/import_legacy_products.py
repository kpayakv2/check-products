#!/usr/bin/env python3
"""นำสินค้าเก่าที่คนจัดหมวดไว้แล้วเข้าตาราง products

ต่างจาก `scripts/import_lessons.py` เดิมตรงที่ใช้ `src/utils/legacy_dataset` ซึ่ง
จัดการ encoding สองชั้นและรวมชื่อหมวดที่ต่างกันแค่ช่องว่างให้แล้ว และคำนวณ embedding
เป็นชุดผ่าน FastAPI (ทั้ง 3,103 รายการใช้เวลาราวครึ่งนาที) แทนการ insert ทีละแถว

หมวดที่ใส่มาจากคนจัด ไม่ใช่ AI — เป็นจุดตั้งต้นให้ AI ตรวจซ้ำใน
`scripts/recheck_legacy_categories.py`

รันซ้ำได้ — ลบสินค้าที่นำเข้าด้วยสคริปต์นี้ทิ้งก่อนเสมอ

    .venv/Scripts/python.exe scripts/import_legacy_products.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv  # noqa: E402
from supabase import create_client  # noqa: E402

from src.utils.legacy_dataset import load_legacy_products  # noqa: E402

load_dotenv(BASE_DIR / "taxonomy-app" / ".env.local")

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
SOURCE = "legacy_labelled"
EMBED_BATCH = 200
INSERT_BATCH = 200


def embed_all(texts: List[str]) -> List[List[float]]:
    """ขอ embedding เป็นชุดจาก FastAPI"""
    vectors: List[List[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        chunk = texts[start:start + EMBED_BATCH]
        request = urllib.request.Request(
            f"{FASTAPI_URL}/api/embed/batch",
            data=json.dumps({"texts": chunk}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            vectors.extend(json.loads(response.read())["embeddings"])
        print(f"   embedding {len(vectors)}/{len(texts)}", end="\r")
    print()
    return vectors


def main() -> int:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        print("❌ ไม่พบ SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY", file=sys.stderr)
        return 1

    supabase = create_client(url, key)
    products = load_legacy_products()
    print(f"📦 อ่านข้อมูลเก่าได้ {len(products)} รายการ")

    nodes = supabase.table("taxonomy_nodes").select("id, name_th, level").execute().data
    name_to_id: Dict[str, str] = {n["name_th"]: n["id"] for n in nodes if n["level"] == 1}

    unmapped = sorted({p.sub_category for p in products} - set(name_to_id))
    if unmapped:
        print(f"❌ หมวดเหล่านี้ยังไม่มีใน taxonomy_nodes: {unmapped}", file=sys.stderr)
        print("   รัน migration เพิ่มหมวดก่อน แล้วค่อยนำเข้าใหม่", file=sys.stderr)
        return 1

    # ลบของเดิมก่อน เพื่อให้รันซ้ำได้โดยไม่เกิดข้อมูลซ้อน
    existing = supabase.table("products").select("id").eq(
        "metadata->>source", SOURCE).execute().data
    if existing:
        for start in range(0, len(existing), 100):
            ids = [row["id"] for row in existing[start:start + 100]]
            supabase.table("products").delete().in_("id", ids).execute()
        print(f"🗑️  ลบสินค้าที่นำเข้ารอบก่อน {len(existing)} รายการ")

    print("🧠 คำนวณ embedding...")
    embeddings = embed_all([p.name for p in products])

    batch = supabase.table("imports").insert({
        "name": f"Legacy labelled products - {datetime.now():%Y-%m-%d %H:%M}",
        "file_name": "รายการสินค้าพร้อมหมวดหมู่_AI.txt",
        "status": "processing",
        "total_records": len(products),
    }).execute().data[0]

    rows = [
        {
            "name_th": product.name,
            "sku": product.sku,
            "category_id": name_to_id[product.sub_category],
            "embedding": embedding,
            # หมวดมาจากคนจัด ไม่ใช่ AI จึงถือว่ามั่นใจเต็ม จนกว่าจะมีคนแก้หลัง AI ตรวจซ้ำ
            "confidence_score": 1.0,
            "status": "approved",
            "import_batch_id": batch["id"],
            "metadata": {
                "source": SOURCE,
                "clean_name": product.clean_name,
                "legacy_main_category": product.main_category,
                "legacy_sub_category": product.sub_category,
            },
        }
        for product, embedding in zip(products, embeddings)
    ]

    inserted = 0
    for start in range(0, len(rows), INSERT_BATCH):
        supabase.table("products").insert(rows[start:start + INSERT_BATCH]).execute()
        inserted += len(rows[start:start + INSERT_BATCH])
        print(f"   insert {inserted}/{len(rows)}", end="\r")
    print()

    supabase.table("imports").update({
        "status": "completed",
        "processed_records": inserted,
        "success_records": inserted,
        "error_records": 0,
        "completed_at": datetime.now().isoformat(),
    }).eq("id", batch["id"]).execute()

    print(f"✅ นำเข้าสินค้า {inserted} รายการ (batch {batch['id']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
