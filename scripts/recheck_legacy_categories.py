#!/usr/bin/env python3
"""ให้ AI ตรวจซ้ำหมวดหมู่ของสินค้าเก่าที่คนจัดไว้ แล้วบันทึกเฉพาะที่เห็นต่าง

เขียนผลลง `product_category_suggestions` โดย **ใส่ `product_id` ชี้กลับไปที่สินค้าเดิม**
ต่างจาก import flow เดิมที่สร้าง suggestion ลอยๆ ด้วย `suggestion_method='hybrid_ai_preview'`
ทำให้ query หารายการที่ AI เห็นต่างจากคนได้ด้วย

    SELECT ... FROM product_category_suggestions s
    JOIN products p ON p.id = s.product_id
    WHERE s.suggestion_method = 'recheck_legacy'
      AND s.suggested_category_id IS DISTINCT FROM p.category_id

รันซ้ำได้ — ลบผลตรวจรอบก่อนทิ้งก่อนเสมอ

    .venv/Scripts/python.exe scripts/recheck_legacy_categories.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict, List

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv  # noqa: E402
from supabase import create_client  # noqa: E402

load_dotenv(BASE_DIR / "taxonomy-app" / ".env.local")

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
SOURCE = "legacy_labelled"
METHOD = "recheck_legacy"
CLASSIFY_BATCH = 200
INSERT_BATCH = 200


def classify_all(names: List[str]) -> List[dict]:
    """จัดหมวดเป็นชุดผ่าน FastAPI"""
    results: List[dict] = []
    for start in range(0, len(names), CLASSIFY_BATCH):
        chunk = names[start:start + CLASSIFY_BATCH]
        request = urllib.request.Request(
            f"{FASTAPI_URL}/api/classify/batch",
            data=json.dumps({"products": chunk, "top_k": 3}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            payload = json.loads(response.read())
        results.extend(payload.get("results") or payload.get("predictions") or [])
        print(f"   จัดหมวด {len(results)}/{len(names)}", end="\r")
    print()
    return results


def fetch_all_products(supabase) -> List[dict]:
    """ดึงสินค้าทีละหน้า — Supabase จำกัด 1000 แถวต่อ query"""
    products: List[dict] = []
    page = 0
    while True:
        rows = (
            supabase.table("products")
            .select("id, name_th, category_id")
            .eq("metadata->>source", SOURCE)
            .range(page * 1000, page * 1000 + 999)
            .execute()
            .data
        )
        products.extend(rows)
        if len(rows) < 1000:
            return products
        page += 1


def main() -> int:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        print("❌ ไม่พบ SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY", file=sys.stderr)
        return 1

    supabase = create_client(url, key)

    products = fetch_all_products(supabase)
    if not products:
        print("❌ ไม่พบสินค้าที่นำเข้า — รัน scripts/import_legacy_products.py ก่อน", file=sys.stderr)
        return 1
    print(f"📦 สินค้าที่จะตรวจซ้ำ {len(products)} รายการ")

    existing = (
        supabase.table("product_category_suggestions")
        .select("id").eq("suggestion_method", METHOD).execute().data
    )
    if existing:
        for start in range(0, len(existing), 100):
            ids = [row["id"] for row in existing[start:start + 100]]
            supabase.table("product_category_suggestions").delete().in_("id", ids).execute()
        print(f"🗑️  ลบผลตรวจรอบก่อน {len(existing)} รายการ")

    print("🤖 ให้ AI จัดหมวดใหม่...")
    predictions = classify_all([p["name_th"] for p in products])
    if len(predictions) != len(products):
        print(f"❌ ผลลัพธ์ไม่ครบ ({len(predictions)}/{len(products)})", file=sys.stderr)
        return 1

    names = {
        node["id"]: node["name_th"]
        for node in supabase.table("taxonomy_nodes").select("id, name_th").execute().data
    }

    rows, stats = [], Counter()
    for product, prediction in zip(products, predictions):
        suggestions = prediction.get("suggestions") or []
        top = suggestions[0] if suggestions else None
        suggested_id = top["category_id"] if top else None

        if suggested_id is None:
            stats["ไม่มีคำตอบ"] += 1
        elif suggested_id == product["category_id"]:
            stats["ตรงกับที่คนจัด"] += 1
        else:
            stats["ต่างจากที่คนจัด"] += 1

        rows.append({
            "product_id": product["id"],
            "suggested_category_id": suggested_id,
            "confidence_score": top["confidence"] if top else 0.0,
            "suggestion_method": METHOD,
            # เก็บทั้งหมวดเดิมและหมวดที่ AI เสนอไว้ใน metadata
            # หน้ารีวิวจะได้แสดงคู่กันโดยไม่ต้อง join เพิ่ม
            "metadata": {
                "current_category_id": product["category_id"],
                "current_category": names.get(product["category_id"]),
                "suggested_category": names.get(suggested_id),
                "agrees": suggested_id == product["category_id"],
                "alternatives": [
                    {"category": names.get(s["category_id"]), "confidence": s["confidence"]}
                    for s in suggestions[1:3]
                ],
            },
        })

    inserted = 0
    for start in range(0, len(rows), INSERT_BATCH):
        supabase.table("product_category_suggestions").insert(
            rows[start:start + INSERT_BATCH]).execute()
        inserted += len(rows[start:start + INSERT_BATCH])
        print(f"   บันทึก {inserted}/{len(rows)}", end="\r")
    print()

    total = len(products)
    print(f"\n✅ ตรวจซ้ำเสร็จ {inserted} รายการ")
    for label, count in stats.most_common():
        print(f"   {label}: {count} ({count / total:.1%})")
    print(f"\n👉 รายการที่ต้องให้คนดู: {stats['ต่างจากที่คนจัด']} รายการ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
