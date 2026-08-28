#!/usr/bin/env python3
"""สร้าง "คัมภีร์" keyword จากข้อมูลสินค้าเก่าที่คนจัดหมวดไว้แล้ว

ทำไมต้องมี: 58% ของสินค้าอยู่ในหมวดที่ชื่อหมวดไม่เคยปรากฏในชื่อสินค้าเลย
เช่นหมวด `ผลิตภัณฑ์ดูแลหนังศรีษะ` มีแต่ชื่อแบรนด์แชมพู และ `ผงซักฟอก/น้ำยาซักผ้า`
มีแต่ "โอโม่" "บรีส" ความรู้ว่าแบรนด์ไหนอยู่หมวดไหนมีอยู่แค่ในข้อมูลเก่าชุดนี้เท่านั้น
อนุมานจากตัวหนังสือหรือ embedding ไม่ได้

เขียนลงตาราง `keyword_rules` (ไม่ใช่ `taxonomy_nodes.keywords` ซึ่ง
`taxonomy_service.load_taxonomy_nodes()` ดึงมาแล้วทิ้ง ไม่มีผลต่อการจัดหมวด)

ใช้ `TaxonomyService.extract_auto_keywords()` ตัวเดียวกับที่ระบบเรียนตอนคนกดยืนยันใน UI
คีย์เวิร์ดจากสองทางจะได้เป็นชนิดเดียวกัน

รันซ้ำได้ — ลบ rule เก่าที่ `match_type='mined_legacy'` ทิ้งก่อนเขียนใหม่ทุกครั้ง

    .venv/Scripts/python.exe scripts/mine_keywords_from_legacy.py
    .venv/Scripts/python.exe scripts/mine_keywords_from_legacy.py --min-precision 0.9 --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv  # noqa: E402
from supabase import create_client  # noqa: E402

from src.services.taxonomy_service import TaxonomyService  # noqa: E402
from src.utils.legacy_dataset import load_legacy_products, stratified_split  # noqa: E402

load_dotenv(BASE_DIR / "taxonomy-app" / ".env.local")

MATCH_TYPE = "mined_legacy"

# กฎที่คนเขียนไว้เดิมมี priority 6-10 และคะแนนคิดจาก match_count / priority
# ให้ค่าต่ำกว่าเพื่อให้กฎที่มาจากข้อมูลจริงชนะเมื่อขัดกัน (ตามที่ตกลงไว้)
MINED_PRIORITY = 4

# ต่อท้ายชื่อกฎเพื่อให้รู้ว่าสกัดจากข้อมูลชุดไหน
# `tests/integration/test_classification_accuracy.py` อ่านค่านี้แล้ว skip ถ้าเป็น all
# เพราะถ้ากฎเห็น test set มาแล้ว ตัวเลข accuracy จะไม่มีความหมาย
SOURCE_TAG = {
    "train": "[source=train]",
    "all": "[source=all]",
}


def build_keyword_index(products, service: TaxonomyService):
    """นับว่า keyword แต่ละคำปรากฏในหมวดไหนบ้าง กี่ครั้ง"""
    per_category: Dict[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()

    for product in products:
        # ใช้ set กันไม่ให้คำที่ซ้ำในชื่อเดียวถูกนับหลายรอบ
        for keyword in set(service.extract_auto_keywords(product.name)):
            per_category[product.sub_category][keyword] += 1
            totals[keyword] += 1

    return per_category, totals


def select_keywords(
    per_category: Dict[str, Counter],
    totals: Counter,
    min_precision: float,
    min_support: int,
    max_per_category: int,
) -> Dict[str, List[str]]:
    """เลือกเฉพาะ keyword ที่ชี้หมวดได้จริง

    เกณฑ์คือความจำเพาะ (precision) = จำนวนครั้งที่คำนี้อยู่ในหมวดนี้ ÷ จำนวนครั้งทั้งหมด
    คำอย่าง "โอโม่" จะได้ precision สูงเพราะอยู่หมวดเดียว
    ส่วนคำกว้างอย่าง "กลม" กระจายหลายหมวดจึงถูกคัดออก — เป็นบทเรียนจากกฎเดิม
    ที่ใส่ "กลม/เหลี่ยม" ไว้แล้วทำให้จัดผิด
    """
    selected: Dict[str, List[str]] = {}

    for category, counts in per_category.items():
        scored = []
        for keyword, hits in counts.items():
            if hits < min_support:
                continue
            precision = hits / totals[keyword]
            if precision < min_precision:
                continue
            # เรียงตามความจำเพาะก่อน แล้วค่อยดูว่าเจอบ่อยแค่ไหน
            scored.append((precision, hits, keyword))

        if not scored:
            continue

        scored.sort(reverse=True)
        selected[category] = [kw for _, _, kw in scored[:max_per_category]]

    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-precision", type=float, default=0.8,
                        help="สัดส่วนขั้นต่ำที่คำนี้ต้องอยู่ในหมวดเดียว (ค่าเริ่มต้น 0.8)")
    parser.add_argument("--min-support", type=int, default=2,
                        help="ต้องเจอคำนี้ในหมวดอย่างน้อยกี่สินค้า (ค่าเริ่มต้น 2)")
    parser.add_argument("--max-per-category", type=int, default=30,
                        help="จำกัดจำนวน keyword ต่อหมวด กันไม่ให้หมวดใหญ่ได้เปรียบเกิน")
    parser.add_argument("--source", choices=["train", "all"], default="all",
                        help="สกัดจากข้อมูลชุดไหน: 'train' สำหรับวัด accuracy อย่างซื่อสัตย์, "
                             "'all' สำหรับใช้งานจริงให้ระบบรู้มากที่สุด (ค่าเริ่มต้น)")
    parser.add_argument("--dry-run", action="store_true",
                        help="แสดงผลอย่างเดียว ไม่เขียนลงฐานข้อมูล")
    args = parser.parse_args()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        print("❌ ไม่พบ SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY", file=sys.stderr)
        return 1

    supabase = create_client(url, key)
    service = TaxonomyService(supabase)

    products = load_legacy_products()
    train, test = stratified_split(products)
    if args.source == "train":
        source_products = train
        print(f"📚 สกัดจาก train {len(train)} รายการ (กัน test ไว้ {len(test)} รายการ ไม่แตะ)")
    else:
        source_products = products
        print(f"📚 สกัดจากข้อมูลทั้งหมด {len(products)} รายการ (สำหรับใช้งานจริง)")
        print("   ⚠️  กฎชุดนี้เห็น test set แล้ว เทสต์วัด accuracy จะ skip ให้อัตโนมัติ")
        print("   ถ้าต้องการวัด ให้รันใหม่ด้วย --source train")

    per_category, totals = build_keyword_index(source_products, service)
    print(f"🔤 คำที่สกัดได้ทั้งหมด {len(totals)} คำ จาก {len(per_category)} หมวด")

    selected = select_keywords(
        per_category, totals,
        min_precision=args.min_precision,
        min_support=args.min_support,
        max_per_category=args.max_per_category,
    )
    total_keywords = sum(len(v) for v in selected.values())
    print(f"✅ ผ่านเกณฑ์ {total_keywords} คำ ใน {len(selected)} หมวด "
          f"(precision ≥ {args.min_precision}, support ≥ {args.min_support})")

    # map ชื่อหมวด -> id
    nodes = supabase.table("taxonomy_nodes").select("id, name_th, level").execute().data
    name_to_id = {n["name_th"]: n["id"] for n in nodes if n["level"] == 1}

    missing = sorted(set(selected) - set(name_to_id))
    if missing:
        print(f"⚠️  ไม่พบหมวดเหล่านี้ใน taxonomy_nodes: {missing}")

    if args.dry_run:
        for category in sorted(selected, key=lambda c: -len(selected[c]))[:12]:
            print(f"   {category}: {', '.join(selected[category][:10])}")
        print("\n(dry-run ไม่ได้เขียนลงฐานข้อมูล)")
        return 0

    deleted = supabase.table("keyword_rules").delete().eq("match_type", MATCH_TYPE).execute()
    print(f"🗑️  ลบกฎเดิมที่ match_type='{MATCH_TYPE}' ออก {len(deleted.data)} แถว")

    rows = []
    for category, keywords in selected.items():
        category_id = name_to_id.get(category)
        if not category_id:
            continue
        rows.append({
            "code": f"mined_{category_id[:8]}",
            "name": f"Mined from legacy data {SOURCE_TAG[args.source]}: {category}",
            "description": (
                "สกัดอัตโนมัติจากชื่อสินค้าเก่าที่คนจัดหมวดไว้แล้ว "
                f"(source={args.source}, precision≥{args.min_precision}, support≥{args.min_support})"
            ),
            "keywords": keywords,
            "category_id": category_id,
            "priority": MINED_PRIORITY,
            "match_type": MATCH_TYPE,
            "confidence_score": 0.85,
            "is_active": True,
        })

    for start in range(0, len(rows), 100):
        supabase.table("keyword_rules").insert(rows[start:start + 100]).execute()

    print(f"💾 เขียนกฎใหม่ {len(rows)} แถว (priority {MINED_PRIORITY})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
