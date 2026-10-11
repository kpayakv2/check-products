"""ปลดป้าย reviewed=true ที่ route นำเข้าประทับให้การทิ้งที่เครื่องตัดสินเอง

`taxonomy-app/app/api/import/commit/route.ts` เคยเขียน `reviewed: true` ให้ทุกคู่ที่ wizard ตัดสินว่า
"มีในสตอกแล้ว" ทั้งที่ตัดสินด้วย threshold ล้วนๆ — ตัวมันเองบันทึกไว้ใน metadata ว่า
`labelled_by='threshold_import'` แต่ยังเรียกว่า "ตรวจแล้ว" ผลคือ ML เอาไปสอนตัวเอง
ซึ่งเป็นบั๊กชนิดเดียวกับการสแกนสินค้าซ้ำ (scripts/unmark_machine_reviewed_matches.py)

route แก้แล้ว สคริปต์นี้แก้ข้อมูลเดิม แถวที่เลือกคือ `metadata.labelled_by = 'threshold_import'`
เท่านั้น — แถวที่คนตัดสินในขั้นรีวิวของ wizard ใช้ป้าย `wizard_review` จึงไม่ถูกแตะ

ไม่ลบอะไร เปลี่ยน `reviewed` เป็น false และล้าง `reviewed_at` โดยเก็บเวลาเดิมไว้ใน
`metadata.original_reviewed_at` เพื่อให้ --revert คืนค่าได้ตรงเดิม

ใช้งาน:
    .venv/Scripts/python.exe scripts/unmark_machine_import_matches.py            # ดูก่อน ไม่เขียน
    .venv/Scripts/python.exe scripts/unmark_machine_import_matches.py --apply    # เขียนจริง
    .venv/Scripts/python.exe scripts/unmark_machine_import_matches.py --revert   # ย้อนกลับ
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv  # noqa: E402
from supabase import create_client  # noqa: E402

load_dotenv(BASE_DIR / "taxonomy-app" / ".env.local")

MACHINE_LABEL = "threshold_import"
FETCH_PAGE_SIZE = 1000


def fetch_rows(supabase, reviewed: bool) -> List[Dict]:
    """ดึงแถวนำเข้าที่เครื่องตัดสิน — กรองด้วยป้าย labelled_by ไม่ใช่แค่ match_type
    เพื่อไม่ไปแตะคู่ที่คนตัดสินในขั้นรีวิวของ wizard"""
    rows: List[Dict] = []
    page = 0
    while True:
        batch = (
            supabase.table("similarity_matches")
            .select("id, reviewed_at, metadata")
            .eq("match_type", "import_duplicate")
            .eq("algorithm", "import_wizard_dedup")
            .eq("reviewed", reviewed)
            .eq("metadata->>labelled_by", MACHINE_LABEL)
            .range(page * FETCH_PAGE_SIZE, (page + 1) * FETCH_PAGE_SIZE - 1)
            .execute()
            .data
            or []
        )
        rows.extend(batch)
        if len(batch) < FETCH_PAGE_SIZE:
            return rows
        page += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--apply", action="store_true", help="เขียนจริง (ค่าเริ่มต้นคือดูก่อน)")
    group.add_argument("--revert", action="store_true", help="ย้อนกลับเป็น reviewed=true")
    args = parser.parse_args()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        print("❌ ไม่พบ SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY", file=sys.stderr)
        return 1

    supabase = create_client(url, key)
    looking_for = not args.revert  # ปกติหาแถวที่ยังติดป้าย (true) / revert หาแถวที่ปลดไปแล้ว (false)
    rows = fetch_rows(supabase, reviewed=looking_for)

    if not rows:
        print(f"ไม่พบแถวที่ต้องแก้ (import_duplicate, labelled_by={MACHINE_LABEL}, reviewed={looking_for})")
        return 0

    action = "ย้อนกลับเป็น reviewed=true" if args.revert else "ปลดป้าย reviewed เป็น false"
    print(f"พบ {len(rows)} คู่ที่เครื่องตัดสินด้วย threshold ตอนนำเข้า — {action}")

    if not (args.apply or args.revert):
        print("\n(โหมดดูก่อน ไม่ได้เขียนอะไร — ใส่ --apply เพื่อเขียนจริง)")
        return 0

    written = 0
    for row in rows:
        metadata = dict(row.get("metadata") or {})
        if args.revert:
            update = {"reviewed": True, "reviewed_at": metadata.pop("original_reviewed_at", None)}
        else:
            # เก็บเวลาเดิมไว้ให้ย้อนกลับได้ตรงเดิม
            metadata["original_reviewed_at"] = row.get("reviewed_at")
            update = {"reviewed": False, "reviewed_at": None}
        update["metadata"] = metadata

        response = supabase.table("similarity_matches").update(update).eq("id", row["id"]).execute()
        written += len(response.data or [])
        print(f"   เขียนแล้ว {written}/{len(rows)}", end="\r", flush=True)
    print()

    # ตรวจผลจริงจาก DB ไม่เชื่อค่าที่ API ตอบกลับมาเพียงอย่างเดียว
    remaining = fetch_rows(supabase, reviewed=looking_for)
    if remaining:
        print(f"\n⚠️  เขียนได้ {written} แถว แต่ยังเหลือ {len(remaining)} แถวที่ไม่เปลี่ยน", file=sys.stderr)
        return 1

    print(f"\n✅ เรียบร้อย — {written} คู่" + (" ย้อนกลับแล้ว" if args.revert else " ไม่ถูกนับว่าคนตรวจอีกแล้ว"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
