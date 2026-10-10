"""ปลดป้าย reviewed=true ที่เครื่องประทับให้ตัวเอง บนคู่สินค้าจาก legacy_embedding_scan

การสแกนหาสินค้าซ้ำในสตอก (`algorithm='legacy_embedding_scan'`) ตัดสินคู่สินค้าด้วย
cosine similarity ล้วนๆ แล้วเขียน reviewed=true ให้ตัวเอง ทั้งที่ไม่มีคนตรวจ ส่งผลสองชั้น:

  1. `ml_feedback_learning._fetch_training_data()` ดึงแถว reviewed=true ไปสอนโมเดล
     → โมเดลถูกสอนให้เลียนแบบความผิดของ embedding
  2. `internal_match.py` กรองคู่ reviewed=true ออกไม่ให้แสดงอีก
     → คู่ที่ผิดถูกซ่อนจากคนถาวร ไม่มีวันถูกแก้

วัดแล้วว่าผิดจริง: 644 คู่ที่ระบบว่า "สินค้าเดียวกัน" มี 412 คู่ (64%) ที่ตัวเลขในชื่อ
ไม่ตรงกัน เช่น `ถ่านCR1620` ↔ `ถ่านCR1616`, `กะละมัง 45cm` ↔ `35cm` — คนละ SKU ทั้งนั้น
สาเหตุคือ embedding มอง `100g` กับ `150g` เกือบเหมือนกัน แต่ในร้านของใช้
ตัวเลขนั่นแหละคือตัวแยก SKU

ตัวชี้ที่ใช้เลือกแถว: `reviewed_at IS NULL` — แถวที่เครื่องประทับตราเองไม่มี timestamp เลย
ส่วนแถวจากเส้นทางอื่น (import_wizard_dedup, cosine) มี reviewed_at ครบทุกแถว
จึงแยกออกจากกันได้สะอาด โดยไม่ไปแตะงานที่คนทำ

สคริปต์นี้ไม่ลบอะไร — เปลี่ยนแค่ reviewed เป็น false ส่วน match_type กับ is_duplicate
คงไว้เป็นบันทึกว่าเครื่องเดาอะไรไว้ ย้อนกลับได้ด้วย --revert

ใช้งาน:
    .venv/Scripts/python.exe scripts/unmark_machine_reviewed_matches.py            # ดูก่อน ไม่เขียน
    .venv/Scripts/python.exe scripts/unmark_machine_reviewed_matches.py --apply    # เขียนจริง
    .venv/Scripts/python.exe scripts/unmark_machine_reviewed_matches.py --revert   # ย้อนกลับ
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv  # noqa: E402
from supabase import create_client  # noqa: E402

load_dotenv(BASE_DIR / "taxonomy-app" / ".env.local")

TARGET_ALGORITHM = "legacy_embedding_scan"

# เฉพาะคู่ที่เครื่องตัดสินเอง — ไม่รวม 'needs_review' ซึ่งเป็นคู่ที่รอคนตรวจอยู่แล้ว
# ถ้าไม่กรองด้วย match_type โหมด --revert จะไปประทับตรา "ตรวจแล้ว" ให้คู่ที่ยังไม่มีใครตรวจ
MACHINE_DECIDED_TYPES = ["auto_duplicate", "auto_different"]

# Supabase คืนค่าทีละ 1000 แถวเป็นค่าเริ่มต้น และแถวเป้าหมายมี 1,344 แถว
# ถ้าไม่วนหน้าจะได้มาไม่ครบแบบเงียบๆ แล้วแก้ไปแค่ส่วนเดียว
FETCH_PAGE_SIZE = 1000

# ขนาดก้อนตอนเขียนกลับ — in_() ที่ยาวเกินไปทำให้ URL ยาวเกินขีดจำกัดของ PostgREST
UPDATE_CHUNK_SIZE = 200


def fetch_target_ids(supabase, reviewed: bool) -> List[Dict]:
    """ดึงแถวที่เครื่องประทับตราเอง — reviewed ตามที่ขอ และไม่มี reviewed_at"""
    rows: List[Dict] = []
    page = 0
    while True:
        response = (
            supabase.table("similarity_matches")
            .select("id, match_type, is_duplicate")
            .eq("algorithm", TARGET_ALGORITHM)
            .in_("match_type", MACHINE_DECIDED_TYPES)
            .eq("reviewed", reviewed)
            .is_("reviewed_at", "null")
            .range(page * FETCH_PAGE_SIZE, (page + 1) * FETCH_PAGE_SIZE - 1)
            .execute()
        )
        batch = response.data or []
        rows.extend(batch)
        if len(batch) < FETCH_PAGE_SIZE:
            return rows
        page += 1


def set_reviewed(supabase, ids: List[str], value: bool) -> int:
    """เขียน reviewed ทีละก้อน คืนจำนวนแถวที่เขียนสำเร็จ"""
    written = 0
    for start in range(0, len(ids), UPDATE_CHUNK_SIZE):
        chunk = ids[start : start + UPDATE_CHUNK_SIZE]
        response = (
            supabase.table("similarity_matches")
            .update({"reviewed": value})
            .in_("id", chunk)
            .execute()
        )
        written += len(response.data or [])
        print(f"   เขียนแล้ว {written}/{len(ids)}", end="\r", flush=True)
    print()
    return written


def describe(rows: List[Dict]) -> None:
    by_type = Counter(row["match_type"] for row in rows)
    for match_type, count in sorted(by_type.items()):
        print(f"   {match_type:<16} {count:>5} คู่")


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

    # โหมดปกติมองหาแถวที่ยังติดป้ายอยู่ (reviewed=true) แล้วปลดเป็น false
    # --revert ทำกลับกัน: มองหาแถวที่ถูกปลดไปแล้ว แล้วประทับตรากลับ
    looking_for = not args.revert
    target_value = args.revert

    rows = fetch_target_ids(supabase, reviewed=looking_for)
    if not rows:
        print(f"ไม่พบแถวที่ต้องแก้ (algorithm={TARGET_ALGORITHM}, reviewed={looking_for}, reviewed_at ว่าง)")
        return 0

    action = "ย้อนกลับเป็น reviewed=true" if args.revert else "ปลดป้าย reviewed เป็น false"
    print(f"พบ {len(rows)} คู่ที่เครื่องประทับตราเอง — {action}")
    describe(rows)

    if not (args.apply or args.revert):
        print("\n(โหมดดูก่อน ไม่ได้เขียนอะไร — ใส่ --apply เพื่อเขียนจริง)")
        return 0

    written = set_reviewed(supabase, [row["id"] for row in rows], target_value)

    # ตรวจผลจริงจาก DB ไม่เชื่อค่าที่ API ตอบกลับมาเพียงอย่างเดียว
    remaining = fetch_target_ids(supabase, reviewed=looking_for)
    if remaining:
        print(f"\n⚠️  เขียนได้ {written} แถว แต่ยังเหลือ {len(remaining)} แถวที่ไม่เปลี่ยน", file=sys.stderr)
        return 1

    print(f"\n✅ เรียบร้อย — {written} คู่กลับเข้าคิวให้คนตรวจแล้ว" if not args.revert
          else f"\n✅ ย้อนกลับแล้ว {written} คู่")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
