"""คืนสินค้าที่ถูกกด "รวม" ในหน้า Verify โดยผิดพลาด

ปุ่ม merge ของ `taxonomy-app/app/api/similarity/resolve/route.ts` เปลี่ยนสถานะสินค้าฝั่งที่ถูกรวมเป็น
`duplicate` ซึ่งคือการถอดออกจากสตอก และบันทึกคู่นั้นใน similarity_matches ว่าซ้ำ — ถ้ากดพลาด
ทั้งสองอย่างต้องแก้คืน สคริปต์นี้ทำให้ครบทั้งคู่ในครั้งเดียว

กรณีที่เจอ: `ขันน้ำลายไทย 18 CM สีเงิน` กับ `ขันน้ำลายไทย 8 CM สีเงิน` ถูกกด merge ทั้งที่ขนาดต่างกัน
(คู่อื่นในชุดเดียวกัน 6 CM ↔ 14 CM และ 14 CM ↔ 30 CM ผู้ใช้กด "เก็บทั้งสอง") ตัว 8 CM จึงถูกถอดออก

ระวังเลือกให้ถูกฝั่ง: `--removed` คือสินค้าที่ถูกถอดออก (สถานะ duplicate) ส่วน `--kept` คืออีกฝั่ง
สคริปต์ปฏิเสธถ้าไม่เจอคู่ที่ตรงกับทั้งสองชื่อ หรือถ้าสินค้าฝั่ง removed ไม่ได้อยู่ในสถานะ duplicate

ใช้งาน (ค่าเริ่มต้นคือดูก่อน ไม่เขียน):
    .venv/Scripts/python.exe scripts/undo_wrong_merge.py --kept "ขันน้ำลายไทย 18 CM สีเงิน" --removed "ขันน้ำลายไทย 8 CM สีเงิน"
    ... --apply   # เขียนจริง
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv  # noqa: E402
from supabase import create_client  # noqa: E402

load_dotenv(BASE_DIR / "taxonomy-app" / ".env.local")

# สินค้าที่ถูกรวมเคยเป็น approved เสมอ — ปุ่ม merge ในหน้า Verify ทำงานกับคู่ของสินค้าในสตอก
RESTORED_STATUS = "approved"


def product_ids(supabase, name: str) -> list:
    rows = supabase.table("products").select("id, name_th, status").eq("name_th", name).execute().data or []
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kept", required=True, help="ชื่อสินค้าฝั่งที่ยังอยู่ (ไม่ถูกถอด)")
    parser.add_argument("--removed", required=True, help="ชื่อสินค้าฝั่งที่ถูกถอดออก (สถานะ duplicate)")
    parser.add_argument("--apply", action="store_true", help="เขียนจริง (ค่าเริ่มต้นคือดูก่อน)")
    args = parser.parse_args()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        print("❌ ไม่พบ SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY", file=sys.stderr)
        return 1
    supabase = create_client(url, key)

    kept, removed = product_ids(supabase, args.kept), product_ids(supabase, args.removed)
    for label, found, name in (("kept", kept, args.kept), ("removed", removed, args.removed)):
        if len(found) != 1:
            print(f"❌ ชื่อ --{label} \"{name}\" ต้องตรงกับสินค้าหนึ่งรายการพอดี แต่เจอ {len(found)}", file=sys.stderr)
            return 1
    kept_row, removed_row = kept[0], removed[0]

    if removed_row["status"] != "duplicate":
        print(f"❌ ฝั่ง --removed สถานะ '{removed_row['status']}' ไม่ใช่ duplicate — ไม่มีอะไรให้คืน หรือเลือกฝั่งผิด", file=sys.stderr)
        return 1

    ids = [kept_row["id"], removed_row["id"]]
    pair = (
        supabase.table("similarity_matches")
        .select("id, is_duplicate, metadata")
        .in_("product_a_id", ids)
        .in_("product_b_id", ids)
        # เฉพาะแถวที่คนกด merge ในหน้า Verify — คู่เดียวกันอาจมีอีกแถวจากการสแกนซึ่งไม่ใช่ที่ต้องแก้
        .eq("match_type", "semantic")
        .eq("metadata->>resolved_action", "merge")
        .execute()
        .data
        or []
    )
    if len(pair) != 1:
        print(f"❌ ต้องเจอคู่ใน similarity_matches หนึ่งแถวพอดี แต่เจอ {len(pair)}", file=sys.stderr)
        return 1
    match = pair[0]

    print(f"ฝั่งที่ยังอยู่     : {kept_row['name_th']}  [{kept_row['status']}]")
    print(f"ฝั่งที่ถูกถอดออก  : {removed_row['name_th']}  [{removed_row['status']}] -> จะคืนเป็น {RESTORED_STATUS}")
    print(f"คู่ใน similarity_matches: is_duplicate={match['is_duplicate']} -> จะแก้เป็น false (คนละสินค้า)")

    if not args.apply:
        print("\n(โหมดดูก่อน ไม่ได้เขียนอะไร — ใส่ --apply เพื่อเขียนจริง)")
        return 0

    supabase.table("products").update({"status": RESTORED_STATUS}).eq("id", removed_row["id"]).execute()

    metadata = dict(match.get("metadata") or {})
    metadata.update({"resolved_action": "keep_both", "corrected_from": "merge"})
    metadata.pop("keep_product_id", None)
    metadata.pop("removed_product_id", None)
    supabase.table("similarity_matches").update({"is_duplicate": False, "metadata": metadata}).eq("id", match["id"]).execute()

    # ตรวจผลจริงจาก DB ไม่เชื่อค่าที่ API ตอบกลับมาเพียงอย่างเดียว
    status_after = supabase.table("products").select("status").eq("id", removed_row["id"]).single().execute().data["status"]
    dup_after = supabase.table("similarity_matches").select("is_duplicate").eq("id", match["id"]).single().execute().data["is_duplicate"]
    if status_after != RESTORED_STATUS or dup_after is not False:
        print(f"\n⚠️  ผลไม่ตรง: status={status_after}, is_duplicate={dup_after}", file=sys.stderr)
        return 1

    print(f"\n✅ คืนแล้ว — {removed_row['name_th']} กลับเป็น {RESTORED_STATUS} และคู่ถูกแก้เป็นคนละสินค้า")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
