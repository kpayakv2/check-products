"""บันทึกผลที่คนตัดสินคู่สินค้าแล้ว ลง similarity_matches

อ่านจาก `tests/fixtures/dedup_human_labels_v1.json` ซึ่งเป็นชุดที่คนตัดป้ายเอง
แล้วตั้ง `reviewed=true` พร้อม `reviewed_at` และ `is_duplicate` ตามที่คนตัดสิน

นี่เป็นครั้งแรกที่ `reviewed=true` ถูกเขียนโดยมีคนตรวจจริง — คำนี้สงวนให้คนเท่านั้น
(ดู CONTEXT.md) ก่อนหน้านี้มี 1,344 แถวที่เครื่องประทับตราให้ตัวเอง ซึ่งถูกปลดออกไปแล้ว
ด้วย `scripts/unmark_machine_reviewed_matches.py`

`match_type` ไม่ถูกแตะ — มันบันทึกว่า**เครื่อง**เดาอะไรไว้ ส่วน `is_duplicate` บอกว่า**คน**ตอบอะไร
แยกกันสองช่องทำให้ภายหลังยังตรวจได้ว่าคนเห็นด้วยกับเครื่องหรือแก้คำตอบมัน

คู่ที่คนตอบ "ไม่แน่ใจ" จะไม่ถูกบันทึก ปล่อยค้างไว้ในคิวต่อ

ใช้งาน:
    .venv/Scripts/python.exe scripts/apply_human_dedup_labels.py            # ดูก่อน
    .venv/Scripts/python.exe scripts/apply_human_dedup_labels.py --apply    # เขียนจริง
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv  # noqa: E402
from supabase import create_client  # noqa: E402

from src.core.product_identity import PairVerdict, classify_pair  # noqa: E402

load_dotenv(BASE_DIR / "taxonomy-app" / ".env.local")

LABELS_FILE = BASE_DIR / "tests" / "fixtures" / "dedup_human_labels_v1.json"

DUPLICATE = "สินค้าซ้ำ"
DIFFERENT = "คนละสินค้า"

# คำตอบของกฎไม่ถูกเก็บในไฟล์ชุดวัดผล (มันเปลี่ยนได้ ถ้าเก็บไว้จะค้าง) จึงรันสดทุกครั้ง
# กฎที่เงียบ (NEEDS_JUDGMENT) ไม่มีใน dict นี้ และไม่นับเป็นการไม่เห็นด้วยกับคน
RULE_ANSWER = {PairVerdict.DIFFERENT: DIFFERENT, PairVerdict.DUPLICATE: DUPLICATE}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="เขียนจริง (ค่าเริ่มต้นคือดูก่อน)")
    args = parser.parse_args()

    if not LABELS_FILE.exists():
        print(f"❌ ไม่พบ {LABELS_FILE}", file=sys.stderr)
        return 1

    labels = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    decided = [r for r in labels if r["human_verdict"] in (DUPLICATE, DIFFERENT)]
    unsure = [r for r in labels if r["human_verdict"] not in (DUPLICATE, DIFFERENT)]

    tally = Counter(r["human_verdict"] for r in decided)
    print(f"ผลที่คนตัดสินแล้ว {len(labels)} คู่")
    for verdict, count in sorted(tally.items()):
        print(f"   {verdict:<16} {count:>4} คู่")
    if unsure:
        print(f"   ไม่แน่ใจ (ไม่บันทึก) {len(unsure):>4} คู่")

    spoken = [(r, RULE_ANSWER.get(classify_pair(r["name_a"], r["name_b"]))) for r in decided]
    disagreed = [(r, rule) for r, rule in spoken if rule is not None and rule != r["human_verdict"]]
    answered = sum(1 for _, rule in spoken if rule)
    print(f"\nกฎตอบ {answered} คู่ จากที่คนตัดสิน {len(decided)} คู่ — คนไม่เห็นด้วย {len(disagreed)}")
    for r, rule in disagreed:
        print(f"   กฎว่า {rule} / คนว่า {r['human_verdict']}: {r['name_a']} || {r['name_b']}")

    if not args.apply:
        print("\n(โหมดดูก่อน ไม่ได้เขียนอะไร — ใส่ --apply เพื่อเขียนจริง)")
        return 0

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        print("❌ ไม่พบ SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY", file=sys.stderr)
        return 1

    supabase = create_client(url, key)
    written = 0
    for row in decided:
        response = (
            supabase.table("similarity_matches")
            .update(
                {
                    "reviewed": True,
                    "reviewed_at": row["labelled_at"],
                    "is_duplicate": row["human_verdict"] == DUPLICATE,
                }
            )
            .eq("id", row["match_id"])
            .execute()
        )
        written += len(response.data or [])
        print(f"   เขียนแล้ว {written}/{len(decided)}", end="\r", flush=True)
    print()

    # ตรวจผลจริงจาก DB ไม่เชื่อค่าที่ API ตอบกลับมาเพียงอย่างเดียว
    check = (
        supabase.table("similarity_matches")
        .select("id", count="exact")
        .in_("id", [r["match_id"] for r in decided])
        .eq("reviewed", True)
        .execute()
    )
    confirmed = check.count or 0
    if confirmed != len(decided):
        print(f"\n⚠️  คาดว่า {len(decided)} แถว แต่ยืนยันได้ {confirmed} แถว", file=sys.stderr)
        return 1

    print(f"\n✅ บันทึกแล้ว {confirmed} คู่ — เป็นครั้งแรกที่ reviewed=true มีคนตรวจจริงอยู่เบื้องหลัง")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
