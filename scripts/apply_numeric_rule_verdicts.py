"""บันทึกคำตัดสินของกฎอ่านตัวเลข ลง match_type ของ similarity_matches

กฎอยู่ที่ `src/core/product_identity.classify_pair` เหตุผลและผลที่วัดได้อยู่ใน
`docs/adr/0002-numeric-veto-before-ml-matching.md`

**สคริปต์นี้ไม่แตะช่อง `reviewed` เด็ดขาด** คำนั้นสงวนให้คนเท่านั้น (ดู CONTEXT.md) —
การที่เครื่องประทับ `reviewed=true` ให้ตัวเองคือบั๊กเดิมที่ทำให้ข้อมูลผิด 1,344 แถว
ถูกใช้สอน ML และถูกซ่อนจากสายตาคนไปพร้อมกัน คำตัดสินของกฎจึงไปอยู่ `match_type`
แยกของตัวเอง เพื่อให้แยกออกได้เสมอว่าอะไรมาจากเครื่องและอะไรมาจากคน

คู่ที่กฎเงียบ (`NEEDS_JUDGMENT`) ไม่ถูกแตะ ปล่อยค้างในคิวให้คนตัดสินต่อ

หมายเหตุ: คิวที่แสดงใน UI จะยังไม่ลดลงหลังรันสคริปต์นี้ เพราะ `internal_match.py`
กรองคิวด้วย `reviewed` ไม่ใช่ `match_type` การต่อสายนั้นเป็นงานแยกที่ยังไม่ได้ทำ

ใช้งาน:
    .venv/Scripts/python.exe scripts/apply_numeric_rule_verdicts.py            # ดูก่อน
    .venv/Scripts/python.exe scripts/apply_numeric_rule_verdicts.py --apply    # เขียนจริง
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

from src.core.product_identity import PairVerdict, classify_pair  # noqa: E402

load_dotenv(BASE_DIR / "taxonomy-app" / ".env.local")

FETCH_PAGE_SIZE = 1000
UPDATE_CHUNK_SIZE = 200

# match_type ของกฎ แยกจาก auto_* ของการสแกนด้วย embedding เดิม
# เพื่อให้ตอบได้ภายหลังว่าแถวนี้ใครตัดสินและด้วยเกณฑ์อะไร
RULE_MATCH_TYPE = {
    PairVerdict.DIFFERENT: ("rule_numeric_mismatch", False),
    PairVerdict.DUPLICATE: ("rule_name_identical", True),
}


def fetch_unreviewed_pairs(supabase) -> List[Dict]:
    """ดึงคู่ที่ยังไม่มีคนตรวจ — ต้องวนหน้า เพราะเกินเพดาน 1,000 แถวของ Supabase"""
    rows: List[Dict] = []
    page = 0
    while True:
        response = (
            supabase.table("similarity_matches")
            .select(
                "id, match_type, "
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="เขียนจริง (ค่าเริ่มต้นคือดูก่อน)")
    args = parser.parse_args()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        print("❌ ไม่พบ SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY", file=sys.stderr)
        return 1

    supabase = create_client(url, key)
    rows = fetch_unreviewed_pairs(supabase)

    by_verdict: Dict[PairVerdict, List[str]] = {v: [] for v in PairVerdict}
    for row in rows:
        name_a = (row.get("product_a") or {}).get("name_th")
        name_b = (row.get("product_b") or {}).get("name_th")
        if not (name_a and name_b):
            continue
        by_verdict[classify_pair(name_a, name_b)].append(row["id"])

    print(f"คู่ที่ยังไม่มีคนตรวจ {sum(len(v) for v in by_verdict.values())} คู่")
    for verdict in PairVerdict:
        count = len(by_verdict[verdict])
        target = RULE_MATCH_TYPE.get(verdict)
        note = f"-> match_type='{target[0]}'" if target else "-> ไม่แตะ ปล่อยให้คนตัดสิน"
        print(f"   {verdict.value:<20} {count:>5} คู่  {note}")

    writable = {v: ids for v, ids in by_verdict.items() if v in RULE_MATCH_TYPE and ids}
    total = sum(len(ids) for ids in writable.values())
    print(f"\nจะเขียน {total} แถว โดยไม่แตะช่อง reviewed")

    if not args.apply:
        print("\n(โหมดดูก่อน ไม่ได้เขียนอะไร — ใส่ --apply เพื่อเขียนจริง)")
        return 0

    written = 0
    for verdict, ids in writable.items():
        match_type, is_duplicate = RULE_MATCH_TYPE[verdict]
        for start in range(0, len(ids), UPDATE_CHUNK_SIZE):
            chunk = ids[start : start + UPDATE_CHUNK_SIZE]
            response = (
                supabase.table("similarity_matches")
                .update({"match_type": match_type, "is_duplicate": is_duplicate})
                .in_("id", chunk)
                .execute()
            )
            written += len(response.data or [])
            print(f"   เขียนแล้ว {written}/{total}", end="\r", flush=True)
    print()

    # ตรวจผลจริงจาก DB — ทั้งจำนวนที่เขียน และว่าไม่มีแถวไหนถูกประทับ reviewed โดยไม่ตั้งใจ
    check = (
        supabase.table("similarity_matches")
        .select("id", count="exact")
        .in_("match_type", [t for t, _ in RULE_MATCH_TYPE.values()])
        .execute()
    )
    leaked = (
        supabase.table("similarity_matches")
        .select("id", count="exact")
        .in_("match_type", [t for t, _ in RULE_MATCH_TYPE.values()])
        .eq("reviewed", True)
        .execute()
    )
    if leaked.count:
        print(f"\n⚠️  มี {leaked.count} แถวที่กฎตัดสินแต่ถูกตั้ง reviewed=true — ผิดกฎของโปรเจกต์", file=sys.stderr)
        return 1

    print(f"\n✅ เขียนแล้ว {written} แถว | รวมแถวที่กฎตัดสินในระบบ {check.count}")
    print("   ไม่มีแถวใดถูกประทับ reviewed — คำนั้นยังเป็นของคนเท่านั้น")
    print("\nคิวใน UI จะยังไม่ลดลง เพราะ internal_match.py กรองด้วย reviewed ไม่ใช่ match_type")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
