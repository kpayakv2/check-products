"""บันทึกผลที่คนตัดสินรอบสอง (481 รายการ) ลงฐานข้อมูล

อ่านจาก `tests/fixtures/dedup_human_labels_v2.json` ซึ่งเป็นคำตอบของผู้ใช้เองผ่านหน้าตรวจ
รายการมีสองชนิด และตอบแล้วเกิดผลต่างกัน:

  สินค้านำเข้า (kind=import) — สินค้าใหม่ที่ wizard ยังตัดสินไม่ได้ว่าซ้ำกับของในสตอกหรือไม่
      ตัวเดียวกัน → status 'rejected' ("มีในสตอกแล้ว") + คู่ซ้ำที่คนยืนยัน
      คนละตัว     → status 'pending_review_category' ("ของใหม่" ยังรอคนยืนยันหมวด) + คู่ที่คนยืนยันว่าไม่ซ้ำ
  คู่ในสตอก (kind=pair) — สองรายการที่อยู่ในสตอกแล้ว
      คนละตัว     → ตั้ง reviewed=true, is_duplicate=false
      ตัวเดียวกัน → **ไม่เขียนอะไร** เพราะการรวมคือการถอดสินค้าออกจากสตอก ต้องให้คนเลือกฝั่งที่จะเก็บ
                    ที่หน้าตรวจซ้ำก่อน ถ้าตีตราว่าตรวจแล้วโดยยังไม่รวม คู่นั้นจะหายจากคิวทั้งที่ยังซ้ำอยู่
  ไม่แน่ใจ ไม่เขียนอะไรทั้งสองชนิด

ก่อนเขียนทุกแถวจะตรวจสถานะปัจจุบันใน DB ถ้ารายการนั้นถูกแก้ไปแล้ว (เช่น คนไปตัดสินที่หน้า Verify
ระหว่างนี้) จะข้ามและรายงาน ไม่เขียนทับคำตัดสินที่ใหม่กว่า

ใช้งาน:
    .venv/Scripts/python.exe scripts/apply_human_dedup_labels_r2.py            # ดูก่อน ไม่เขียน
    .venv/Scripts/python.exe scripts/apply_human_dedup_labels_r2.py --apply    # เขียนจริง
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv  # noqa: E402
from supabase import create_client  # noqa: E402

load_dotenv(BASE_DIR / "taxonomy-app" / ".env.local")

LABELS_FILE = BASE_DIR / "tests" / "fixtures" / "dedup_human_labels_v2.json"
FETCH_PAGE_SIZE = 1000
IN_CHUNK = 100


@dataclass
class Operations:
    product_status: Dict[str, str] = field(default_factory=dict)     # product_id -> status ใหม่
    pair_inserts: List[dict] = field(default_factory=list)           # แถวใหม่ใน similarity_matches
    pairs_marked_different: Dict[str, str] = field(default_factory=dict)  # match_id -> reviewed_at
    to_merge: List[str] = field(default_factory=list)                # match_id ที่คนบอกว่าซ้ำ รอเลือกฝั่งรวม
    skipped: List[str] = field(default_factory=list)                 # id ที่ทำต่อไม่ได้ (เช่น ไม่รู้ว่าซ้ำกับของไหน)


def plan_operations(labels: List[dict], stock_match_of: Mapping[str, str]) -> Operations:
    """คำตอบของคน → สิ่งที่ต้องเขียน ไม่แตะ DB

    stock_match_of: id สินค้านำเข้า -> id สินค้าในสตอกที่จับคู่ไว้ (อ่านจาก products.metadata)
    """
    ops = Operations()
    for row in labels:
        verdict, ident, at = row["human_verdict"], row["id"], row["labelled_at"]
        if verdict == "unsure":
            continue

        if row["kind"] == "import":
            stock_id = stock_match_of.get(ident)
            if not stock_id:
                ops.skipped.append(ident)
                continue
            same = verdict == "same"
            ops.product_status[ident] = "rejected" if same else "pending_review_category"
            ops.pair_inserts.append(
                {
                    "product_a_id": ident,
                    "product_b_id": stock_id,
                    "similarity_score": row.get("embedding_score") or 0,
                    "match_type": "import_duplicate",
                    "algorithm": "import_wizard_dedup",
                    "is_duplicate": same,
                    "reviewed": True,
                    "reviewed_at": at,
                    "metadata": {
                        "product_a_name": row["name_a"],
                        "labelled_by": "human_review_r2",
                    },
                }
            )
        elif verdict == "diff":
            ops.pairs_marked_different[ident] = at
        else:  # pair + same — รอเลือกฝั่งที่จะเก็บ
            ops.to_merge.append(ident)
    return ops


def fetch_in(supabase, table: str, columns: str, ids: List[str]) -> List[dict]:
    rows: List[dict] = []
    for start in range(0, len(ids), IN_CHUNK):
        rows += supabase.table(table).select(columns).in_("id", ids[start : start + IN_CHUNK]).execute().data or []
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="เขียนจริง (ค่าเริ่มต้นคือดูก่อน)")
    args = parser.parse_args()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        print("❌ ไม่พบ SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY", file=sys.stderr)
        return 1
    supabase = create_client(url, key)

    labels = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    import_ids = [r["id"] for r in labels if r["kind"] == "import" and r["human_verdict"] != "unsure"]
    pair_ids = [r["id"] for r in labels if r["kind"] == "pair" and r["human_verdict"] == "diff"]

    products = {p["id"]: p for p in fetch_in(supabase, "products", "id, status, metadata", import_ids)}
    matches = {m["id"]: m for m in fetch_in(supabase, "similarity_matches", "id, reviewed", pair_ids)}

    # การ์ด: ทำเฉพาะรายการที่ยังอยู่ในสภาพที่ตอนตรวจ — ถ้าถูกแก้ไปแล้วคือมีคำตัดสินที่ใหม่กว่า ห้ามทับ
    stale: List[str] = []
    current: List[dict] = []
    for row in labels:
        if row["kind"] == "import" and row["human_verdict"] != "unsure":
            p = products.get(row["id"])
            if not p or p["status"] != "pending_review_dedup":
                stale.append(row["id"])
                continue
        if row["kind"] == "pair" and row["human_verdict"] == "diff":
            m = matches.get(row["id"])
            if not m or m["reviewed"]:
                stale.append(row["id"])
                continue
        current.append(row)

    stock_match_of = {
        pid: (p["metadata"] or {}).get("matched_product_id")
        for pid, p in products.items()
        if (p["metadata"] or {}).get("matched_product_id")
    }
    ops = plan_operations(current, stock_match_of)

    to_rejected = sum(1 for s in ops.product_status.values() if s == "rejected")
    print(f"คำตอบของคน {len(labels)} รายการ")
    print(f"   สินค้านำเข้า → ตั้งเป็น 'มีในสตอกแล้ว' {to_rejected} · เป็นของใหม่ {len(ops.product_status) - to_rejected}")
    print(f"   คู่ในสตอก    → บันทึกว่าคนละสินค้า {len(ops.pairs_marked_different)}")
    print(f"   ตอบ 'ตัวเดียวกัน' ที่ยังไม่รวม (รอเลือกฝั่งที่เก็บ) {len(ops.to_merge)}")
    print(f"   ข้าม (ถูกแก้ไปแล้วหรือหาคู่ในสตอกไม่เจอ) {len(stale) + len(ops.skipped)}")

    if not args.apply:
        print("\n(โหมดดูก่อน ไม่ได้เขียนอะไร — ใส่ --apply เพื่อเขียนจริง)")
        return 0

    for pid, status in ops.product_status.items():
        metadata = dict((products[pid]["metadata"] or {}))
        metadata.update({"dedup_bucket": "duplicate" if status == "rejected" else "new", "decided_by": "human_review_r2"})
        supabase.table("products").update({"status": status, "metadata": metadata}).eq("id", pid).execute()
    for start in range(0, len(ops.pair_inserts), IN_CHUNK):
        supabase.table("similarity_matches").upsert(
            ops.pair_inserts[start : start + IN_CHUNK], on_conflict="product_a_id,product_b_id"
        ).execute()
    for match_id, at in ops.pairs_marked_different.items():
        supabase.table("similarity_matches").update(
            {"reviewed": True, "reviewed_at": at, "is_duplicate": False}
        ).eq("id", match_id).execute()
    print("   เขียนเสร็จ — กำลังตรวจกับ DB")

    # ตรวจผลจริงจาก DB ไม่เชื่อค่าที่ API ตอบกลับมาเพียงอย่างเดียว
    problems = []
    got = {p["id"]: p["status"] for p in fetch_in(supabase, "products", "id, status", list(ops.product_status))}
    problems += [f"สินค้า {pid}: คาด {s} ได้ {got.get(pid)}" for pid, s in ops.product_status.items() if got.get(pid) != s]
    done = {m["id"]: m for m in fetch_in(supabase, "similarity_matches", "id, reviewed, is_duplicate", list(ops.pairs_marked_different))}
    problems += [
        f"คู่ {mid}: ไม่ตรง ({done.get(mid)})"
        for mid in ops.pairs_marked_different
        if not (done.get(mid) and done[mid]["reviewed"] and done[mid]["is_duplicate"] is False)
    ]
    if problems:
        print(f"\n⚠️  ผลไม่ตรง {len(problems)} จุด:", *problems[:8], sep="\n   ", file=sys.stderr)
        return 1

    print(f"\n✅ บันทึกแล้ว — สินค้า {len(ops.product_status)} รายการ · คู่ที่คนยืนยันว่าไม่ซ้ำ {len(ops.pairs_marked_different)}")
    print(f"   ที่ตอบ 'ตัวเดียวกัน' {len(ops.to_merge)} คู่ยังอยู่ในคิว รอเลือกฝั่งที่เก็บที่หน้าตรวจซ้ำ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
