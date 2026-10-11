"""ย้ายสินค้านำเข้าที่ตัดสินไปแล้ว เข้ากองที่ถูกต้องตามกฎอ่านตัวเลข

ไฟล์ทดสอบ 405 รายการ (นำเข้า 27 ส.ค. ผ่าน wizard) ถูกตัดสินด้วยคะแนนความคล้ายอย่างเดียว
ก่อนที่นโยบายจะเปลี่ยนให้กฎอ่านตัวเลขมีส่วนตัดสิน (docs/adr/0002) ผลที่ค้างอยู่ใน DB:

  * กอง "มีในสตอกแล้ว" (rejected) 37 รายการ — กฎว่าคนละสินค้า 3 รายการ นั่นคือสินค้าใหม่ที่ถูกทิ้งผิด
    เช่น `กะละมัง 004 58` ถูกจับคู่กับ `004/53`
  * กอง "ก้ำกึ่ง" (pending_review_dedup) 147 รายการ — กฎว่าคนละสินค้า 77 รายการ ไม่ควรกินเวลาคนดู

สคริปต์ย้ายเฉพาะที่กฎว่า "คนละสินค้า" ไปกอง "ของใหม่" (pending_review_category) ซึ่งยังมีคนยืนยันหมวด
ต่ออีกด่าน การย้ายผิดจึงไม่ถาวร ที่กฎเงียบหรือเห็นด้วยกับการตัดสินเดิมไม่ถูกแตะ

เก็บสถานะและ bucket เดิมไว้ใน metadata ให้ --revert คืนได้ตรงเดิม และสำหรับสินค้าที่เคยถูกทิ้ง
จะแก้คู่ใน similarity_matches ที่บอกว่า "ซ้ำ" ให้เป็น "คนละสินค้า" ด้วย ไม่งั้นข้อมูลขัดกันเอง
(สินค้าไม่ถูกทิ้งแล้ว แต่คู่ยังบอกว่าซ้ำกับของในสตอก)

ไม่แตะช่อง `reviewed` — การย้ายนี้เป็นคำตัดสินของกฎ ไม่ใช่ของคน

ใช้งาน:
    .venv/Scripts/python.exe scripts/reroute_import_batch_by_rule.py            # ดูก่อน ไม่เขียน
    .venv/Scripts/python.exe scripts/reroute_import_batch_by_rule.py --apply    # เขียนจริง
    .venv/Scripts/python.exe scripts/reroute_import_batch_by_rule.py --revert   # ย้อนกลับ
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv  # noqa: E402
from supabase import create_client  # noqa: E402

from src.core.product_identity import PairVerdict, classify_pair  # noqa: E402

load_dotenv(BASE_DIR / "taxonomy-app" / ".env.local")

NEW_STATUS = "pending_review_category"
# สองกองที่ตัดสินด้วยคะแนนอย่างเดียวและกฎอาจเห็นต่าง — ไม่รวมกอง "ของใหม่" ที่อยู่ถูกกองอยู่แล้ว
REROUTABLE = {"rejected", "pending_review_dedup"}
FETCH_PAGE_SIZE = 1000


def reroute_target(status: str, name: str, matched_name: Optional[str]) -> Optional[str]:
    """กองที่สินค้านี้ควรอยู่ ถ้าต่างจากกองปัจจุบัน; None = ไม่แตะ"""
    if status not in REROUTABLE or not matched_name:
        return None
    if classify_pair(name, matched_name) is PairVerdict.DIFFERENT:
        return NEW_STATUS
    return None


def fetch_products(supabase, statuses) -> List[Dict]:
    rows: List[Dict] = []
    page = 0
    while True:
        batch = (
            supabase.table("products")
            .select("id, name_th, status, metadata")
            .eq("metadata->>source", "import_wizard")
            .in_("status", list(statuses))
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--apply", action="store_true", help="เขียนจริง (ค่าเริ่มต้นคือดูก่อน)")
    group.add_argument("--revert", action="store_true", help="ย้อนกลับไปสถานะเดิม")
    args = parser.parse_args()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        print("❌ ไม่พบ SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY", file=sys.stderr)
        return 1
    supabase = create_client(url, key)

    if args.revert:
        # คืนเฉพาะที่สคริปต์นี้เคยย้าย — ดูจากรอยใน metadata ไม่ใช่เดาจากสถานะ
        moved = [p for p in fetch_products(supabase, [NEW_STATUS]) if (p["metadata"] or {}).get("rerouted_by") == "numeric_rule"]
        print(f"พบ {len(moved)} รายการที่เคยถูกย้ายโดยกฎ — จะคืนสถานะเดิม")
        plan = [(p, p["metadata"]["rerouted_from_status"]) for p in moved]
    else:
        plan = []
        for p in fetch_products(supabase, REROUTABLE):
            matched = (p["metadata"] or {}).get("duplicate_of")
            target = reroute_target(p["status"], p["name_th"], matched)
            if target:
                plan.append((p, target))
        by_origin = Counter(p["status"] for p, _ in plan)
        print(f"กฎว่า \"คนละสินค้า\" แต่ถูกตัดสินไปคนละกอง {len(plan)} รายการ -> ย้ายไป 'ของใหม่' ({NEW_STATUS})")
        for status, count in sorted(by_origin.items()):
            label = "มีในสตอกแล้ว (ถูกทิ้งผิด)" if status == "rejected" else "ก้ำกึ่ง"
            print(f"   จากกอง {label:<28} {count:>4} รายการ")
        for p, _ in [x for x in plan if x[0]["status"] == "rejected"]:
            print(f"      ✗ เคยถูกทิ้ง: {p['name_th']}  ||  {(p['metadata'] or {}).get('duplicate_of')}")

    if not plan:
        print("ไม่มีอะไรต้องย้าย")
        return 0

    if not (args.apply or args.revert):
        print("\n(โหมดดูก่อน ไม่ได้เขียนอะไร — ใส่ --apply เพื่อเขียนจริง)")
        return 0

    done = 0
    for product, target in plan:
        metadata = dict(product["metadata"] or {})
        was_rejected = product["status"] == "rejected" if not args.revert else metadata.get("rerouted_from_status") == "rejected"

        if args.revert:
            metadata["dedup_bucket"] = metadata.pop("original_dedup_bucket", metadata.get("dedup_bucket"))
            for trace in ("rerouted_by", "rerouted_from_status"):
                metadata.pop(trace, None)
        else:
            metadata.update(
                {
                    "rerouted_by": "numeric_rule",
                    "rerouted_from_status": product["status"],
                    "original_dedup_bucket": metadata.get("dedup_bucket"),
                    "dedup_bucket": "new",
                }
            )

        supabase.table("products").update({"status": target, "metadata": metadata}).eq("id", product["id"]).execute()

        # สินค้าที่เคยถูกทิ้งมีคู่ import_duplicate ผูกอยู่ — ต้องไปทางเดียวกับสถานะสินค้า
        if was_rejected:
            supabase.table("similarity_matches").update({"is_duplicate": bool(args.revert)}).eq(
                "product_a_id", product["id"]
            ).eq("match_type", "import_duplicate").execute()

        done += 1
        print(f"   เขียนแล้ว {done}/{len(plan)}", end="\r", flush=True)
    print()

    # ตรวจผลจริงจาก DB ไม่เชื่อค่าที่ API ตอบกลับมาเพียงอย่างเดียว
    ids = [p["id"] for p, _ in plan]
    expected = Counter(target for _, target in plan)
    actual = Counter(
        row["status"]
        for start in range(0, len(ids), 100)
        for row in supabase.table("products").select("status").in_("id", ids[start : start + 100]).execute().data
    )
    if actual != expected:
        print(f"\n⚠️  ผลไม่ตรง: คาด {dict(expected)} ได้ {dict(actual)}", file=sys.stderr)
        return 1

    print(f"\n✅ {'ย้อนกลับ' if args.revert else 'ย้าย'}แล้ว {done} รายการ — {dict(actual)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
