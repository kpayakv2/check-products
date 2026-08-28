"""วัด accuracy ของการจัดหมวดหมู่จริง เทียบกับหมวดที่คนจัดไว้ในข้อมูลเก่า

เดิมรีโปนี้ไม่มีเทสต์วัด accuracy เลย — ตัวเลข 72% ที่อ้างในเอกสารมาจาก
`tests/benchmark_similarity.py` ซึ่ง print ข้อความ hardcode โดยไม่ได้จัดหมวดอะไร
เทสต์นี้วัดของจริงจาก test split ที่กันไว้ไม่ให้ระบบเห็นตอนเรียน

ต้องมี FastAPI ทำงานอยู่ที่ 127.0.0.1:8000 และ Supabase local — ถ้าไม่มีจะ skip
รันเดี่ยว:  .venv/Scripts/python.exe -m pytest tests/integration/test_classification_accuracy.py -v -s
"""
from __future__ import annotations

import os
from typing import Dict, List

import pytest
import requests

from src.utils.legacy_dataset import load_legacy_products, stratified_split

API_BASE_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
CLASSIFY_URL = f"{API_BASE_URL}/api/classify/category"

# Baseline วัดจาก test split เต็ม 595 รายการ — ห้ามต่ำกว่านี้ ยกขึ้นเมื่อปรับปรุงสำเร็จ
#
#   2026-08-26 ก่อนปรับปรุง               top-1 25.5% | top-3 41.8% | หมวดหลัก 42.7%
#   2026-08-26 หลังตัดคำไทย (Phase 2)     top-1 29.1% | top-3 44.5% | หมวดหลัก 47.2%
#   2026-08-26 หลังสกัดคัมภีร์ (Phase 3)  top-1 72.3% | top-3 77.1% | หมวดหลัก 80.3%
#
# หมายเหตุ: กฎที่สกัดมาอยู่ในตาราง keyword_rules (match_type='mined_legacy')
# ถ้าตัวเลขตกฮวบให้เช็คก่อนว่ากฎยังอยู่ไหม แล้วรัน scripts/mine_keywords_from_legacy.py ใหม่
BASELINE_TOP1_SUB = 0.723
BASELINE_TOP3_SUB = 0.771
BASELINE_TOP1_MAIN = 0.803

# เผื่อความแกว่งเวลาลดขนาดตัวอย่างเพื่อรันเร็ว ไม่ให้เทสต์แดงเพราะ noise
TOLERANCE = 0.03

# ค่าเริ่มต้นคือ test set ทั้งชุด เพื่อให้ตัวเลขนิ่งและเทียบข้ามเฟสได้
# ลดลงได้ด้วย ACCURACY_SAMPLE_SIZE เวลาต้องการรันเร็วๆ
SAMPLE_SIZE = int(os.getenv("ACCURACY_SAMPLE_SIZE", "0")) or None


def _service_available() -> bool:
    try:
        return requests.get(f"{API_BASE_URL}/api/v1/health", timeout=5).ok
    except requests.RequestException:
        return False


pytestmark = pytest.mark.skipif(
    not _service_available(),
    reason=f"ต้องมี FastAPI ทำงานที่ {API_BASE_URL} (รัน: python -m src.api.api_server)",
)


@pytest.fixture(scope="module")
def category_lookup() -> Dict[str, tuple]:
    """map category_id -> (ชื่อหมวด, ชื่อหมวดแม่) จาก Supabase"""
    from supabase import create_client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        pytest.skip("ไม่มี SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY ใน environment")

    client = create_client(url, key)
    rows = client.table("taxonomy_nodes").select("id, name_th, parent_id").execute().data
    names = {row["id"]: row["name_th"] for row in rows}
    return {
        row["id"]: (row["name_th"], names.get(row.get("parent_id"), ""))
        for row in rows
    }


@pytest.fixture(scope="module", autouse=True)
def _require_untainted_rules():
    """ปฏิเสธการวัดถ้ากฎถูกสกัดจากข้อมูลทั้งหมด (เห็น test set แล้ว)

    ตัวเลขที่ได้จะสูงลวงเพราะระบบเคยเห็นเฉลยของสินค้าที่กำลังทดสอบ
    ยอม skip ดีกว่ารายงานตัวเลขที่เชื่อไม่ได้
    """
    from supabase import create_client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        pytest.skip("ไม่มี SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY ใน environment")

    rows = (
        create_client(url, key)
        .table("keyword_rules")
        .select("name")
        .eq("match_type", "mined_legacy")
        .execute()
        .data
    )
    if any("[source=all]" in (row.get("name") or "") for row in rows):
        pytest.skip(
            "keyword_rules ถูกสกัดจากข้อมูลทั้งหมด (source=all) จึงเห็น test set แล้ว "
            "— วัด accuracy ไม่ได้ ให้รัน: "
            "python scripts/mine_keywords_from_legacy.py --source train"
        )


@pytest.fixture(scope="module")
def predictions() -> List[dict]:
    """จัดหมวดสินค้าใน test split แล้วเก็บผลไว้ให้ทุกเทสต์ใช้ร่วมกัน"""
    _, test_set = stratified_split(load_legacy_products())
    sample = test_set[:SAMPLE_SIZE] if SAMPLE_SIZE else test_set

    results = []
    session = requests.Session()
    for product in sample:
        response = session.post(
            CLASSIFY_URL,
            json={"product_name": product.name, "top_k": 3},
            timeout=30,
        )
        response.raise_for_status()
        suggestions = response.json().get("suggestions") or []
        results.append(
            {
                "product": product,
                "predicted_ids": [s["category_id"] for s in suggestions[:3]],
                "top_keyword": suggestions[0].get("matched_keyword") if suggestions else None,
            }
        )
    return results


def _rate(hits: int, total: int) -> float:
    return hits / total if total else 0.0


def test_top1_subcategory_accuracy(predictions, category_lookup):
    hits = sum(
        1
        for r in predictions
        if r["predicted_ids"]
        and category_lookup.get(r["predicted_ids"][0], ("", ""))[0] == r["product"].sub_category
    )
    accuracy = _rate(hits, len(predictions))
    print(f"\ntop-1 หมวดย่อย = {accuracy:.1%} ({hits}/{len(predictions)})")
    assert accuracy >= BASELINE_TOP1_SUB - TOLERANCE


def test_top3_subcategory_accuracy(predictions, category_lookup):
    hits = sum(
        1
        for r in predictions
        if any(
            category_lookup.get(cid, ("", ""))[0] == r["product"].sub_category
            for cid in r["predicted_ids"]
        )
    )
    accuracy = _rate(hits, len(predictions))
    print(f"\ntop-3 หมวดย่อย = {accuracy:.1%} ({hits}/{len(predictions)})")
    assert accuracy >= BASELINE_TOP3_SUB - TOLERANCE


def test_top1_maincategory_accuracy(predictions, category_lookup):
    hits = 0
    for r in predictions:
        if not r["predicted_ids"]:
            continue
        name, parent = category_lookup.get(r["predicted_ids"][0], ("", ""))
        if (parent or name) == r["product"].main_category:
            hits += 1
    accuracy = _rate(hits, len(predictions))
    print(f"\ntop-1 หมวดหลัก = {accuracy:.1%} ({hits}/{len(predictions)})")
    assert accuracy >= BASELINE_TOP1_MAIN - TOLERANCE


def test_every_product_gets_a_prediction(predictions):
    """ระบบต้องไม่เงียบ — ทุกสินค้าต้องได้คำตอบอย่างน้อยหนึ่งหมวด"""
    silent = [r["product"].name for r in predictions if not r["predicted_ids"]]
    assert not silent, f"ไม่ได้คำตอบ {len(silent)} รายการ เช่น {silent[:3]}"


@pytest.mark.parametrize(
    "product_name,forbidden_category",
    [
        # "สี" เคย match กลางคำ "ยาสีฟัน" แล้วลากไปหมวดสีทาบ้าน
        ("ยาสีฟันดอกบัวคู่ 150g", "สีและอุปกรณ์ทาสี"),
        ("ยาสีฟันคอลเกต 150 กรัม", "สีและอุปกรณ์ทาสี"),
    ],
)
def test_short_keyword_does_not_hijack_classification(
    product_name, forbidden_category, category_lookup
):
    """Regression: keyword สั้นต้องไม่ match กลางคำอีก (แก้ด้วยการตัดคำใน Phase 2)"""
    response = requests.post(
        CLASSIFY_URL, json={"product_name": product_name, "top_k": 1}, timeout=30
    )
    response.raise_for_status()
    predicted_id = response.json().get("category_id")
    predicted_name = category_lookup.get(predicted_id, ("", ""))[0]
    assert predicted_name != forbidden_category, (
        f"{product_name!r} ถูกจัดเข้า {forbidden_category!r} อีกแล้ว "
        "— การเทียบ keyword กลับไป match กลางคำ"
    )
