"""ย้ายสินค้านำเข้าที่ตัดสินไปแล้วเข้ากองที่ถูกต้องตามกฎอ่านตัวเลข

ไฟล์ทดสอบ 405 รายการเก็บอยู่ใน DB: ทิ้งอัตโนมัติ ("มีในสตอกแล้ว") 37 รายการ กฎไม่เห็นด้วย 3 และ
กองก้ำกึ่ง 147 รายการ กฎว่าคนละสินค้า 77 — ก่อนนโยบายใหม่ทั้งสองกองถูกตัดสินด้วยคะแนนอย่างเดียว
สคริปต์ย้ายเฉพาะที่กฎว่า "คนละสินค้า" ไปกอง "ของใหม่" ซึ่งยังมีคนยืนยันหมวดต่อ
ที่กฎเงียบหรือเห็นด้วยกับการตัดสินเดิมจะไม่ถูกแตะ
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "reroute_import_batch_by_rule.py"


def load():
    spec = importlib.util.spec_from_file_location("reroute_import_batch_by_rule", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reroute_target = load().reroute_target

NEW = "pending_review_category"


def test_discarded_product_the_rule_calls_different_goes_back_to_new():
    # คู่จริงที่ถูกทิ้งผิดในไฟล์ทดสอบ
    assert reroute_target("rejected", "กะละมัง 004 58 สี sp", "กะละมัง 004/53 สี SP") == NEW


def test_uncertain_product_the_rule_calls_different_goes_to_new():
    assert reroute_target("pending_review_dedup", "มีด478 kw", "มีด 476 KW") == NEW


def test_discard_the_rule_confirms_stays_discarded():
    assert reroute_target("rejected", "ถังปูน 15 สี sp", "ถังปูน 15 สี SP") is None


def test_discard_the_rule_is_silent_about_is_left_for_a_person():
    # เก้าอี้เตี้ย 610 ดำ cs / 610 ดำ — กฎเงียบ ไม่แตะ อย่าเดาแทนคน
    assert reroute_target("rejected", "เก้าอี้เตี้ย 610 ดำ cs", "เก้าอี้เตี้ย 610 ดำ") is None


def test_uncertain_product_the_rule_cannot_judge_stays_uncertain():
    assert reroute_target("pending_review_dedup", "แปรงสีฟันคอลเกต", "แปรงสีฟันคอลเกตสลิมซอฟ") is None


def test_products_already_in_the_new_pile_are_not_touched():
    assert reroute_target("pending_review_category", "ถ่านCR1620", "ถ่านCR1616") is None


def test_a_product_with_no_recorded_match_is_not_touched():
    assert reroute_target("rejected", "ถ่านCR1620", None) is None
    assert reroute_target("rejected", "ถ่านCR1620", "") is None
