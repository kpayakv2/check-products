"""การนำเข้าต้องรู้ว่ากฎอ่านตัวเลขว่าอย่างไรกับแต่ละคู่ที่จับคู่ได้

wizard ตัดสินว่าสินค้านำเข้า "มีในสตอกแล้ว" (ทิ้งโดยไม่มีคนเห็น) จากคะแนนความคล้ายอย่างเดียว
ซึ่งวัดกับไฟล์ทดสอบ 405 รายการแล้วทิ้งผิด 3 จาก 37 (`กะละมัง 004 58` ถูกจับคู่กับ `004/53`)
ฝั่ง backend จึงต้องส่งคำตัดสินของกฎไปด้วย เพื่อให้ frontend ไม่ต้องมีกฎซ้ำอีกภาษาหนึ่ง
"""
from __future__ import annotations

from src.api.models import ProductMatchResult
from src.api.routers.internal_match import annotate_identity


def match(new: str, old: str) -> ProductMatchResult:
    return ProductMatchResult(
        id="review_1",
        newProduct=new,
        oldProduct=old,
        similarity=0.97,
        confidence=0.95,
        mlPrediction="similar",
        reason="ทดสอบ",
    )


def verdicts(*pairs: tuple[str, str]) -> list[str | None]:
    return [m.identityVerdict for m in annotate_identity([match(a, b) for a, b in pairs])]


def test_mismatched_numbers_are_reported_as_different():
    assert verdicts(("กะละมัง 004 58 สี sp", "กะละมัง 004/53 สี SP")) == ["different"]


def test_case_only_difference_is_reported_as_duplicate():
    assert verdicts(("ถังปูน 15 สี sp", "ถังปูน 15 สี SP")) == ["duplicate"]


def test_rule_silence_is_reported_as_needs_judgment():
    assert verdicts(("เก้าอี้เตี้ย 610 ดำ cs", "เก้าอี้เตี้ย 610 ดำ")) == ["needs_judgment"]


def test_every_match_gets_a_verdict_in_order():
    assert verdicts(
        ("ถ่านCR1620", "ถ่านCR1616"),
        ("ถังปูน 15 สี sp", "ถังปูน 15 สี SP"),
        ("แปรงสีฟันคอลเกต", "แปรงสีฟันคอลเกตสลิมซอฟ"),
    ) == ["different", "duplicate", "needs_judgment"]
