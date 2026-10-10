"""คัดคู่สินค้าที่ไม่ต้องให้คนดูแล้ว ออกจากคิวของการสแกนหาสินค้าซ้ำในสตอก

เดิมการสแกนคัดออกเฉพาะคู่ที่ `reviewed=true` ซึ่งทำให้คู่ที่ embedding จับผิด
(`หม้อหู - จระเข้ #22` กับ `#28` ได้คะแนน 0.996) ถูกยื่นให้คนดูซ้ำทุกครั้งที่สแกน
ตอนนี้กฎอ่านตัวเลขคัดคู่พวกนั้นออกที่ต้นทาง — ดู docs/adr/0002-numeric-veto-before-ml-matching.md
"""
from __future__ import annotations

from src.api.routers.internal_match import is_pair_resolved

PAIR = ("a", "b")


class TestAlreadyReviewedByHuman:
    def test_reviewed_pair_stays_out_of_the_queue(self):
        assert is_pair_resolved(PAIR, {PAIR}, "ถังปูน 15 สี sp", "ถังปูน 15 สี SP") is True


class TestRuleDecides:
    def test_numeric_mismatch_is_dropped_without_human_review(self):
        assert (
            is_pair_resolved(PAIR, set(), "หม้อหู - จระเข้ #22 TNH", "หม้อหู - จระเข้ #28 TNH")
            is True
        )

    def test_rule_silence_keeps_the_pair_in_the_queue(self):
        """กฎเงียบเมื่อมีเลขฝั่งเดียว — คู่นั้นต้องถึงมือคน ไม่ใช่ถูกคัดออกเงียบๆ"""
        assert (
            is_pair_resolved(PAIR, set(), "ขวดซอสใหญ่ (สีหวาน) 7026 BG", "ขวดซอสใหญ่(สีหวาน)")
            is False
        )

    def test_rule_calling_a_pair_duplicate_still_needs_a_person(self):
        """กฎตัดออกได้เฉพาะฝั่ง "คนละสินค้า" — ฝั่งที่ว่าซ้ำยังต้องให้คนยืนยันก่อนรวมสินค้า"""
        assert (
            is_pair_resolved(PAIR, set(), "ถังปูน 15 สี sp", "ถังปูน 15 สี SP")
            is False
        )
