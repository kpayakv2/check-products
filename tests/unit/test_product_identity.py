"""ตัดสินว่าคู่สินค้าเป็นตัวเดียวกันหรือไม่ จากตัวเลขในชื่อ

เคสทั้งหมดเป็นคู่จริงจาก `similarity_matches` ไม่ใช่ตัวอย่างสมมติ
คู่ที่ "ห้ามวีโต้" สำคัญกว่าคู่ที่ "ต้องวีโต้" เพราะถ้าวีโต้ผิด ของซ้ำจะหลุดเข้าสตอก
"""
from __future__ import annotations

from src.core.product_identity import PairVerdict, classify_pair


class TestVetoWhenNumbersDiffer:
    """เลขในชื่อไม่ตรงกัน = คนละ SKU — embedding ให้คู่พวกนี้ถึง 0.99 ว่าเหมือนกัน"""

    def test_different_pot_size_is_different_product(self):
        assert (
            classify_pair("หม้อหู - จระเข้ #22 TNH", "หม้อหู - จระเข้ #28 TNH")
            is PairVerdict.DIFFERENT
        )

    def test_different_model_code_is_different_product(self):
        assert (
            classify_pair("ถ่านCR1620พานาโซนิค", "ถ่านCR1616พานาโซนิค")
            is PairVerdict.DIFFERENT
        )

    def test_decimal_is_one_token_not_two_digits(self):
        """`3.5` ต้องเป็นก้อนเดียว ถ้าแตกเป็น {3,5} คู่นี้จะกลายเป็นเลขตรงกันทั้งที่คนละของ"""
        assert (
            classify_pair("สายยาง 3.5 มม.", "สายยาง 5.3 มม.")
            is PairVerdict.DIFFERENT
        )


class TestPriceInName:
    """ราคาในชื่อเป็นตัวระบุรายการ ยกเว้นเขียนเป็นหมายเหตุ `(ราคา …)` — ดู CONTEXT.md"""

    def test_price_band_identifies_a_separate_stock_line(self):
        assert (
            classify_pair(
                "หวี/หวีแปรง/หวีไดร์/หวีสับ 20 บ.", "หวี/หวีแปรง/หวีไดร์/หวีสับ 45 บ."
            )
            is PairVerdict.DIFFERENT
        )

    def test_explicit_price_note_is_not_an_identifier(self):
        assert (
            classify_pair(
                "เคเบิ้ลไทร์3*100 (ราคา 25 บาท)", "เคเบิ้ลไทร์3*100 (ราคา 20 บาท)"
            )
            is PairVerdict.NEEDS_JUDGMENT
        )


class TestDeclareDuplicate:
    """เหมือนกันทุกตัวอักษรหลังตัดเครื่องหมายกับตัวพิมพ์ = ของชิ้นเดียวกัน"""

    def test_case_only_difference_is_duplicate(self):
        assert (
            classify_pair("ถังปูน 15 สี sp", "ถังปูน 15 สี SP")
            is PairVerdict.DUPLICATE
        )

    def test_punctuation_only_difference_is_duplicate(self):
        assert (
            classify_pair("กล่องล็อค 4 ด้านใส 8023 SET LOCK SW", "กล่องล็อค 4 ด้าน ใส 8023 SET LOCK SW")
            is PairVerdict.DUPLICATE
        )


class TestNeedsJudgment:
    """กฎต้องเงียบเมื่อมันไม่รู้ ห้ามเดา"""

    def test_same_numbers_but_extra_word_needs_judgment(self):
        assert (
            classify_pair("เก้าอี้เตี้ย 610 ดำ cs", "เก้าอี้เตี้ย 610 ดำ")
            is PairVerdict.NEEDS_JUDGMENT
        )

    def test_no_numbers_at_all_needs_judgment(self):
        assert (
            classify_pair("แปรงสีฟันคอลเกต", "แปรงสีฟันคอลเกตสลิมซอฟ")
            is PairVerdict.NEEDS_JUDGMENT
        )

    def test_number_on_one_side_only_with_same_words_needs_judgment(self):
        """ตัดเลขออกแล้วชื่อเหลือเหมือนกันเป๊ะ — `1*6` อาจเป็นแพ็ค 6 ก้อน ซึ่งคนต้องตัดสิน"""
        assert (
            classify_pair("สบู่อัมพิกาก้อนกลม สูตรแตงโม 1*6", "สบู่อัมพิกาก้อนกลม สูตรแตงโม")
            is PairVerdict.NEEDS_JUDGMENT
        )


class TestOneSidedNumbers:
    """เลขฝั่งเดียวตัดสินไม่ได้ — วัดแล้วได้แค่ 12/15 (80%) ต่ำกว่าเกณฑ์ 95%

    เดิมกิ่งนี้วีโต้ว่าคนละสินค้าเมื่อส่วนที่ไม่ใช่เลขต่างกันด้วย แต่ผลที่คนตรวจ 15 คู่
    ชี้ว่าความต่างนั้นมักเป็นรหัสรุ่นหรือรหัสผู้ผลิตที่อีกฝั่งไม่ได้บันทึกไว้ ไม่ใช่สินค้าคนละตัว
    จึงเปลี่ยนให้กิ่งนี้เงียบทั้งหมด — ยอมส่ง 96 คู่ให้คนดู ดีกว่าวีโต้ผิดแล้วของซ้ำหลุดเข้าสตอก
    """

    def test_supplier_code_on_one_side_needs_judgment(self):
        assert (
            classify_pair("ขวดซอสใหญ่ (สีหวาน) 7026 BG", "ขวดซอสใหญ่(สีหวาน)")
            is PairVerdict.NEEDS_JUDGMENT
        )

    def test_unrelated_products_still_need_judgment_when_only_one_has_numbers(self):
        """แม้คู่นี้เป็นของคนละอย่างชัดเจน กฎก็ต้องเงียบ — กิ่งนี้แยกสองกรณีนั้นไม่ออก"""
        assert (
            classify_pair("ถังปูน 15 สี SP", "ขวดซอส-กลาง")
            is PairVerdict.NEEDS_JUDGMENT
        )
