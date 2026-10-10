"""ตัดสินว่าคู่สินค้าเป็น SKU เดียวกันหรือไม่ โดยอ่านตัวเลขในชื่อ

วางไว้หน้า pipeline ML เพราะ cosine similarity ของ embedding แยก SKU ไม่ออก —
ที่คะแนน 0.993-0.998 คู่ที่ได้เป็นคนละ SKU ทุกคู่ (`หม้อหู - จระเข้ #22` ↔ `#28`)
เหตุผลเต็มและทางเลือกที่ปฏิเสธไป: docs/adr/0002-numeric-veto-before-ml-matching.md

กฎนี้พูดได้สองอย่างเท่านั้น — "คนละสินค้า" และ "สินค้าซ้ำ" เมื่อมันไม่รู้ต้องเงียบ
ผลของกฎห้ามเขียนลงช่อง `reviewed` คำนั้นสงวนให้คนเท่านั้น (ดู CONTEXT.md)

กรณีที่มีเลขฝั่งเดียวเคยถูกวีโต้ว่าคนละสินค้าเมื่อคำที่เหลือต่างกันด้วย แต่วัดกับผลที่คนตรวจ
15 คู่ได้แค่ 12/15 (80%) ต่ำกว่าเกณฑ์ 95% เพราะความต่างนั้นมักเป็นรหัสรุ่นหรือรหัสผู้ผลิต
ที่อีกฝั่งไม่ได้บันทึก (`ขวดซอสใหญ่ (สีหวาน) 7026 BG` กับ `ขวดซอสใหญ่(สีหวาน)` คือของชิ้นเดียวกัน)
กิ่งนั้นจึงเงียบทั้งหมด ผลที่คนตรวจอยู่ใน tests/fixtures/dedup_human_labels_v1.json
"""
from __future__ import annotations

import re
from enum import Enum

# ทศนิยมต้องเป็นก้อนเดียว ไม่งั้น `3.5` กับ `5.3` จะกลายเป็นชุดเลขเดียวกัน
_NUMBER = re.compile(r"\d+(?:\.\d+)?")

# ราคาที่เขียนเป็นหมายเหตุ เช่น `(ราคา 25 บาท)` ไม่ใช่ตัวระบุสินค้า
_PRICE = re.compile(r"\d+(?:\.\d+)?\s*(?:บาท|บ\.)")

# เครื่องหมายที่คนคีย์ไม่ตรงกันระหว่างสองแถว แต่ไม่เปลี่ยนว่าเป็นสินค้าอะไร
_PUNCTUATION = re.compile(r'[ \-#/()*".,+]')


class PairVerdict(Enum):
    DIFFERENT = "คนละสินค้า"
    DUPLICATE = "สินค้าซ้ำ"
    NEEDS_JUDGMENT = "ต้องใช้วิจารณญาณ"


def _numbers(name: str) -> tuple[str, ...]:
    return tuple(sorted(_NUMBER.findall(name)))


def _collapse(name: str) -> str:
    return _PUNCTUATION.sub("", name.lower())


def _differs_only_by_price_note(name_a: str, name_b: str) -> bool:
    """จริงเมื่อเลขที่ต่างกันคือหมายเหตุราคาที่เขียนคำว่า "ราคา" กำกับไว้

    ถ้าไม่มีคำว่า "ราคา" ตัวเลขท้ายชื่อแบบ `หวี... 20 บ.` คือตัวระบุรายการคละราคา
    ซึ่งเป็นคนละรายการสตอกกับช่วงราคาอื่น
    """
    if "ราคา" not in name_a or "ราคา" not in name_b:
        return False
    return _numbers(_PRICE.sub("", name_a)) == _numbers(_PRICE.sub("", name_b))


def classify_pair(name_a: str, name_b: str) -> PairVerdict:
    numbers_a, numbers_b = _numbers(name_a), _numbers(name_b)

    if numbers_a and numbers_b:
        if numbers_a != numbers_b:
            if _differs_only_by_price_note(name_a, name_b):
                return PairVerdict.NEEDS_JUDGMENT
            return PairVerdict.DIFFERENT
        if _collapse(name_a) == _collapse(name_b):
            return PairVerdict.DUPLICATE
        return PairVerdict.NEEDS_JUDGMENT

    # เลขฝั่งเดียว หรือไม่มีเลขทั้งสองฝั่ง — กฎแยกไม่ออกว่าเป็นคนละสินค้าหรือของเดียวกัน
    return PairVerdict.NEEDS_JUDGMENT
