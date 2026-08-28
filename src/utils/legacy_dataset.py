"""โหลดข้อมูลสินค้าเก่าที่คนจัดหมวดหมู่ไว้แล้ว (ground truth ของระบบ)

ไฟล์ `input/รายการสินค้าพร้อมหมวดหมู่_AI.txt` เป็น tab-separated ที่เข้ารหัสซ้อนกันสองชั้น:
บันทึกเป็น UTF-16 แต่ข้อความข้างในเป็นไบต์ cp874 (TIS-620) จึงต้องถอดสองรอบ
ถอดผิดจะได้ตัวอักษรขยะแบบเงียบๆ ไม่ error

ใช้ร่วมกันระหว่างการวัด accuracy, การสกัด keyword และการตรวจซ้ำ
เพื่อให้ทุกที่เห็นข้อมูลชุดเดียวกันและแบ่ง train/test แบบเดิมทุกครั้ง
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parents[2]
LEGACY_FILE = BASE_DIR / "input" / "รายการสินค้าพร้อมหมวดหมู่_AI.txt"

# คอลัมน์ในไฟล์ (0-indexed)
_COL_SKU = 1
_COL_NAME = 2
_COL_CLEAN_NAME = 3
_COL_MAIN_CATEGORY = 8
_COL_SUB_CATEGORY = 9

# หมวดย่อยที่ต่างจากชื่อใน taxonomy_nodes แค่ช่องว่าง — รวมให้เป็นชื่อเดียวกับใน DB
# ไม่งั้นจะกลายเป็นสองหมวดที่ map ไม่ตรงกัน
SUB_CATEGORY_ALIASES: Dict[str, str] = {
    "ภาชนะใส่เครื่องปรุง/ขวดซอส": "ภาชนะใส่เครื่องปรุง / ขวดซอส",
}


@dataclass(frozen=True)
class LegacyProduct:
    """สินค้าเก่าหนึ่งรายการพร้อมหมวดหมู่ที่คนจัดไว้"""

    sku: str
    name: str
    clean_name: str
    main_category: str
    sub_category: str


def _decode_cell(value: str) -> str:
    """แปลงข้อความที่ถูกอ่านเป็น latin1 กลับเป็นภาษาไทย (cp874)"""
    try:
        return value.encode("latin1").decode("cp874")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def load_legacy_products(path: Optional[Path] = None) -> List[LegacyProduct]:
    """อ่านไฟล์ข้อมูลเก่า คืนเฉพาะแถวที่มีชื่อสินค้าและหมวดหมู่ครบ"""
    source = Path(path) if path else LEGACY_FILE
    if not source.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ข้อมูลสินค้าเก่า: {source}")

    lines = source.read_bytes().decode("utf-16").splitlines()

    products: List[LegacyProduct] = []
    for line in lines[1:]:  # ข้ามหัวตาราง
        fields = _decode_cell(line).split("\t")
        if len(fields) <= _COL_SUB_CATEGORY:
            continue

        name = fields[_COL_NAME].strip()
        main_category = fields[_COL_MAIN_CATEGORY].strip()
        sub_category = fields[_COL_SUB_CATEGORY].strip()
        if not (name and main_category and sub_category):
            continue

        products.append(
            LegacyProduct(
                sku=fields[_COL_SKU].strip(),
                name=name,
                clean_name=fields[_COL_CLEAN_NAME].strip() or name,
                main_category=main_category,
                sub_category=SUB_CATEGORY_ALIASES.get(sub_category, sub_category),
            )
        )

    return products


def stratified_split(
    products: Sequence[LegacyProduct],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[LegacyProduct], List[LegacyProduct]]:
    """แบ่ง train/test แบบ stratified ตามหมวดย่อย

    หมวดที่มีสินค้าน้อยกว่า 2 รายการจะอยู่ใน train ทั้งหมด เพราะถ้าดึงไปไว้ test
    จะกลายเป็นหมวดที่ไม่เคยเห็นตอนเรียน ทำให้วัด accuracy ไม่ยุติธรรม
    """
    if not 0 < test_ratio < 1:
        raise ValueError(f"test_ratio ต้องอยู่ระหว่าง 0 ถึง 1 (ได้ {test_ratio})")

    by_category: Dict[str, List[LegacyProduct]] = defaultdict(list)
    for product in products:
        by_category[product.sub_category].append(product)

    rng = random.Random(seed)
    train: List[LegacyProduct] = []
    test: List[LegacyProduct] = []

    for category in sorted(by_category):  # เรียงชื่อหมวดก่อน เพื่อให้ผลคงที่ทุกครั้ง
        group = sorted(by_category[category], key=lambda p: (p.sku, p.name))
        rng.shuffle(group)

        # หมวดที่มี 2-4 รายการต้องมีตัวแทนใน test อย่างน้อย 1 ไม่งั้น int(n*0.2) จะได้ 0
        # แล้วหมวดเล็กจะหายจากการวัดทั้งหมด ทำให้ accuracy ดูดีเกินจริง
        n_test = max(1, int(len(group) * test_ratio))
        if len(group) < 2:
            n_test = 0

        test.extend(group[:n_test])
        train.extend(group[n_test:])

    return train, test
