"""Tests for the legacy labelled-product dataset loader.

ไฟล์ `input/รายการสินค้าพร้อมหมวดหมู่_AI.txt` คือข้อมูลสินค้าเก่าที่คนจัดหมวดไว้แล้ว
ใช้เป็น ground truth ทั้งสำหรับสกัด keyword และวัด accuracy จึงต้องโหลดให้ถูกต้องเป๊ะ
"""
import pytest

from src.utils.legacy_dataset import (
    LegacyProduct,
    load_legacy_products,
    stratified_split,
)


@pytest.fixture(scope="module")
def products():
    return load_legacy_products()


class TestLoading:
    def test_loads_expected_row_count(self, products):
        # ไฟล์มี 3,103 แถวที่มีทั้งหมวดหลักและหมวดย่อยครบ
        assert len(products) == 3103

    def test_returns_legacy_product_objects(self, products):
        assert all(isinstance(p, LegacyProduct) for p in products)

    def test_thai_text_decoded_correctly(self, products):
        """ไฟล์เป็น utf-16 ที่ข้างในเป็น cp874 ถ้าถอดผิดจะได้ตัวขยะ"""
        names = [p.name for p in products]
        assert any("ยาสีฟัน" in n for n in names)
        assert any("ไขควง" in n for n in names)
        # ตัวอักษรขยะจากการถอดรหัสผิดต้องไม่มี
        assert not any("Ã" in n or "à¸" in n for n in names)

    def test_every_row_has_name_and_categories(self, products):
        assert all(p.name and p.main_category and p.sub_category for p in products)

    def test_first_row_matches_source_file(self, products):
        first = products[0]
        assert first.sku == "HW001"
        assert "คราด" in first.name
        assert first.main_category == "เครื่องมือ_ฮาร์ดแวร์"

    def test_category_counts_match_taxonomy(self, products):
        assert len({p.main_category for p in products}) == 16
        # ไฟล์ดิบมี 116 รูปแบบ แต่ `ภาชนะใส่เครื่องปรุง/ขวดซอส` กับ
        # `ภาชนะใส่เครื่องปรุง / ขวดซอส` ปรากฏทั้งคู่ในไฟล์เดียวกัน (ข้อมูลไม่สม่ำเสมอ)
        # หลังรวมให้ตรงกับชื่อใน taxonomy_nodes จึงเหลือ 115
        assert len({p.sub_category for p in products}) == 115

    def test_spacing_variant_is_normalised(self, products):
        """`ภาชนะใส่เครื่องปรุง/ขวดซอส` ต่างจากใน DB แค่ช่องว่าง ต้องถูกรวมเป็นชื่อเดียว"""
        subs = {p.sub_category for p in products}
        assert "ภาชนะใส่เครื่องปรุง / ขวดซอส" in subs
        assert "ภาชนะใส่เครื่องปรุง/ขวดซอส" not in subs


class TestStratifiedSplit:
    def test_split_is_disjoint_and_complete(self, products):
        train, test = stratified_split(products)
        assert len(train) + len(test) == len(products)
        train_ids = {id(p) for p in train}
        assert not (train_ids & {id(p) for p in test})

    def test_test_set_is_roughly_the_requested_ratio(self, products):
        train, test = stratified_split(products, test_ratio=0.2)
        assert 0.15 <= len(test) / len(products) <= 0.25

    def test_split_is_deterministic(self, products):
        a_train, a_test = stratified_split(products, seed=42)
        b_train, b_test = stratified_split(products, seed=42)
        assert [p.sku for p in a_test] == [p.sku for p in b_test]
        assert [p.sku for p in a_train] == [p.sku for p in b_train]

    def test_different_seed_gives_different_split(self, products):
        _, test_a = stratified_split(products, seed=1)
        _, test_b = stratified_split(products, seed=2)
        assert [p.sku for p in test_a] != [p.sku for p in test_b]

    def test_every_test_category_is_represented_in_train(self, products):
        """กันไม่ให้มีหมวดที่โผล่แต่ใน test — จะวัด accuracy ไม่ยุติธรรม"""
        train, test = stratified_split(products)
        train_cats = {p.sub_category for p in train}
        assert {p.sub_category for p in test} <= train_cats

    def test_singleton_categories_go_to_train(self, products):
        """หมวดที่มีสินค้าเดียวต้องอยู่ใน train เท่านั้น ไม่งั้นเรียนไม่ได้"""
        from collections import Counter

        counts = Counter(p.sub_category for p in products)
        singletons = {c for c, n in counts.items() if n == 1}
        _, test = stratified_split(products)
        assert not ({p.sub_category for p in test} & singletons)

    def test_small_categories_still_appear_in_test(self, products):
        """หมวดที่มี 2-4 รายการต้องมีตัวแทนใน test ด้วย

        ถ้าใช้ int(n*0.2) ตรงๆ หมวดเล็กจะได้ 0 แล้วหายจากการวัด
        ทำให้ accuracy ดูดีเกินจริงเพราะเหลือแต่หมวดใหญ่ที่ทายง่ายกว่า
        """
        from collections import Counter

        counts = Counter(p.sub_category for p in products)
        small = {c for c, n in counts.items() if 2 <= n <= 4}
        _, test = stratified_split(products)
        assert small <= {p.sub_category for p in test}
