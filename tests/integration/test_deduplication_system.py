"""
Integration Tests — ProductDeduplicationSystem (Offline)
=========================================================
ย้าย logic ที่ทำงานได้ offline จาก root files มาเป็น proper pytest:
  - test_cleaned_text_simple.py  → TestTextCleaningOffline
  - test_cleaned_text_system.py  → TestDeduplicationPipeline

ทดสอบ:
  1. ระบบ Deduplication ทำงานได้ (mock embedding)
  2. product1_cleaned / product2_cleaned มีค่า (ข้อความถูก clean แล้ว)
  3. ThaiTextProcessor ทำงานถูกต้องผ่าน ProductDeduplicationSystem
  4. HumanFeedbackDatabase สร้างและ query ได้ (SQLite local)
  5. similarity_score อยู่ในช่วง 0–1
"""

import os
import pytest
from pathlib import Path

# ตรวจสอบว่า module ที่ต้องใช้มีอยู่ ทั้งหมดเป็น local deps
try:
    from human_feedback_system import (
        ProductDeduplicationSystem,
        HumanFeedbackDatabase,
        ProductComparison,
        FeedbackType,
    )
    from fresh_implementations import ComponentFactory
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    IMPORT_ERROR = str(e)

pytestmark = pytest.mark.integration

# Skip ทั้งไฟล์ถ้า import ไม่ได้
if not HAS_DEPS:
    pytest.skip(
        f"human_feedback_system ไม่พบ: {IMPORT_ERROR}",
        allow_module_level=True,
    )


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def dedup_system():
    """สร้าง ProductDeduplicationSystem ด้วย mock model (ไม่ต้องโหลด AI จริง)"""
    return ProductDeduplicationSystem(
        similarity_threshold=0.6,
        embedding_model_type="mock",
    )


@pytest.fixture
def similar_products():
    """ชุดสินค้าที่คาดว่าจะซ้ำกัน"""
    return [
        "iPhone 14 Pro Max 256GB สี-ดำ!!!",
        "iPhone 14 Pro Max 256GB Black",
        "iPhone 14 Pro Max 256 GB ดำ",
    ]


@pytest.fixture
def mixed_products():
    """ชุดสินค้าที่ผสม Thai/English"""
    return [
        "iPhone 14 Pro Max 256GB สี-ดำ!!!",
        "iPhone 14 Pro Max 256GB Black",
        "Samsung Galaxy S23 Ultra (256GB)",
        "Samsung Galaxy S23 Ultra - 256 GB",
        "MacBook Pro 14\" M2 Chip!!!",
        "MacBook Pro 14 inch M2",
    ]


@pytest.fixture
def tmp_db(tmp_path):
    """สร้าง SQLite DB ชั่วคราวสำหรับทดสอบ"""
    db_path = str(tmp_path / "test_dedup.db")
    db = HumanFeedbackDatabase(db_path)
    yield db
    # cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


# =============================================================================
# 1. ProductDeduplicationSystem — find_potential_duplicates()
# =============================================================================

class TestDeduplicationPipeline:
    """ทดสอบ Pipeline การหาสินค้าที่ซ้ำกัน"""

    @pytest.mark.integration
    def test_find_duplicates_returns_list(self, dedup_system, mixed_products):
        """find_potential_duplicates ต้องคืน list"""
        comparisons = dedup_system.find_potential_duplicates(mixed_products)
        assert isinstance(comparisons, list)

    @pytest.mark.integration
    def test_comparison_count_reasonable(self, dedup_system, mixed_products):
        """จำนวนคู่ที่เปรียบเทียบต้องไม่เกิน nC2"""
        n = len(mixed_products)
        max_pairs = n * (n - 1) // 2   # nC2
        comparisons = dedup_system.find_potential_duplicates(mixed_products)
        assert len(comparisons) <= max_pairs

    @pytest.mark.integration
    def test_comparison_has_required_fields(self, dedup_system, similar_products):
        """แต่ละ comparison ต้องมี field ที่จำเป็น"""
        comparisons = dedup_system.find_potential_duplicates(similar_products)
        if comparisons:
            comp = comparisons[0]
            assert hasattr(comp, "product1"),           "ขาด product1"
            assert hasattr(comp, "product2"),           "ขาด product2"
            assert hasattr(comp, "product1_cleaned"),   "ขาด product1_cleaned"
            assert hasattr(comp, "product2_cleaned"),   "ขาด product2_cleaned"
            assert hasattr(comp, "similarity_score"),   "ขาด similarity_score"

    @pytest.mark.integration
    def test_similarity_score_in_valid_range(self, dedup_system, mixed_products):
        """similarity_score ต้องอยู่ในช่วง 0.0–1.0"""
        comparisons = dedup_system.find_potential_duplicates(mixed_products)
        for comp in comparisons:
            assert 0.0 <= comp.similarity_score <= 1.0, (
                f"similarity_score ={comp.similarity_score:.4f} ไม่อยู่ใน [0,1]"
            )

    @pytest.mark.integration
    def test_product1_and_product2_are_different(self, dedup_system, mixed_products):
        """product1 ต้องไม่เหมือน product2 ในแต่ละคู่"""
        comparisons = dedup_system.find_potential_duplicates(mixed_products)
        for comp in comparisons:
            assert comp.product1 != comp.product2, (
                f"product1 และ product2 เหมือนกัน: {comp.product1!r}"
            )

    @pytest.mark.integration
    def test_empty_input_returns_empty(self, dedup_system):
        """input ว่างต้องได้ list ว่าง"""
        comparisons = dedup_system.find_potential_duplicates([])
        assert comparisons == []

    @pytest.mark.integration
    def test_single_product_returns_empty(self, dedup_system):
        """1 สินค้าไม่สามารถเปรียบเทียบได้ ต้องได้ list ว่าง"""
        comparisons = dedup_system.find_potential_duplicates(["เก้าอี้เตี้ย"])
        assert comparisons == []


# =============================================================================
# 2. Text Cleaning Integration
# =============================================================================

class TestTextCleaningIntegration:
    """ทดสอบว่าข้อความถูก clean ก่อน process จริง"""

    @pytest.mark.integration
    def test_cleaned_text_is_lowercase_english(self, dedup_system, similar_products):
        """ส่วนภาษาอังกฤษใน cleaned text ต้องเป็น lowercase"""
        comparisons = dedup_system.find_potential_duplicates(similar_products)
        for comp in comparisons:
            english_chars_1 = [c for c in comp.product1_cleaned if c.isalpha() and c.isascii()]
            english_chars_2 = [c for c in comp.product2_cleaned if c.isalpha() and c.isascii()]
            if english_chars_1:
                assert all(c.islower() for c in english_chars_1), (
                    f"product1_cleaned มีตัวพิมพ์ใหญ่: {comp.product1_cleaned!r}"
                )
            if english_chars_2:
                assert all(c.islower() for c in english_chars_2), (
                    f"product2_cleaned มีตัวพิมพ์ใหญ่: {comp.product2_cleaned!r}"
                )

    @pytest.mark.integration
    def test_cleaned_text_no_double_spaces(self, dedup_system, mixed_products):
        """cleaned text ไม่มี double space"""
        comparisons = dedup_system.find_potential_duplicates(mixed_products)
        for comp in comparisons:
            assert "  " not in comp.product1_cleaned, (
                f"product1_cleaned มี double space: {comp.product1_cleaned!r}"
            )
            assert "  " not in comp.product2_cleaned, (
                f"product2_cleaned มี double space: {comp.product2_cleaned!r}"
            )

    @pytest.mark.integration
    def test_cleaned_text_no_leading_trailing_spaces(self, dedup_system, similar_products):
        """cleaned text ไม่มี leading/trailing spaces"""
        comparisons = dedup_system.find_potential_duplicates(similar_products)
        for comp in comparisons:
            assert comp.product1_cleaned == comp.product1_cleaned.strip()
            assert comp.product2_cleaned == comp.product2_cleaned.strip()

    @pytest.mark.integration
    def test_cleaned_is_shorter_or_equal_to_original(self, dedup_system):
        """cleaned text ไม่ยาวกว่าหรือเท่ากับ original (punctuation ถูกลบออก)"""
        dirty_products = [
            "iPhone 14 Pro Max 256GB สี-ดำ!!!",
            "iPhone 14 Pro Max 256GB Black (NEW)",
        ]
        comparisons = dedup_system.find_potential_duplicates(dirty_products)
        for comp in comparisons:
            # หลัง clean ต้องไม่ยาวกว่า original
            assert len(comp.product1_cleaned) <= len(comp.product1), (
                f"cleaned ยาวกว่า original: {comp.product1_cleaned!r} > {comp.product1!r}"
            )

    @pytest.mark.integration
    def test_text_processor_standalone(self):
        """ThaiTextProcessor ทำงานถูกต้องโดยตรง"""
        processor = ComponentFactory.create_text_processor("thai")
        test_cases = [
            ("iPhone 14 Pro Max 256GB สี-ดำ!!!", "iphone"),
            ("Samsung Galaxy S23 Ultra (256GB)", "samsung"),
            ("   iPad   Air   5th   Generation   ", "ipad"),
            ("AirPods Pro 2nd Generation - White", "airpods"),
        ]
        for original, expected_substr in test_cases:
            result = processor.process(original)
            assert expected_substr in result, (
                f"ไม่พบ {expected_substr!r} ใน process({original!r}) = {result!r}"
            )


# =============================================================================
# 3. HumanFeedbackDatabase — SQLite Local
# =============================================================================

class TestHumanFeedbackDatabase:
    """ทดสอบ Database Layer (SQLite — ไม่ต้องการ external deps)"""

    @pytest.mark.integration
    def test_database_creates_successfully(self, tmp_db):
        """Database สร้างได้โดยไม่ error"""
        assert tmp_db is not None

    @pytest.mark.integration
    def test_get_pending_reviews_returns_list(self, tmp_db):
        """get_pending_reviews ต้องคืน list เสมอ (แม้จะว่าง)"""
        result = tmp_db.get_pending_reviews(10)
        assert isinstance(result, list)

    @pytest.mark.integration
    def test_new_database_has_no_pending_reviews(self, tmp_db):
        """Database ใหม่ที่ว่างเปล่าต้องไม่มี pending reviews"""
        result = tmp_db.get_pending_reviews(10)
        assert len(result) == 0

    @pytest.mark.integration
    def test_database_file_created_on_disk(self, tmp_path):
        """ไฟล์ database ต้องถูกสร้างบน disk จริง"""
        db_path = str(tmp_path / "verify_creation.db")
        HumanFeedbackDatabase(db_path)
        assert os.path.exists(db_path), f"ไม่พบไฟล์ database: {db_path}"


# =============================================================================
# 4. FeedbackType Enum
# =============================================================================

class TestFeedbackType:
    """ทดสอบ FeedbackType Enum"""

    @pytest.mark.integration
    def test_feedback_type_has_essential_values(self):
        """FeedbackType ต้องมี value ที่จำเป็นสำหรับ deduplication workflow"""
        types = [ft.value for ft in FeedbackType]
        # ต้องมีอย่างน้อย 1 value ที่บ่งบอกว่า "ซ้ำ" และ "ไม่ซ้ำ"
        assert len(types) >= 2, "FeedbackType ต้องมีอย่างน้อย 2 ค่า"
        assert all(isinstance(t, str) for t in types), "ทุก FeedbackType value ต้องเป็น string"

    @pytest.mark.integration
    def test_ml_prediction_is_feedback_type(self, dedup_system, similar_products):
        """ml_prediction ใน comparison ต้องเป็น FeedbackType"""
        comparisons = dedup_system.find_potential_duplicates(similar_products)
        for comp in comparisons:
            assert isinstance(comp.ml_prediction, FeedbackType), (
                f"ml_prediction ไม่ใช่ FeedbackType: {type(comp.ml_prediction)}"
            )
