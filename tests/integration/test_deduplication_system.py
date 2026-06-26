"""
Integration Tests — ProductDeduplicationSystem (Refactored)
=========================================================

ทดสอบประเภทข้อมูล Enums และ Dataclasses สำหรับ Deduplication
- ตรวจสอบความสมบูรณ์ของโมเดลข้อมูลหลักที่ API/ML ยังนำไปใช้งานร่วมกัน
- ลบโค้ดทดสอบ SQLite และ ProductDeduplicationSystem เก่าออกเพื่อความสะอาด
"""

import pytest
from src.services.human_feedback_system import (
    ProductComparison,
    FeedbackType,
    UniqueProduct
)

pytestmark = pytest.mark.integration

class TestFeedbackType:
    """ทดสอบ FeedbackType Enum"""

    @pytest.mark.integration
    def test_feedback_type_has_essential_values(self):
        """FeedbackType ต้องมี value ที่จำเป็นสำหรับ deduplication workflow"""
        types = [ft.value for ft in FeedbackType]
        # ต้องมีอย่างน้อย 1 value ที่บ่งบอกว่า "ซ้ำ" และ "ไม่ซ้ำ"
        assert len(types) >= 4
        assert "duplicate" in types
        assert "different" in types
        assert "similar" in types
        assert "uncertain" in types


class TestProductModels:
    """ทดสอบ Dataclasses ของสินค้าซ้ำ"""

    @pytest.mark.integration
    def test_product_comparison_creation(self):
        """ตรวจสอบความถูกต้องในการสร้าง ProductComparison"""
        comp = ProductComparison(
            id="test_id",
            product1="iPhone 14 สีดำ",
            product2="iPhone 14 Black",
            product1_cleaned="iphone 14 สีดำ",
            product2_cleaned="iphone 14 black",
            similarity_score=0.92,
            confidence_score=0.8,
            ml_prediction=FeedbackType.SIMILAR
        )
        assert comp.id == "test_id"
        assert comp.similarity_score == 0.92
        assert comp.ml_prediction == FeedbackType.SIMILAR
