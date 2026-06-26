import pytest
from src.core.fresh_implementations import ThaiTextProcessor
from src.services.taxonomy_service import TaxonomyService
import sys
import os

# ม็อก Supabase สำหรับทดสอบ logic
class MockSupabase:
    def table(self, name):
        return self
    def select(self, columns):
        return self
    def execute(self):
        return self

@pytest.fixture
def taxonomy_service():
    mock_supabase = MockSupabase()
    return TaxonomyService(mock_supabase)

def test_extract_auto_keywords_basic(taxonomy_service):
    """ทดสอบการสกัดคำพื้นฐาน"""
    # มาม่า รสต้มยำกุ้ง 60ก.
    product_name = "\u0e21\u0e32\u0e21\u0e48\u0e32 \u0e23\u0e2a\u0e15\u0e49\u0e21\u0e22\u0e33\u0e01\u0e38\u0e49\u0e07 60\u0e01."
    keywords = taxonomy_service.extract_auto_keywords(product_name)
    
    mama = "\u0e21\u0e32\u0e21\u0e48\u0e32"
    tomyum = "\u0e15\u0e49\u0e21\u0e22\u0e33\u0e01\u0e38\u0e49\u0e07"
    
    assert any(mama in k for k in keywords)
    assert any(tomyum in k for k in keywords)
    assert "60" not in "".join(keywords)

def test_extract_auto_keywords_with_stop_words(taxonomy_service):
    """ทดสอบการตัดคำขยะไทย (Stopwords)"""
    # น้ำดื่ม ตราสิงห์ ขนาด 600มล. ราคาพิเศษ
    product_name = "\u0e19\u0e49\u0e33\u0e14\u0e37\u0e48\u0e21 \u0e15\u0e23\u0e32\u0e2a\u0e34\u0e07\u0e2b\u0e4c \u0e02\u0e19\u0e32\u0e14 600\u0e21\u0e25. \u0e23\u0e32\u0e04\u0e32\u0e1e\u0e34\u0e40\u0e28\u0e29"
    keywords = taxonomy_service.extract_auto_keywords(product_name)
    
    nam_duem = "\u0e19\u0e49\u0e33\u0e14\u0e37\u0e48\u0e21"
    singha = "\u0e2a\u0e34\u0e07\u0e2b\u0e4c"
    
    assert any(nam_duem in k for k in keywords)
    assert any(singha in k for k in keywords)
    assert "ขนาด" not in keywords
    assert "ราคาพิเศษ" not in keywords

def test_extract_auto_keywords_mixed_lang(taxonomy_service):
    """ทดสอบชื่อสินค้าไทยปนอังกฤษ"""
    product_name = "Breeze Excel ผงซักฟอก บรีส เอกเซล 1500g"
    keywords = taxonomy_service.extract_auto_keywords(product_name)
    assert any("breeze" in k for k in keywords)
    assert any("excel" in k for k in keywords)
    assert any("ผงซักฟอก" in k for k in keywords)
    assert any("บรีส" in k for k in keywords)
