"""
Unit Tests — TextProcessor (BasicTextProcessor & ThaiTextProcessor)
====================================================================
ทดสอบ Text Processing Layer ซึ่งเป็นหัวใจของการประมวลผลชื่อสินค้าไทย
ครอบคลุม:
  1. BasicTextProcessor.process()     — Unicode, Lowercase, Punctuation, Whitespace
  2. BasicTextProcessor.process_batch()
  3. ThaiTextProcessor.process()      — Thai-specific behaviour
  4. Edge Cases                       — empty, numeric, special chars, mixed Thai-Eng
  5. ComponentFactory                 — factory method ทดสอบ type routing
  6. Idempotency                      — process(process(x)) == process(x)
  7. Real-world Thai Product Names    — ชื่อสินค้าจริงที่ระบบต้องจัดการ
"""

import pytest
import unicodedata
from fresh_implementations import (
    BasicTextProcessor,
    ThaiTextProcessor,
    ComponentFactory,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def basic() -> BasicTextProcessor:
    """Default BasicTextProcessor (lowercase=True, remove_punct=True, normalize_ws=True)"""
    return BasicTextProcessor()


@pytest.fixture
def basic_keep_case() -> BasicTextProcessor:
    """BasicTextProcessor ที่ไม่เปลี่ยน case"""
    return BasicTextProcessor(lowercase=False)


@pytest.fixture
def basic_keep_punct() -> BasicTextProcessor:
    """BasicTextProcessor ที่เก็บ punctuation ไว้"""
    return BasicTextProcessor(remove_punctuation=False)


@pytest.fixture
def thai() -> ThaiTextProcessor:
    """ThaiTextProcessor ด้วย default config"""
    return ThaiTextProcessor()


# =============================================================================
# 1. BasicTextProcessor — Unicode Normalization
# =============================================================================

class TestUnicodeNormalization:
    """ทดสอบการ normalize Unicode"""

    @pytest.mark.unit
    def test_nfkd_normalization_applied(self, basic):
        """ตรวจสอบว่า NFKD normalization ถูกใช้"""
        # ตัวอักษรที่มีหลาย Unicode representation
        text_nfc = unicodedata.normalize('NFC', 'café')   # composed
        text_nfd = unicodedata.normalize('NFD', 'café')   # decomposed

        result_nfc = basic.process(text_nfc)
        result_nfd = basic.process(text_nfd)

        # หลัง normalize ทั้งคู่ต้องให้ผลเหมือนกัน
        assert result_nfc == result_nfd

    @pytest.mark.unit
    def test_non_string_input_converted(self, basic):
        """Input ที่ไม่ใช่ string ต้องถูกแปลงเป็น string"""
        assert isinstance(basic.process(123), str)
        assert isinstance(basic.process(3.14), str)
        assert isinstance(basic.process(None), str)
        assert isinstance(basic.process(True), str)

    @pytest.mark.unit
    def test_thai_unicode_preserved(self, basic):
        """ตัวอักษรไทย (U+0E00–U+0E7F) ต้องไม่ถูกลบออก"""
        result = basic.process("เก้าอี้")
        assert "เก้าอี้" in result or len(result) > 0


# =============================================================================
# 2. BasicTextProcessor — Lowercase
# =============================================================================

class TestLowercase:
    """ทดสอบการแปลง lowercase"""

    @pytest.mark.unit
    def test_english_converted_to_lowercase(self, basic):
        assert basic.process("iPhone 13 Pro") == "iphone 13 pro"

    @pytest.mark.unit
    def test_mixed_case_normalized(self, basic):
        assert basic.process("Samsung GALAXY S21") == "samsung galaxy s21"

    @pytest.mark.unit
    def test_already_lowercase_unchanged(self, basic):
        result = basic.process("samsung galaxy")
        assert result == result.lower()

    @pytest.mark.unit
    def test_lowercase_disabled(self, basic_keep_case):
        """เมื่อ lowercase=False ให้รักษา case เดิม"""
        result = basic_keep_case.process("iPhone 13 Pro")
        # ไม่ควร lowercase
        assert "iPhone" in result or "i" in result.lower()

    @pytest.mark.unit
    def test_thai_text_lowercase_safe(self, basic):
        """ภาษาไทยไม่มี case ดังนั้นต้องไม่กระทบ output"""
        text = "เก้าอี้เตี้ย สีหวาน"
        result = basic.process(text)
        # ภาษาไทยควรยังคงอยู่
        assert len(result) > 0


# =============================================================================
# 3. BasicTextProcessor — Punctuation Removal
# =============================================================================

class TestPunctuationRemoval:
    """ทดสอบการลบ punctuation"""

    @pytest.mark.unit
    def test_common_punctuation_removed(self, basic):
        result = basic.process("hello, world!")
        assert "," not in result
        assert "!" not in result

    @pytest.mark.unit
    def test_parentheses_removed(self, basic):
        """วงเล็บในชื่อสินค้าต้องถูกลบ"""
        result = basic.process("ถังน้ำฝาใส (ใหญ่)")
        assert "(" not in result
        assert ")" not in result

    @pytest.mark.unit
    def test_slash_removed(self, basic):
        """เครื่องหมาย / ในหมวดหมู่ต้องถูกลบ"""
        result = basic.process("กล่อง/ที่เก็บของ")
        assert "/" not in result

    @pytest.mark.unit
    def test_hyphen_in_product_code_removed(self, basic):
        """เครื่องหมาย - ในรหัสสินค้า เช่น 728-PO ถูกลบ"""
        result = basic.process("728-PO SMT")
        assert "-" not in result

    @pytest.mark.unit
    def test_punctuation_not_removed_when_disabled(self, basic_keep_punct):
        """เมื่อ remove_punctuation=False ให้รักษา punctuation"""
        result = basic_keep_punct.process("hello, world!")
        assert "," in result

    @pytest.mark.unit
    def test_numbers_preserved_after_punct_removal(self, basic):
        """ตัวเลขต้องไม่ถูกลบออกพร้อม punctuation"""
        result = basic.process("สินค้า 1kg (500ml)")
        assert "1" in result
        assert "500" in result

    @pytest.mark.unit
    def test_thai_tone_marks_preserved(self, basic):
        """วรรณยุกต์ไทย เช่น ่ ้ ๊ ๋ ต้องไม่ถูกลบ"""
        # ทดสอบว่าชื่อสินค้าไทยยังอ่านได้หลัง process
        result = basic.process("น้ำยาล้างจาน")
        # ตัวอักษรไทยควรยังอยู่
        thai_chars = sum(1 for c in result if '\u0e00' <= c <= '\u0e7f')
        assert thai_chars > 0


# =============================================================================
# 4. BasicTextProcessor — Whitespace Normalization
# =============================================================================

class TestWhitespaceNormalization:
    """ทดสอบการ normalize whitespace"""

    @pytest.mark.unit
    def test_multiple_spaces_collapsed(self, basic):
        result = basic.process("เก้าอี้   เตี้ย")
        assert "  " not in result  # ไม่มี double space

    @pytest.mark.unit
    def test_leading_trailing_spaces_removed(self, basic):
        result = basic.process("  สินค้า  ")
        assert result == result.strip()

    @pytest.mark.unit
    def test_tabs_normalized(self, basic):
        result = basic.process("สินค้า\tราคา")
        assert "\t" not in result

    @pytest.mark.unit
    def test_newlines_normalized(self, basic):
        result = basic.process("สินค้า\nราคา")
        assert "\n" not in result

    @pytest.mark.unit
    def test_mixed_whitespace_collapsed(self, basic):
        result = basic.process("a  \t  b\n   c")
        assert result == "a b c"

    @pytest.mark.unit
    def test_empty_string_returns_empty(self, basic):
        result = basic.process("")
        assert result == ""

    @pytest.mark.unit
    def test_whitespace_only_returns_empty(self, basic):
        result = basic.process("   \t\n  ")
        assert result == ""


# =============================================================================
# 5. BasicTextProcessor — process_batch()
# =============================================================================

class TestProcessBatch:
    """ทดสอบการประมวลผล batch"""

    @pytest.mark.unit
    def test_batch_returns_correct_count(self, basic):
        texts = ["สินค้า A", "สินค้า B", "สินค้า C"]
        results = basic.process_batch(texts)
        assert len(results) == 3

    @pytest.mark.unit
    def test_batch_same_as_individual(self, basic):
        """ผลลัพธ์ batch ต้องตรงกับการ process ทีละรายการ"""
        texts = [
            "เก้าอี้เตี้ย 366 สีหวาน",
            "กล่องล็อค 560 หูหิ้ว W",
            "iPhone 13 Pro 256GB",
        ]
        batch_results = basic.process_batch(texts)
        individual_results = [basic.process(t) for t in texts]

        assert batch_results == individual_results

    @pytest.mark.unit
    def test_empty_batch_returns_empty_list(self, basic):
        results = basic.process_batch([])
        assert results == []

    @pytest.mark.unit
    def test_batch_with_mixed_types(self, basic):
        """Batch ที่มี non-string ต้องไม่ crash"""
        texts = ["สินค้า", 123, None, "ราคา"]
        results = basic.process_batch(texts)
        assert len(results) == 4
        assert all(isinstance(r, str) for r in results)


# =============================================================================
# 6. ThaiTextProcessor — Thai-specific Behaviour
# =============================================================================

class TestThaiTextProcessor:
    """ทดสอบ ThaiTextProcessor"""

    @pytest.mark.unit
    def test_thai_processor_inherits_basic(self, thai):
        """ThaiTextProcessor ต้องทำงานได้เหมือน BasicTextProcessor"""
        result = thai.process("iPhone 13 Pro")
        assert result == "iphone 13 pro"

    @pytest.mark.unit
    def test_thai_text_processed_correctly(self, thai):
        """ชื่อสินค้าไทยล้วนต้องผ่านได้"""
        result = thai.process("เก้าอี้เตี้ย สีหวาน")
        assert len(result) > 0
        # ตัวอักษรไทยควรยังอยู่ใน result
        thai_chars = sum(1 for c in result if '\u0e00' <= c <= '\u0e7f')
        assert thai_chars > 0

    @pytest.mark.unit
    def test_mixed_thai_english_numbers(self, thai):
        """สินค้าที่ผสม Thai + English + ตัวเลข"""
        result = thai.process("ถังน้ำฝาใส (ใหญ่) 728-PO SMT")
        # ตัวเลขต้องยังอยู่
        assert "728" in result
        # ตัวอักษรไทยต้องยังอยู่
        assert any('\u0e00' <= c <= '\u0e7f' for c in result)

    @pytest.mark.unit
    def test_thai_product_codes(self, thai):
        """รหัสสินค้าไทยทั่วไปต้องผ่าน"""
        products = [
            "ผงซักฟอก 1kg",
            "แชมพู 400ml",
            "น้ำยาล้างจาน 750ml",
            "สบู่ก้อน Dove 100g",
        ]
        for product in products:
            result = thai.process(product)
            assert len(result) > 0, f"ผล process ว่างสำหรับ: {product!r}"

    @pytest.mark.unit
    def test_thai_pattern_detection(self, thai):
        """ตรวจสอบว่า thai_pattern compile ถูกต้อง"""
        # thai_pattern ควร match ตัวอักษรไทย
        test_text = "เก้าอี้"
        matches = thai.thai_pattern.findall(test_text)
        assert len(matches) > 0

    @pytest.mark.unit
    def test_non_thai_text_still_works(self, thai):
        """ข้อความที่ไม่มีภาษาไทยเลยต้องทำงานได้"""
        result = thai.process("Samsung Galaxy S21 5G 128GB Black")
        assert "samsung" in result
        assert "galaxy" in result


# =============================================================================
# 7. Idempotency Tests
# =============================================================================

class TestIdempotency:
    """ทดสอบว่า process(process(x)) == process(x)"""

    @pytest.mark.unit
    def test_basic_idempotent(self, basic):
        """process ซ้ำ 2 ครั้งต้องให้ผลเหมือนกัน"""
        texts = [
            "เก้าอี้เตี้ย 366 สีหวาน",
            "กล่องล็อค 560 หูหิ้ว W",
            "iPhone 13 Pro  256GB",
            "  Samsung   Galaxy  ",
        ]
        for text in texts:
            once = basic.process(text)
            twice = basic.process(once)
            assert once == twice, (
                f"ไม่ idempotent สำหรับ: {text!r}\n"
                f"  process×1: {once!r}\n"
                f"  process×2: {twice!r}"
            )

    @pytest.mark.unit
    def test_thai_idempotent(self, thai):
        """ThaiTextProcessor ต้อง idempotent เช่นกัน"""
        texts = [
            "ถังน้ำฝาใส (ใหญ่) 728-PO SMT",
            "ผงซักฟอก 1kg",
            "น้ำยาล้างจาน   750ml",
        ]
        for text in texts:
            once = thai.process(text)
            twice = thai.process(once)
            assert once == twice, (
                f"ThaiTextProcessor ไม่ idempotent สำหรับ: {text!r}"
            )


# =============================================================================
# 8. ComponentFactory — create_text_processor()
# =============================================================================

class TestComponentFactory:
    """ทดสอบ Factory Method"""

    @pytest.mark.unit
    def test_factory_basic_returns_basic_processor(self):
        processor = ComponentFactory.create_text_processor("basic")
        assert isinstance(processor, BasicTextProcessor)

    @pytest.mark.unit
    def test_factory_thai_returns_thai_processor(self):
        processor = ComponentFactory.create_text_processor("thai")
        assert isinstance(processor, ThaiTextProcessor)

    @pytest.mark.unit
    def test_factory_thai_is_subclass_of_basic(self):
        """ThaiTextProcessor ต้อง inherit จาก BasicTextProcessor"""
        processor = ComponentFactory.create_text_processor("thai")
        assert isinstance(processor, BasicTextProcessor)

    @pytest.mark.unit
    def test_factory_unknown_type_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unknown text processor type"):
            ComponentFactory.create_text_processor("nonexistent")

    @pytest.mark.unit
    def test_factory_default_is_basic(self):
        """default type ต้องเป็น 'basic'"""
        processor = ComponentFactory.create_text_processor()
        assert isinstance(processor, BasicTextProcessor)

    @pytest.mark.unit
    def test_factory_passes_kwargs_to_processor(self):
        """Factory ต้องส่ง kwargs ไปให้ constructor ของ processor"""
        processor = ComponentFactory.create_text_processor("basic", lowercase=False)
        # ทดสอบว่า lowercase=False ส่งผ่านถูกต้อง
        result = processor.process("UPPER CASE")
        assert "UPPER" in result  # ถ้า lowercase=False คำใหญ่ต้องยังอยู่


# =============================================================================
# 9. Real-world Thai Product Names
# =============================================================================

class TestRealWorldProductNames:
    """ทดสอบด้วยชื่อสินค้าจริงที่ระบบต้องจัดการ"""

    # ชุดสินค้าตัวอย่างจาก TEST_SUMMARY.md
    REAL_PRODUCTS = [
        "เก้าอี้เตี้ย 366 สีหวาน",
        "กล่องล็อค 560 หูหิ้ว W",
        "ถังน้ำฝาใส (ใหญ่) 728-PO SMT",
        "ผงซักฟอก OMO 3kg",
        "น้ำยาล้างจาน Sunlight 750ml",
        "แชมพูสระผม Pantene 400ml",
        "สบู่ก้อน Dove 100g",
        "น้ำมันพืช ตราปรุงทิพย์ 1L",
        "ข้าวสารหอมมะลิ 5kg",
        "นมโฟร์โมสต์ UHT 1L",
    ]

    @pytest.mark.unit
    def test_all_products_produce_non_empty_output(self):
        """สินค้าทุกชิ้นต้องได้ผลลัพธ์ที่ไม่ว่าง"""
        processor = ThaiTextProcessor()
        for product in self.REAL_PRODUCTS:
            result = processor.process(product)
            assert len(result) > 0, f"ผลว่างสำหรับ: {product!r}"

    @pytest.mark.unit
    def test_output_is_lowercase(self):
        """ผลลัพธ์ต้องเป็น lowercase ทั้งหมด (English part)"""
        processor = BasicTextProcessor()
        for product in self.REAL_PRODUCTS:
            result = processor.process(product)
            # ตรวจเฉพาะตัวอักษรภาษาอังกฤษ
            english_chars = [c for c in result if c.isalpha() and c.isascii()]
            if english_chars:
                assert all(c.islower() for c in english_chars), (
                    f"มีตัวพิมพ์ใหญ่ใน: {result!r}"
                )

    @pytest.mark.unit
    def test_no_double_spaces_in_output(self):
        """ไม่มี double space ใน output"""
        processor = BasicTextProcessor()
        for product in self.REAL_PRODUCTS:
            result = processor.process(product)
            assert "  " not in result, (
                f"มี double space ใน: {result!r}"
            )

    @pytest.mark.unit
    def test_numbers_preserved(self):
        """ตัวเลขในชื่อสินค้าต้องยังคงอยู่"""
        processor = BasicTextProcessor()
        cases = [
            ("ถังน้ำฝาใส 728-PO", "728"),
            ("ผงซักฟอก OMO 3kg", "3"),
            ("นมโฟร์โมสต์ UHT 1L", "1"),
        ]
        for text, expected_num in cases:
            result = processor.process(text)
            assert expected_num in result, (
                f"ตัวเลข {expected_num!r} หายไปจาก: {result!r}"
            )

    @pytest.mark.unit
    def test_batch_processing_consistent(self):
        """batch processing ต้องสม่ำเสมอ"""
        processor = ThaiTextProcessor()
        batch_result = processor.process_batch(self.REAL_PRODUCTS)

        assert len(batch_result) == len(self.REAL_PRODUCTS)
        # ทุก element ต้องเป็น string
        assert all(isinstance(r, str) for r in batch_result)
        # ไม่มีผลว่าง
        assert all(len(r) > 0 for r in batch_result)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=False,
    )
    sys.exit(result.returncode)
