"""Tests for Thai word segmentation used by keyword matching.

เดิมระบบเทียบ keyword ด้วย substring (`kw in product_name`) ซึ่งทำให้
"สี" ไป match กลางคำ "ยาสีฟัน" แล้วจัดยาสีฟันเข้าหมวดสีทาบ้าน
การตัดคำที่ถูกต้องคือหัวใจของการแก้ปัญหานี้
"""
import pytest

from src.core.fresh_implementations import (
    merge_short_token_runs,
    tokenize_thai,
    tokens_contain_phrase,
)


class TestMergeShortTokenRuns:
    """พจนานุกรมไม่รู้จักชื่อแบรนด์จึงตัดแตกเป็นเศษ ต้องรวมกลับให้ได้ชื่อแบรนด์"""

    def test_merges_a_run_of_short_fragments(self):
        assert merge_short_token_runs(["บ", "รี", "ส"]) == ["บรีส"]

    def test_attaches_leading_fragment_to_the_following_word(self):
        """`โอ|โม่` ต้องได้ `โอโม่` ไม่ใช่ทิ้ง `โอ` แล้วเหลือแค่ `โม่`"""
        assert merge_short_token_runs(["โอ", "โม่"]) == ["โอโม่"]

    def test_leaves_normal_words_untouched(self):
        assert merge_short_token_runs(["ยาสีฟัน", "ดอกบัว"]) == ["ยาสีฟัน", "ดอกบัว"]

    def test_handles_trailing_fragments(self):
        assert merge_short_token_runs(["เอก", "เซ", "ล"]) == ["เอก", "เซล"]

    def test_empty_input(self):
        assert merge_short_token_runs([]) == []


class TestTokenizeThai:
    def test_splits_thai_without_spaces(self):
        assert tokenize_thai("ยาสีฟันดอกบัวคู่") == ["ยาสีฟัน", "ดอกบัว", "คู่"]

    def test_keeps_compound_word_intact(self):
        """`ยาสีฟัน` ต้องเป็นคำเดียว ไม่ถูกแยกเป็น ยา/สี/ฟัน"""
        assert "ยาสีฟัน" in tokenize_thai("ยาสีฟันคอลเกต")
        assert "สี" not in tokenize_thai("ยาสีฟันคอลเกต")

    def test_drops_whitespace_tokens(self):
        assert " " not in tokenize_thai("ไขควงแฉก 6 นิ้ว")
        assert "" not in tokenize_thai("ไขควงแฉก 6 นิ้ว")

    def test_keeps_numbers_and_latin(self):
        tokens = tokenize_thai("ผงซักฟอก 900 กรัม OMO")
        assert "900" in tokens
        assert "omo" in tokens or "OMO" in tokens

    def test_empty_input_returns_empty_list(self):
        assert tokenize_thai("") == []
        assert tokenize_thai("   ") == []

    def test_non_string_input_is_safe(self):
        assert tokenize_thai(None) == []

    def test_is_lowercased_for_matching(self):
        assert tokenize_thai("Colgate") == ["colgate"]


class TestTokensContainPhrase:
    def test_matches_single_word(self):
        tokens = tokenize_thai("ไขควงแฉก 6 นิ้ว")
        assert tokens_contain_phrase(tokens, "ไขควง")

    def test_rejects_substring_inside_a_word(self):
        """หัวใจของการแก้บั๊ก — "สี" ต้องไม่ match ใน "ยาสีฟัน" """
        tokens = tokenize_thai("ยาสีฟันดอกบัวคู่ 150g")
        assert not tokens_contain_phrase(tokens, "สี")

    def test_matches_multi_word_phrase_in_order(self):
        tokens = tokenize_thai("น้ำยาล้างจานซันไลต์")
        assert tokens_contain_phrase(tokens, "น้ำยาล้างจาน")

    def test_rejects_phrase_whose_words_are_out_of_order(self):
        tokens = ["ฟัน", "ยา"]
        assert not tokens_contain_phrase(tokens, "ยา ฟัน")

    def test_empty_phrase_never_matches(self):
        assert not tokens_contain_phrase(["ไขควง"], "")

    def test_matching_is_case_insensitive(self):
        assert tokens_contain_phrase(tokenize_thai("OMO 900g"), "omo")


class TestRegressionRealCases:
    """เคสจริงที่เคยจัดผิด — วัดจากข้อมูลเก่า 3,103 รายการ"""

    @pytest.mark.parametrize(
        "product,wrong_keyword",
        [
            ("ยาสีฟันดอกบัวคู่150g ออริจิ", "สี"),
            ("ยาสีฟันคอลเกต 150 กรัม", "สี"),
            ("น้ำยาปรับผ้านุ่มดาวน์นี่", "น้ำ"),
        ],
    )
    def test_short_keyword_no_longer_matches_mid_word(self, product, wrong_keyword):
        assert not tokens_contain_phrase(tokenize_thai(product), wrong_keyword)
