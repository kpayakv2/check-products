"""Tests for the shared scoring helpers in `src/core/scoring_logic.py`.

`calculate_hybrid_score` คือที่เดียวที่สูตร Keyword 60% + Embedding 40% ถูกใช้จริง
(อีก 4 ที่ในรีโปเป็น SQL function ที่ไม่มีใครเรียก) แต่เดิมไม่มีเทสต์เลย
"""
import pytest

from src.core.scoring_logic import calculate_hybrid_score, classify_confidence


class TestCalculateHybridScore:
    def test_default_weighting_is_60_40(self):
        # keyword 1.0, embedding 0.0 -> ได้น้ำหนักฝั่ง keyword ล้วน
        assert calculate_hybrid_score(1.0, 0.0) == pytest.approx(0.6)
        assert calculate_hybrid_score(0.0, 1.0) == pytest.approx(0.4)

    def test_weights_always_sum_to_one(self):
        """คะแนนเต็มทั้งสองฝั่งต้องได้ 1.0 พอดี ไม่เกินไม่ขาด"""
        for weight in (0.0, 0.25, 0.5, 0.6, 0.75, 1.0):
            assert calculate_hybrid_score(1.0, 1.0, keyword_weight=weight) == pytest.approx(1.0)

    def test_zero_scores_give_zero(self):
        assert calculate_hybrid_score(0.0, 0.0) == pytest.approx(0.0)

    def test_blends_partial_scores(self):
        assert calculate_hybrid_score(0.5, 0.5) == pytest.approx(0.5)
        assert calculate_hybrid_score(0.8, 0.3) == pytest.approx(0.8 * 0.6 + 0.3 * 0.4)

    def test_custom_weight_shifts_the_balance(self):
        # ให้ฝั่ง embedding นำ
        assert calculate_hybrid_score(1.0, 0.0, keyword_weight=0.2) == pytest.approx(0.2)
        assert calculate_hybrid_score(0.0, 1.0, keyword_weight=0.2) == pytest.approx(0.8)

    def test_result_is_monotonic_in_each_input(self):
        """คะแนนฝั่งใดฝั่งหนึ่งสูงขึ้น ผลรวมต้องไม่ลดลง"""
        base = calculate_hybrid_score(0.4, 0.4)
        assert calculate_hybrid_score(0.5, 0.4) > base
        assert calculate_hybrid_score(0.4, 0.5) > base

    def test_keyword_side_outweighs_embedding_side(self):
        """ฝั่ง keyword ต้องมีอิทธิพลมากกว่าเมื่อคะแนนเท่ากัน — เป็นเจตนาของสูตร 60/40"""
        keyword_only = calculate_hybrid_score(0.9, 0.1)
        embedding_only = calculate_hybrid_score(0.1, 0.9)
        assert keyword_only > embedding_only


class TestClassifyConfidence:
    @pytest.mark.parametrize(
        "score,level,prediction",
        [
            (1.0, "very_high", "duplicate"),
            (0.95, "very_high", "duplicate"),
            (0.94, "high", "likely_duplicate"),
            (0.85, "high", "likely_duplicate"),
            (0.84, "medium", "similar"),
            (0.70, "medium", "similar"),
            (0.69, "low", "different"),
            (0.0, "low", "different"),
        ],
    )
    def test_thresholds_are_inclusive_at_the_boundary(self, score, level, prediction):
        assert classify_confidence(score) == (level, prediction)
