"""
/ai-brain shows a tile called "average confidence". The status endpoint used to
answer with a literal 85.0 -- see the old comment "Fixed or from training
history" -- so the number on screen had nothing to do with the model.

These tests pin two things:
  1. training records the model's real mean confidence on its test split
  2. the status endpoint reports that value, and reports nothing at all when a
     model was trained before the field existed (rather than inventing one)
"""
import asyncio
import types
import uuid

import pytest

from src.api.routers import learn as learn_router
from src.services.human_feedback_system import FeedbackType, ProductComparison
from src.services.ml_feedback_learning import FeedbackLearningModel


def _comparison(name_a: str, name_b: str, similarity: float, verdict: FeedbackType) -> ProductComparison:
    return ProductComparison(
        id=str(uuid.uuid4()),
        product1=name_a,
        product2=name_b,
        product1_cleaned=name_a,
        product2_cleaned=name_b,
        similarity_score=similarity,
        confidence_score=similarity,
        ml_prediction=verdict,
        human_feedback=verdict,
    )


def _training_set():
    """คู่ที่เหมือนกันเกือบหมด vs คู่ที่ต่างกันชัดเจน อย่างละ 15 คู่"""
    duplicates = [
        _comparison(f"pen blue {i}", f"pen blue  {i}", 0.95, FeedbackType.DUPLICATE)
        for i in range(15)
    ]
    different = [
        _comparison(f"pen blue {i}", f"hammer steel {i}", 0.15, FeedbackType.DIFFERENT)
        for i in range(15)
    ]
    return duplicates + different


def _status(learning_system):
    """เรียก endpoint แบบ async จากเทสต์ปกติ (รีโปนี้ไม่มี pytest-asyncio)"""
    original = learn_router.get_ml_learning_system
    learn_router.get_ml_learning_system = lambda: learning_system
    try:
        return asyncio.run(learn_router.get_ml_status())
    finally:
        learn_router.get_ml_learning_system = original


def _fake_learning_system(training_record):
    model = types.SimpleNamespace(
        is_trained=True,
        model=types.SimpleNamespace(),
        training_history=[training_record],
    )
    return types.SimpleNamespace(model=model)


def test_training_records_real_average_confidence():
    model = FeedbackLearningModel()

    result = model.train_from_feedback(_training_set())

    assert "average_confidence" in result
    assert 0.0 <= result["average_confidence"] <= 1.0


def test_status_reports_the_trained_confidence():
    status = _status(_fake_learning_system({
        "test_accuracy": 0.9964,
        "total_samples": 1381,
        "average_confidence": 0.9123,
    }))

    assert status["average_confidence"] == pytest.approx(91.23)


def test_status_reports_nothing_when_the_model_predates_the_field():
    status = _status(_fake_learning_system({
        "test_accuracy": 0.9964,
        "total_samples": 1381,
    }))

    assert status["average_confidence"] is None
