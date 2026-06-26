#!/usr/bin/env python3
"""
Scoring Logic
=============
Centralized logic for confidence levels, ML predictions, and result enhancement.
Ensures consistent classification of similarity matches across the system.
"""

import time
from typing import List, Dict, Any, Tuple


def classify_confidence(score: float) -> Tuple[str, str]:
    """
    Classify a similarity score into confidence level and ML prediction.
    """
    if score >= 0.95:
        return "very_high", "duplicate"
    elif score >= 0.85:
        return "high", "likely_duplicate"
    elif score >= 0.70:
        return "medium", "similar"
    else:
        return "low", "different"


def enhance_match_result(match: Dict[str, Any], rank: int = 1) -> Dict[str, Any]:
    """
    Enhance a single match result with metadata and confidence classification.
    """
    score = match.get('similarity_score', 0.0)
    level, prediction = classify_confidence(score)
    
    enhanced = match.copy()
    enhanced.update({
        "confidence_level": level,
        "ml_prediction": prediction,
        "rank": rank,
        "processor_version": "4.1.0",
        "processing_timestamp": time.time()
    })
    return enhanced


def enhance_matches(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enhance a list of match results.
    """
    return [enhance_match_result(m, i + 1) for i, m in enumerate(matches)]


def calculate_hybrid_score(keyword_score: float, embedding_score: float, 
                           keyword_weight: float = 0.6) -> float:
    """
    Calculate hybrid confidence score from keyword and embedding scores.
    """
    embedding_weight = 1.0 - keyword_weight
    return (keyword_score * keyword_weight) + (embedding_score * embedding_weight)
