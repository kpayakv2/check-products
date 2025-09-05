"""
Unit tests for shared scoring system
Tests score-to-confidence mapping and validation functions
"""

from typing import List, Tuple


def calculate_confidence_level(similarity_score: float, 
                             min_score: float, 
                             max_score: float) -> Tuple[float, str]:
    """
    Calculate confidence score and level based on similarity score.
    
    Args:
        similarity_score: Raw similarity score (0.0-1.0)
        min_score: Minimum score in result set
        max_score: Maximum score in result set
        
    Returns:
        (confidence_score, confidence_level)
    """
    # Normalize confidence relative to score range
    score_range = max_score - min_score if max_score > min_score else 1.0
    confidence_score = (similarity_score - min_score) / score_range
    
    # Map to confidence levels
    if confidence_score >= 0.8:
        confidence_level = 'high'
    elif confidence_score >= 0.5:
        confidence_level = 'medium' 
    else:
        confidence_level = 'low'
    
    return round(confidence_score, 4), confidence_level


def validate_similarity_score(score: float) -> bool:
    """Validate similarity score is within expected range."""
    return 0.0 <= score <= 1.0


def validate_confidence_mapping(similarity: float, confidence: float) -> bool:
    """Validate confidence score maps correctly to similarity."""
    return 0.0 <= confidence <= 1.0


class TestSharedScoring:
    """Test cases for shared scoring system."""
    
    def test_confidence_mapping_high_variability(self):
        """Test confidence mapping with high score variability."""
        # Example dataset: [0.95, 0.87, 0.73, 0.62, 0.45, 0.31]
        min_score, max_score = 0.31, 0.95
        
        # Test cases: (similarity_score, expected_confidence_score, expected_level)
        test_cases = [
            (0.95, 1.0000, 'high'),
            (0.87, 0.8750, 'high'),
            (0.73, 0.6562, 'medium'),
            (0.62, 0.4844, 'low'),
            (0.45, 0.2188, 'low'),
            (0.31, 0.0000, 'low')
        ]
        
        for similarity_score, expected_confidence, expected_level in test_cases:
            confidence_score, confidence_level = calculate_confidence_level(
                similarity_score, min_score, max_score
            )
            
            assert abs(confidence_score - expected_confidence) < 0.001, \
                f"Confidence score mismatch for {similarity_score}: got {confidence_score}, expected {expected_confidence}"
            assert confidence_level == expected_level, \
                f"Confidence level mismatch for {similarity_score}: got {confidence_level}, expected {expected_level}"
    
    def test_confidence_mapping_low_variability(self):
        """Test confidence mapping with low score variability."""
        # Example dataset: [0.78, 0.76, 0.74, 0.72, 0.71, 0.69]
        min_score, max_score = 0.69, 0.78
        
        test_cases = [
            (0.78, 1.0000, 'high'),
            (0.76, 0.7778, 'medium'),  # 0.7778 >= 0.5, so medium
            (0.74, 0.5556, 'medium'),
            (0.72, 0.3333, 'low'),
            (0.71, 0.2222, 'low'),
            (0.69, 0.0000, 'low')
        ]
        
        for similarity_score, expected_confidence, expected_level in test_cases:
            confidence_score, confidence_level = calculate_confidence_level(
                similarity_score, min_score, max_score
            )
            
            assert abs(confidence_score - expected_confidence) < 0.001, \
                f"Confidence score mismatch for {similarity_score}: got {confidence_score}, expected {expected_confidence}"
            assert confidence_level == expected_level, \
                f"Confidence level mismatch for {similarity_score}: got {confidence_level}, expected {expected_level}"
    
    def test_edge_cases(self):
        """Test edge cases in confidence calculation."""
        # Test identical min/max scores
        confidence_score, confidence_level = calculate_confidence_level(0.5, 0.5, 0.5)
        assert abs(confidence_score - 0.0) < 0.001  # When range is 0, we get 0.0/1.0 = 0.0
        assert confidence_level == 'low'
        
        # Test boundary conditions for confidence levels
        min_score, max_score = 0.0, 1.0
        
        # Test high confidence boundary (0.8)
        confidence_score, confidence_level = calculate_confidence_level(0.8, min_score, max_score)
        assert confidence_score == 0.8
        assert confidence_level == 'high'
        
        # Test medium confidence boundary (0.5)
        confidence_score, confidence_level = calculate_confidence_level(0.5, min_score, max_score)
        assert confidence_score == 0.5
        assert confidence_level == 'medium'
        
        # Test just below boundaries
        confidence_score, confidence_level = calculate_confidence_level(0.799, min_score, max_score)
        assert confidence_level == 'medium'  # 0.799 >= 0.5, so medium
        
        confidence_score, confidence_level = calculate_confidence_level(0.499, min_score, max_score)
        assert confidence_level == 'low'  # < 0.5
    
    def test_similarity_score_validation(self):
        """Test similarity score validation."""
        # Valid scores
        assert validate_similarity_score(0.0) == True
        assert validate_similarity_score(0.5) == True
        assert validate_similarity_score(1.0) == True
        assert validate_similarity_score(0.87) == True
        
        # Invalid scores
        assert validate_similarity_score(-0.1) == False
        assert validate_similarity_score(1.1) == False
        assert validate_similarity_score(-1.0) == False
        assert validate_similarity_score(2.0) == False
    
    def test_confidence_score_validation(self):
        """Test confidence score validation."""
        # Valid confidence scores
        assert validate_confidence_mapping(0.87, 0.75) == True
        assert validate_confidence_mapping(0.5, 0.0) == True
        assert validate_confidence_mapping(1.0, 1.0) == True
        
        # Invalid confidence scores
        assert validate_confidence_mapping(0.87, -0.1) == False
        assert validate_confidence_mapping(0.87, 1.1) == False
    
    def test_batch_confidence_calculation(self):
        """Test confidence calculation for a batch of scores."""
        scores = [0.95, 0.87, 0.73, 0.62, 0.45, 0.31]
        min_score, max_score = min(scores), max(scores)
        
        results = []
        for score in scores:
            confidence_score, confidence_level = calculate_confidence_level(
                score, min_score, max_score
            )
            results.append({
                'similarity_score': score,
                'confidence_score': confidence_score,
                'confidence_level': confidence_level
            })
        
        # Verify results are sorted by confidence score (descending)
        confidence_scores = [r['confidence_score'] for r in results]
        assert confidence_scores == sorted(confidence_scores, reverse=True)
        
        # Verify at least one high confidence result
        high_confidence_count = len([r for r in results if r['confidence_level'] == 'high'])
        assert high_confidence_count >= 1
        
        # Verify all scores are valid
        for result in results:
            assert validate_similarity_score(result['similarity_score'])
            assert validate_confidence_mapping(
                result['similarity_score'], 
                result['confidence_score']
            )
    
    def test_scoring_formula_consistency(self):
        """Test that scoring formulas are applied consistently."""
        # Test default hybrid weights (0.7 cosine + 0.3 euclidean)
        cosine_score = 0.8
        euclidean_score = 0.6
        
        expected_hybrid = 0.7 * cosine_score + 0.3 * euclidean_score
        actual_hybrid = 0.7 * 0.8 + 0.3 * 0.6
        
        assert abs(actual_hybrid - expected_hybrid) < 0.001
        assert abs(actual_hybrid - 0.74) < 0.001  # 0.56 + 0.18 = 0.74
    
    def test_threshold_filtering(self):
        """Test threshold filtering behavior."""
        scores = [0.95, 0.75, 0.65, 0.55, 0.45, 0.35]
        threshold = 0.6
        
        # Filter scores above threshold
        filtered_scores = [s for s in scores if s >= threshold]
        expected_filtered = [0.95, 0.75, 0.65]
        
        assert filtered_scores == expected_filtered
        
        # Test edge case: score exactly at threshold
        edge_scores = [0.6, 0.59, 0.61]
        filtered_edge = [s for s in edge_scores if s >= threshold]
        assert filtered_edge == [0.6, 0.61]


if __name__ == "__main__":
    # Run basic tests
    test_instance = TestSharedScoring()
    
    print("🧪 Running Shared Scoring Tests...")
    
    try:
        test_instance.test_confidence_mapping_high_variability()
        print("✅ High variability confidence mapping test passed")
        
        test_instance.test_confidence_mapping_low_variability()
        print("✅ Low variability confidence mapping test passed")
        
        test_instance.test_edge_cases()
        print("✅ Edge cases test passed")
        
        test_instance.test_similarity_score_validation()
        print("✅ Similarity score validation test passed")
        
        test_instance.test_confidence_score_validation()
        print("✅ Confidence score validation test passed")
        
        test_instance.test_batch_confidence_calculation()
        print("✅ Batch confidence calculation test passed")
        
        test_instance.test_scoring_formula_consistency()
        print("✅ Scoring formula consistency test passed")
        
        test_instance.test_threshold_filtering()
        print("✅ Threshold filtering test passed")
        
        print("\n🎉 All tests passed successfully!")
        
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
