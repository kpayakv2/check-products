#!/usr/bin/env python3
"""
Test script for fresh_architecture and fresh_implementations modules
"""

import numpy as np
from fresh_implementations import (
    ComponentFactory, 
    MockEmbeddingModel,
    TFIDFEmbeddingModel,
    CosineSimilarityCalculator,
    ThaiTextProcessor
)
from fresh_architecture import ProductMatcher, Config

def test_basic_functionality():
    """Test basic functionality of all components."""
    print("🧪 Testing Basic Functionality...")
    
    # Test embedding models
    mock_model = MockEmbeddingModel(dimension=10)
    tfidf_model = TFIDFEmbeddingModel(max_features=100, dimension=10)
    
    # Test data
    texts = ["เสื้อยืดสีขาว", "กางเกงยีนส์", "รองเท้าผ้าใบ"]
    
    # Test mock model
    mock_embeddings = mock_model.encode(texts)
    print(f"✅ Mock embeddings shape: {mock_embeddings.shape}")
    
    # Test TF-IDF model
    tfidf_embeddings = tfidf_model.encode(texts)
    print(f"✅ TF-IDF embeddings shape: {tfidf_embeddings.shape}")
    
    # Test similarity calculator
    sim_calc = CosineSimilarityCalculator()
    similarity = sim_calc.calculate(mock_embeddings[0], mock_embeddings[1])
    print(f"✅ Cosine similarity: {similarity:.3f}")
    
    # Test text processor
    processor = ThaiTextProcessor()
    processed = processor.process("เสื้อยืด สี-ขาว Nike!!")
    print(f"✅ Processed text: '{processed}'")
    
    return True

def test_product_matcher():
    """Test complete ProductMatcher workflow."""
    print("\n🔍 Testing ProductMatcher...")
    
    # Create components
    embedding_model = ComponentFactory.create_embedding_model("mock", dimension=50)
    similarity_calc = ComponentFactory.create_similarity_calculator("cosine")
    text_processor = ComponentFactory.create_text_processor("thai")
    
    # Create matcher with lower threshold for testing with mock data
    from fresh_architecture import Config
    config = Config()
    config.similarity_threshold = 0.1  # Lower threshold for mock data
    matcher = ProductMatcher(
        embedding_model=embedding_model,
        similarity_calculator=similarity_calc,
        text_processor=text_processor,
        config=config
    )
    
    # Test data
    new_products = ["เสื้อยืดสีขาว Nike", "กางเกงยีนส์ Levi's"]
    old_products = ["เสื้อยืดสีขาว", "กางเกงยีนส์", "รองเท้าผ้าใบ", "หมวกแก๊ป"]
    
    # Find matches
    matches = matcher.find_matches(new_products, old_products)
    
    print(f"✅ Found {len(matches)} matches")
    for match in matches[:3]:  # Show first 3
        print(f"  - {match['query_product']} → {match['matched_product']} ({match['similarity_score']:.3f})")
    
    return len(matches) > 0

def test_caching():
    """Test embedding caching functionality."""
    print("\n🗄️ Testing Caching...")
    
    config = Config()
    config.cache_enabled = True
    
    matcher = ProductMatcher(
        embedding_model=MockEmbeddingModel(dimension=10),
        similarity_calculator=CosineSimilarityCalculator(),
        config=config
    )
    
    texts = ["เสื้อยืดสีขาว"] * 3
    
    # First call - should compute
    embeddings1 = matcher.get_embeddings(texts)
    
    # Second call - should use cache
    embeddings2 = matcher.get_embeddings(texts)
    
    # Should be identical
    cache_works = np.allclose(embeddings1, embeddings2)
    print(f"✅ Cache working: {cache_works}")
    print(f"✅ Cache size: {len(matcher._embedding_cache)}")
    
    return cache_works

def test_error_conditions():
    """Test error handling."""
    print("\n⚠️ Testing Error Conditions...")
    
    try:
        # Test invalid model type
        ComponentFactory.create_embedding_model("invalid_model")
        print("❌ Should have raised error for invalid model")
        return False
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")
    
    try:
        # Test empty inputs
        matcher = ProductMatcher(
            embedding_model=MockEmbeddingModel(),
            similarity_calculator=CosineSimilarityCalculator()
        )
        
        matches = matcher.find_matches([], [])
        print(f"✅ Handled empty inputs: {len(matches)} matches")
        
    except Exception as e:
        print(f"⚠️ Error with empty inputs: {e}")
        return False
    
    return True

def run_all_tests():
    """Run all tests and report results."""
    print("🚀 Starting Fresh Architecture Module Tests\n")
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("ProductMatcher", test_product_matcher), 
        ("Caching", test_caching),
        ("Error Handling", test_error_conditions)
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"❌ {name} failed with error: {e}")
            results[name] = False
    
    # Summary
    print(f"\n📊 Test Results:")
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    print(f"\n🎯 Overall: {total_passed}/{total_tests} tests passed")
    
    return total_passed == total_tests

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)