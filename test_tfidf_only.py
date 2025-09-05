#!/usr/bin/env python3
"""
Test optimized TF-IDF model only (without PyTorch dependencies).
"""

import sys
sys.path.insert(0, '.')

print("🧪 Testing Optimized TF-IDF Model...")

# Test data
test_texts = [
    "iPhone 14 Pro Max 256GB สีดำ",
    "Samsung Galaxy S23 Ultra สีเงิน", 
    "MacBook Pro M2 13 นิ้ว",
    "iPad Air สีขาว 64GB",
    "AirPods Pro รุ่นใหม่"
]

print(f"📝 Test texts: {len(test_texts)} items")

try:
    # Test basic sklearn TF-IDF first
    from sklearn.feature_extraction.text import TfidfVectorizer
    print("✅ scikit-learn available")
    
    vectorizer = TfidfVectorizer(max_features=1000)
    embeddings = vectorizer.fit_transform(test_texts).toarray()
    print(f"📐 Basic TF-IDF shape: {embeddings.shape}")
    
    # Now test our optimized version
    from advanced_models import OptimizedTFIDFModel
    print("✅ OptimizedTFIDFModel imported")
    
    model = OptimizedTFIDFModel(
        max_features=1000,
        ngram_range=(1, 2),
        cache_dir=None  # No caching for now
    )
    
    optimized_embeddings = model.encode(test_texts)
    print(f"📐 Optimized TF-IDF shape: {optimized_embeddings.shape}")
    
    model_info = model.get_model_info()
    print(f"📊 Model info: {model_info}")
    
    print("✅ Optimized TF-IDF test successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
