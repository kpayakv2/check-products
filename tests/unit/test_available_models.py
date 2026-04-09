#!/usr/bin/env python3
"""
Test Product Similarity with Available Models (TF-IDF & Mock)
=============================================================

ทดสอบความคล้ายของสินค้าจากโฟลเดอร์ input โดยใช้โมเดลที่พร้อมใช้งาน
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

# Add project path
sys.path.append(str(Path(__file__).parent))

def load_input_products():
    """โหลดข้อมูลสินค้าจากโฟลเดอร์ input"""
    print("📂 Loading products from input folder...")
    
    input_dir = Path("input")
    
    # Load new products
    new_product_file = input_dir / "new_product" / "POS_เพิ่มสินค้า_20250727_063658_จากไฟล์สินค้าใหม่.csv"
    old_product_file = input_dir / "old_product" / "cleaned_products.csv"
    
    products = {}
    
    # Load new products
    if new_product_file.exists():
        df_new = pd.read_csv(new_product_file)
        products['new'] = df_new['รายการ'].dropna().astype(str).tolist()
        print(f"   ✅ Loaded {len(products['new'])} new products")
        
        # Show samples
        print(f"   🔍 Sample new products:")
        for i, product in enumerate(products['new'][:5], 1):
            print(f"      {i}. {product}")
    
    # Load old products
    if old_product_file.exists():
        df_old = pd.read_csv(old_product_file)
        products['old'] = df_old['name'].dropna().astype(str).tolist()
        print(f"   ✅ Loaded {len(products['old'])} old products")
        
        # Show samples
        print(f"   🔍 Sample old products:")
        for i, product in enumerate(products['old'][:5], 1):
            print(f"      {i}. {product}")
    
    return products

def test_with_different_models(new_products, old_products):
    """ทดสอบกับโมเดลต่างๆ"""
    print(f"\n🧪 Testing with different models...")
    
    from fresh_implementations import ComponentFactory
    from fresh_architecture import ProductMatcher, Config
    
    # Test with different model types
    model_types = ['mock', 'tfidf', 'optimized-tfidf']
    
    results = {}
    
    for model_type in model_types:
        print(f"\n🔍 Testing with {model_type.upper()} model:")
        print("-" * 40)
        
        try:
            # Create components
            embedding_model = ComponentFactory.create_embedding_model(model_type, dimension=200)
            similarity_calc = ComponentFactory.create_similarity_calculator("cosine")
            text_processor = ComponentFactory.create_text_processor("thai")
            
            # Create config with different thresholds
            config = Config()
            config.similarity_threshold = 0.5  # Lower threshold for testing
            
            # Create matcher
            matcher = ProductMatcher(
                embedding_model=embedding_model,
                similarity_calculator=similarity_calc,
                text_processor=text_processor,
                config=config
            )
            
            # Limit products for testing (first 20 each)
            test_new = new_products[:20]
            test_old = old_products[:20]
            
            print(f"   📊 Testing with {len(test_new)} new vs {len(test_old)} old products")
            print(f"   🎯 Similarity threshold: {config.similarity_threshold}")
            
            # Find matches
            start_time = time.time()
            matches = matcher.find_matches(test_new, test_old)
            match_time = time.time() - start_time
            
            print(f"   ⏱️ Processing time: {match_time:.2f} seconds")
            print(f"   🎯 Found {len(matches)} matches")
            
            # Show top matches
            if matches:
                print(f"\n   🏆 Top matches:")
                print(f"   {'New Product':<35} {'Old Product':<35} {'Score':<8}")
                print(f"   {'-'*35} {'-'*35} {'-'*8}")
                
                for match in matches[:5]:
                    new_name = match['query_product'][:32] + "..." if len(match['query_product']) > 35 else match['query_product']
                    old_name = match['matched_product'][:32] + "..." if len(match['matched_product']) > 35 else match['matched_product']
                    print(f"   {new_name:<35} {old_name:<35} {match['similarity_score']:.4f}")
            
            results[model_type] = {
                'matches': len(matches),
                'processing_time': match_time,
                'top_matches': matches[:10] if matches else []
            }
            
        except Exception as e:
            print(f"   ❌ Error with {model_type}: {e}")
            results[model_type] = {'error': str(e)}
    
    return results

def test_sentence_transformer_if_available(new_products, old_products):
    """ทดสอบ SentenceTransformer ถ้าโหลดเสร็จแล้ว"""
    print(f"\n🤖 Testing SentenceTransformer (if available)...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        cache_dir = Path("model_cache")
        
        # Check if model is fully downloaded
        model_dir = cache_dir / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
        if not model_dir.exists():
            print("   ❌ SentenceTransformer model not found in cache")
            return None
        
        # Try to load model with local_files_only to avoid network
        print("   🔄 Attempting to load from local cache...")
        
        model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2",
            cache_folder=str(cache_dir),
            local_files_only=True  # Don't try to download
        )
        
        print(f"   ✅ SentenceTransformer loaded successfully!")
        print(f"   📊 Dimension: {model.get_sentence_embedding_dimension()}")
        
        # Quick test with small sample
        test_new = new_products[:5]
        test_old = old_products[:5]
        
        print(f"   🧪 Quick test with {len(test_new)} vs {len(test_old)} products...")
        
        # Encode
        new_embeddings = model.encode(test_new)
        old_embeddings = model.encode(test_old)
        
        # Calculate similarities
        similarities = np.dot(new_embeddings, old_embeddings.T)
        
        print(f"   ✅ Test successful! Similarity range: [{similarities.min():.3f}, {similarities.max():.3f}]")
        
        return model
        
    except Exception as e:
        print(f"   ❌ SentenceTransformer not available: {e}")
        return None

def analyze_product_categories(new_products, old_products):
    """วิเคราะห์หมวดหมู่สินค้า"""
    print(f"\n📊 Product Category Analysis:")
    
    # Analyze new products
    new_categories = {}
    for product in new_products[:50]:  # First 50
        words = product.lower().split()
        for word in words:
            if len(word) > 2:  # Skip short words
                new_categories[word] = new_categories.get(word, 0) + 1
    
    # Top words in new products
    top_new_words = sorted(new_categories.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"   🆕 Top words in new products:")
    for word, count in top_new_words:
        print(f"      {word}: {count}")
    
    # Analyze old products  
    old_categories = {}
    for product in old_products[:100]:  # First 100
        words = product.lower().split()
        for word in words:
            if len(word) > 2:
                old_categories[word] = old_categories.get(word, 0) + 1
    
    # Top words in old products
    top_old_words = sorted(old_categories.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"   📝 Top words in old products:")
    for word, count in top_old_words:
        print(f"      {word}: {count}")
    
    # Common words
    common_words = set(new_categories.keys()) & set(old_categories.keys())
    print(f"   🔗 Common words: {len(common_words)} ({', '.join(list(common_words)[:10])}...)")

def save_test_results(results, new_products, old_products):
    """บันทึกผลการทดสอบ"""
    print(f"\n💾 Saving test results...")
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    
    # Save summary
    summary_data = []
    for model_type, result in results.items():
        if 'error' in result:
            summary_data.append({
                'model_type': model_type,
                'status': 'error',
                'matches': 0,
                'processing_time': 0,
                'error': result['error']
            })
        else:
            summary_data.append({
                'model_type': model_type,
                'status': 'success',
                'matches': result['matches'],
                'processing_time': result['processing_time'],
                'error': None
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / f"model_comparison_summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"   ✅ Summary saved: {summary_file}")
    
    # Save detailed matches
    for model_type, result in results.items():
        if 'error' not in result and result['top_matches']:
            matches_df = pd.DataFrame(result['top_matches'])
            matches_file = output_dir / f"matches_{model_type}_{timestamp}.csv"
            matches_df.to_csv(matches_file, index=False, encoding='utf-8-sig')
            print(f"   ✅ {model_type} matches saved: {matches_file}")

def main():
    """Main function"""
    print("🚀 Testing Product Similarity with Available Models")
    print("=" * 60)
    
    # Load products
    products = load_input_products()
    
    if not products:
        print("❌ Could not load product data")
        return
    
    new_products = products.get('new', [])
    old_products = products.get('old', [])
    
    if not new_products or not old_products:
        print("❌ Missing product data")
        return
    
    print(f"\n📊 Dataset loaded:")
    print(f"   New products: {len(new_products)}")
    print(f"   Old products: {len(old_products)}")
    
    # Analyze categories
    analyze_product_categories(new_products, old_products)
    
    # Test with different models
    results = test_with_different_models(new_products, old_products)
    
    # Try SentenceTransformer if available
    st_model = test_sentence_transformer_if_available(new_products, old_products)
    
    # Save results
    save_test_results(results, new_products, old_products)
    
    print(f"\n🎉 Testing completed!")
    print(f"📁 Results saved in output/ folder")
    
    # Summary
    print(f"\n📈 Summary:")
    for model_type, result in results.items():
        if 'error' not in result:
            print(f"   {model_type.upper()}: {result['matches']} matches in {result['processing_time']:.2f}s")
        else:
            print(f"   {model_type.upper()}: Error - {result['error'][:50]}...")

if __name__ == "__main__":
    main()