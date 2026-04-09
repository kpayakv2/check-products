#!/usr/bin/env python3
"""
Test Taxonomy Import Pipeline
==============================

ทดสอบการทำงานของ Backend + API + Database
"""

import requests
import pandas as pd
import time
import json
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"
CSV_FILE = "output/approved_products_for_import_20250914_110653.csv"
TEST_LIMIT = 5  # ทดสอบแค่ 5 รายการก่อน

def test_api_health():
    """ทดสอบว่า API Server ทำงานหรือไม่"""
    print("🔍 Testing API Health...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is healthy!")
            print(f"   Version: {data['version']}")
            print(f"   Uptime: {data['uptime']:.1f}s")
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API is not reachable: {e}")
        return False


def test_single_embedding():
    """ทดสอบ Single Embedding"""
    print("\n🔍 Testing Single Embedding...")
    try:
        payload = {"text": "กล่องล็อค 560 มล"}
        response = requests.post(
            f"{API_BASE_URL}/api/embed",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Single embedding successful!")
            print(f"   Dimension: {data['dimension']}")
            print(f"   Model: {data['model']}")
            print(f"   Processing Time: {data['processing_time']}s")
            print(f"   Sample values: {data['embedding'][:5]}...")
            return True
        else:
            print(f"❌ Embedding failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Embedding request failed: {e}")
        return False


def test_batch_embedding():
    """ทดสอบ Batch Embedding"""
    print("\n🔍 Testing Batch Embedding...")
    try:
        # อ่านสินค้าจาก CSV
        df = pd.read_csv(CSV_FILE, encoding='utf-8-sig')
        products = df['product_name'].head(TEST_LIMIT).tolist()
        
        print(f"   Testing with {len(products)} products...")
        
        start_time = time.time()
        payload = {"texts": products}
        response = requests.post(
            f"{API_BASE_URL}/api/embed/batch",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            elapsed = time.time() - start_time
            
            print(f"✅ Batch embedding successful!")
            print(f"   Products processed: {data['count']}")
            print(f"   Dimension: {data['dimension']}")
            print(f"   Model: {data['model']}")
            print(f"   API Time: {data['processing_time']}s")
            print(f"   Total Time: {elapsed:.3f}s")
            print(f"   Avg per product: {elapsed/len(products):.3f}s")
            
            # แสดงตัวอย่าง embedding
            print(f"\n📊 Sample Products & Embeddings:")
            for i, (product, embedding) in enumerate(zip(products, data['embeddings'])):
                vector_sample = embedding[:3]
                print(f"   {i+1}. {product[:30]}...")
                print(f"      Vector: [{vector_sample[0]:.4f}, {vector_sample[1]:.4f}, {vector_sample[2]:.4f}, ...]")
            
            return data
        else:
            print(f"❌ Batch embedding failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Batch embedding request failed: {e}")
        return None


def analyze_embeddings(embedding_data):
    """วิเคราะห์ Embeddings"""
    if not embedding_data:
        return
    
    print("\n📈 Embedding Analysis:")
    
    import numpy as np
    embeddings = np.array(embedding_data['embeddings'])
    
    # คำนวณ statistics
    print(f"   Shape: {embeddings.shape}")
    print(f"   Mean: {embeddings.mean():.6f}")
    print(f"   Std: {embeddings.std():.6f}")
    print(f"   Min: {embeddings.min():.6f}")
    print(f"   Max: {embeddings.max():.6f}")
    
    # คำนวณ Cosine Similarity ระหว่างสินค้า
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(embeddings)
    
    print(f"\n🔄 Similarity Matrix (ความคล้ายกันระหว่างสินค้า):")
    print(f"   Shape: {similarities.shape}")
    
    # หา pairs ที่คล้ายกันมาก (แต่ไม่ใช่ตัวเอง)
    np.fill_diagonal(similarities, 0)  # เอา diagonal ออก
    max_sim_idx = np.unravel_index(similarities.argmax(), similarities.shape)
    max_sim = similarities[max_sim_idx]
    
    print(f"   Most similar pair: products {max_sim_idx[0]} & {max_sim_idx[1]}")
    print(f"   Similarity score: {max_sim:.4f}")
    
    # แสดง similarity matrix
    df = pd.read_csv(CSV_FILE, encoding='utf-8-sig')
    products = df['product_name'].head(TEST_LIMIT).tolist()
    
    print(f"\n   Similarity Matrix:")
    print(f"   {'':20}", end='')
    for i in range(len(products)):
        print(f"P{i+1:2d} ", end='')
    print()
    
    for i in range(len(products)):
        print(f"   P{i+1:2d} {products[i][:15]:15}", end='')
        for j in range(len(products)):
            print(f"{similarities[i,j]:4.2f}", end=' ')
        print()


def test_product_matching():
    """ทดสอบ Product Matching API (ถ้ามี)"""
    print("\n🔍 Testing Product Matching...")
    try:
        df = pd.read_csv(CSV_FILE, encoding='utf-8-sig')
        query = df['product_name'].iloc[0]
        references = df['product_name'].iloc[1:6].tolist()
        
        payload = {
            "query_product": query,
            "reference_products": references,
            "threshold": 0.6,
            "top_k": 5
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/match/single",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            matches = response.json()
            print(f"✅ Product matching successful!")
            print(f"   Query: {query}")
            print(f"   Matches found: {len(matches)}")
            
            for match in matches:
                print(f"   - {match['matched_product'][:40]:40} (score: {match['similarity_score']:.4f})")
            
            return True
        else:
            print(f"⚠️ Product matching endpoint not available or failed")
            return False
    except Exception as e:
        print(f"⚠️ Product matching test skipped: {e}")
        return False


def main():
    """Main test function"""
    print("=" * 70)
    print("🧪 TAXONOMY IMPORT PIPELINE TEST")
    print("=" * 70)
    
    # Check if CSV exists
    csv_path = Path(CSV_FILE)
    if not csv_path.exists():
        print(f"❌ CSV file not found: {CSV_FILE}")
        return
    
    df = pd.read_csv(CSV_FILE, encoding='utf-8-sig')
    print(f"\n📄 CSV File: {CSV_FILE}")
    print(f"   Total products: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Testing with: {TEST_LIMIT} products")
    
    # Run tests
    results = {}
    
    # Test 1: API Health
    results['health'] = test_api_health()
    if not results['health']:
        print("\n❌ API is not running. Please start api_server.py first!")
        return
    
    # Test 2: Single Embedding
    results['single_embedding'] = test_single_embedding()
    
    # Test 3: Batch Embedding
    embedding_data = test_batch_embedding()
    results['batch_embedding'] = embedding_data is not None
    
    # Test 4: Analyze Embeddings
    if embedding_data:
        analyze_embeddings(embedding_data)
    
    # Test 5: Product Matching
    results['matching'] = test_product_matching()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"   {test_name:20} : {status}")
    
    print(f"\n   Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Backend & API are working correctly!")
        print("\n📝 Next Steps:")
        print("   1. Start Next.js dev server: cd taxonomy-app && npm run dev")
        print("   2. Open http://localhost:3000/import/wizard")
        print("   3. Upload CSV file and test end-to-end flow")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
