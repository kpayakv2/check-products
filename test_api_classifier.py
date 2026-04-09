#!/usr/bin/env python3
"""
Test Category Classification API
=================================

Quick test script for the new category classification endpoints.
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("\n" + "=" * 60)
    print("🏥 Testing Health Endpoint")
    print("=" * 60)
    
    response = requests.get(f"{API_BASE_URL}/api/v1/health")
    if response.ok:
        data = response.json()
        print(f"✅ Status: {data['status']}")
        print(f"✅ Version: {data['version']}")
        print(f"✅ Components: {json.dumps(data['components'], indent=2, ensure_ascii=False)}")
    else:
        print(f"❌ Health check failed: {response.status_code}")

def test_root():
    """Test root endpoint."""
    print("\n" + "=" * 60)
    print("🏠 Testing Root Endpoint")
    print("=" * 60)
    
    response = requests.get(API_BASE_URL)
    if response.ok:
        data = response.json()
        print(f"✅ API Name: {data['name']}")
        print(f"✅ Endpoints:")
        for key, value in data['endpoints'].items():
            print(f"   - {key:20} → {value}")
    else:
        print(f"❌ Root endpoint failed: {response.status_code}")

def test_single_classification():
    """Test single product classification."""
    print("\n" + "=" * 60)
    print("🎯 Testing Single Product Classification")
    print("=" * 60)
    
    test_products = [
        "กล่องล็อค 560 มล",
        "ถังน้ำ 1000 ลิตร",
        "เก้าอี้พลาสติก",
        "ขวดน้ำ 500ml",
        "ชาม 8 นิ้ว"
    ]
    
    for product in test_products:
        print(f"\n📦 Product: {product}")
        
        payload = {
            "product_name": product,
            "method": "hybrid",
            "top_k": 3
        }
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/api/classify/category",
            json=payload,
            timeout=30
        )
        elapsed = time.time() - start_time
        
        if response.ok:
            data = response.json()
            print(f"   ⏱️  Processing time: {data['processing_time']}s (Total: {elapsed:.3f}s)")
            
            if data['top_suggestion']:
                top = data['top_suggestion']
                print(f"   ✅ Top: {top['category_name']} ({top['confidence']:.3f})")
                print(f"      Method: {top['method']}")
                if top.get('matched_keyword'):
                    print(f"      Keyword: {top['matched_keyword']}")
                if top.get('methods'):
                    print(f"      Methods: {', '.join(top['methods'])}")
            else:
                print("   ⚠️  No suggestions")
            
            # Show all suggestions
            if data['suggestions']:
                print(f"\n   All suggestions ({len(data['suggestions'])}):")
                for i, sug in enumerate(data['suggestions'], 1):
                    print(f"   {i}. {sug['category_name']:30} ({sug['confidence']:.3f}) - {sug['method']}")
        else:
            print(f"   ❌ Classification failed: {response.status_code}")
            print(f"      {response.text}")

def test_batch_classification():
    """Test batch product classification."""
    print("\n" + "=" * 60)
    print("📦 Testing Batch Classification")
    print("=" * 60)
    
    products = [
        "กล่องล็อค 560 มล",
        "ถังน้ำ",
        "เก้าอี้",
        "ขวดน้ำ",
        "ชาม"
    ]
    
    payload = {
        "products": products,
        "method": "hybrid",
        "top_k": 2
    }
    
    print(f"\n📊 Classifying {len(products)} products...")
    
    start_time = time.time()
    response = requests.post(
        f"{API_BASE_URL}/api/classify/batch",
        json=payload,
        timeout=60
    )
    elapsed = time.time() - start_time
    
    if response.ok:
        data = response.json()
        print(f"✅ Total products: {data['total_products']}")
        print(f"⏱️  Processing time: {data['processing_time']:.3f}s (Total: {elapsed:.3f}s)")
        print(f"⚡ Avg per product: {data['processing_time']/data['total_products']:.3f}s")
        
        print("\n📋 Results:")
        for i, result in enumerate(data['results'], 1):
            product = result['product_name']
            if result['top_suggestion']:
                top = result['top_suggestion']
                print(f"\n{i}. {product:30}")
                print(f"   → {top['category_name']:30} ({top['confidence']:.3f})")
            else:
                print(f"\n{i}. {product:30}")
                print(f"   → No suggestions")
    else:
        print(f"❌ Batch classification failed: {response.status_code}")
        print(f"   {response.text}")

def test_methods_comparison():
    """Compare different classification methods."""
    print("\n" + "=" * 60)
    print("🔬 Testing Classification Methods Comparison")
    print("=" * 60)
    
    product = "กล่องล็อค 560 มล"
    methods = ['keyword', 'embedding', 'hybrid']
    
    print(f"\n📦 Product: {product}\n")
    
    for method in methods:
        payload = {
            "product_name": product,
            "method": method,
            "top_k": 3
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/classify/category",
            json=payload
        )
        
        if response.ok:
            data = response.json()
            print(f"🔍 Method: {method.upper()}")
            print(f"   Time: {data['processing_time']:.3f}s")
            
            if data['suggestions']:
                for i, sug in enumerate(data['suggestions'][:3], 1):
                    print(f"   {i}. {sug['category_name']:30} ({sug['confidence']:.3f})")
            else:
                print("   No suggestions")
            print()
        else:
            print(f"❌ {method} failed: {response.status_code}\n")

def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 Category Classification API Test Suite")
    print("=" * 60)
    print(f"\n🔗 API URL: {API_BASE_URL}")
    print("\n⚠️  Make sure the API server is running:")
    print("   python api_server.py")
    
    input("\n Press Enter to start tests... ")
    
    try:
        # Test basic endpoints
        test_root()
        test_health()
        
        # Test classification
        test_single_classification()
        test_batch_classification()
        test_methods_comparison()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error!")
        print("   Make sure API server is running: python api_server.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
