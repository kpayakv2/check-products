#!/usr/bin/env python3
"""
Test Phase 5 API Client
=======================

Simple client to test our Phase 5 API.
"""

import requests
import json
import time


def test_health_check():
    """Test API health endpoint."""
    print("🔍 Testing Health Check...")
    response = requests.get("http://127.0.0.1:8000/api/v1/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    print(f"✅ Health Check Success!")
    print(f"   Status: {data['status']}")
    print(f"   Version: {data['version']}")
    print(f"   Uptime: {data['uptime']:.1f}s")


def test_single_match():
    """Test single product matching."""
    print("\n🔍 Testing Single Product Match...")
    
    payload = {
        "query_product": "ไอโฟน 14 โปร แมกซ์",
        "reference_products": [
            "iPhone 14 Pro Max 128GB สีทอง",
            "Samsung Galaxy S23 Ultra",
            "iPad Pro 12.9นิ้ว M2",
            "MacBook Air M2"
        ],
        "threshold": 0.5,
        "top_k": 5,
        "include_metadata": True,
        "include_confidence": True
    }
    
    start_time = time.time()
    response = requests.post(
        "http://127.0.0.1:8000/api/v1/match/single",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    response_time = time.time() - start_time
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    results = response.json()
    print(f"✅ Single Match Success!")
    print(f"   Response Time: {response_time:.3f}s")
    print(f"   Matches Found: {len(results)}")
    
    for i, match in enumerate(results[:2], 1):
        print(f"   {i}. {match['matched_product'][:40]}...")
        print(f"      Score: {match['similarity_score']:.4f}")
        if 'confidence_level' in match:
            print(f"      Confidence: {match['confidence_level']}")


def test_batch_match():
    """Test batch processing."""
    print("\n🔍 Testing Batch Processing...")
    
    payload = {
        "query_products": [
            "ไอโฟน 14 โปร แมกซ์",
            "แกแลกซี่ เอส 23",
            "ไอแพด โปร"
        ],
        "reference_products": [
            "iPhone 14 Pro Max 128GB สีทอง",
            "Samsung Galaxy S23 Ultra",
            "iPad Pro 12.9นิ้ว M2 WiFi 128GB"
        ],
        "threshold": 0.5,
        "include_confidence": True
    }
    
    # Start batch job
    response = requests.post(
        "http://127.0.0.1:8000/api/v1/match/batch",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    batch_response = response.json()
    job_id = batch_response["job_id"]
    print(f"✅ Batch Job Started!")
    print(f"   Job ID: {job_id}")
    print(f"   Total Queries: {batch_response['total_queries']}")
    
    # Check job status
    print("   Checking job status...")
    job_completed = False
    for attempt in range(10):  # Wait up to 10 seconds
        status_response = requests.get(f"http://127.0.0.1:8000/api/v1/jobs/{job_id}")
        assert status_response.status_code == 200, f"Status check failed: {status_response.text}"
        status_data = status_response.json()
        print(f"   Progress: {status_data['progress']:.1%} - {status_data['message']}")
        
        if status_data['status'] == 'completed':
            print(f"✅ Batch Job Completed!")
            job_completed = True
            break
        elif status_data['status'] == 'failed':
            assert False, f"Batch Job Failed: {status_data['message']}"
        
        time.sleep(1)
    
    if not job_completed:
        print("⏰ Batch job taking longer than expected...")


def test_api_documentation():
    """Test API documentation endpoint."""
    print("\n🔍 Testing API Documentation...")
    response = requests.get("http://127.0.0.1:8000/docs")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    print(f"✅ API Documentation Available!")
    print(f"   URL: http://127.0.0.1:8000/docs")


def main():
    """Run all API tests."""
    print("🚀 Phase 5 API Testing Suite")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("Single Match", test_single_match),
        ("Batch Processing", test_batch_match),
        ("API Documentation", test_api_documentation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 20}")
        try:
            test_func()
            results.append((test_name, True))
        except AssertionError as e:
            print(f"❌ {test_name} failed: {e}")
            results.append((test_name, False))
        except Exception as e:
            print(f"❌ {test_name} error: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n🎯 Test Results Summary")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"🎉 All tests passed! Phase 5 API is working perfectly!")
        print(f"\n🌐 Access Points:")
        print(f"   - Web Interface: http://127.0.0.1:8000/web")
        print(f"   - API Documentation: http://127.0.0.1:8000/docs")
        print(f"   - Health Check: http://127.0.0.1:8000/api/v1/health")
    else:
        print(f"⚠️  Some tests failed. Check the API server status.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
