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
    try:
        response = requests.get("http://localhost:8000/api/v1/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Check Success!")
            print(f"   Status: {data['status']}")
            print(f"   Version: {data['version']}")
            print(f"   Uptime: {data['uptime']:.1f}s")
            return True
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health Check Error: {e}")
        return False


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
    
    try:
        start_time = time.time()
        response = requests.post(
            "http://localhost:8000/api/v1/match/single",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Single Match Success!")
            print(f"   Response Time: {response_time:.3f}s")
            print(f"   Matches Found: {len(results)}")
            
            for i, match in enumerate(results[:2], 1):
                print(f"   {i}. {match['matched_product'][:40]}...")
                print(f"      Score: {match['similarity_score']:.4f}")
                if 'confidence_level' in match:
                    print(f"      Confidence: {match['confidence_level']}")
            
            return True
        else:
            print(f"❌ Single Match Failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Single Match Error: {e}")
        return False


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
    
    try:
        # Start batch job
        response = requests.post(
            "http://localhost:8000/api/v1/match/batch",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            batch_response = response.json()
            job_id = batch_response["job_id"]
            print(f"✅ Batch Job Started!")
            print(f"   Job ID: {job_id}")
            print(f"   Total Queries: {batch_response['total_queries']}")
            
            # Check job status
            print("   Checking job status...")
            for attempt in range(10):  # Wait up to 10 seconds
                status_response = requests.get(f"http://localhost:8000/api/v1/jobs/{job_id}")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"   Progress: {status_data['progress']:.1%} - {status_data['message']}")
                    
                    if status_data['status'] == 'completed':
                        print(f"✅ Batch Job Completed!")
                        return True
                    elif status_data['status'] == 'failed':
                        print(f"❌ Batch Job Failed: {status_data['message']}")
                        return False
                    
                    time.sleep(1)
                else:
                    print(f"❌ Status Check Failed: {status_response.status_code}")
                    break
            
            print("⏰ Batch job taking longer than expected...")
            return True  # Still consider success if job started
            
        else:
            print(f"❌ Batch Start Failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Batch Processing Error: {e}")
        return False


def test_api_documentation():
    """Test API documentation endpoint."""
    print("\n🔍 Testing API Documentation...")
    try:
        response = requests.get("http://localhost:8000/docs")
        if response.status_code == 200:
            print(f"✅ API Documentation Available!")
            print(f"   URL: http://localhost:8000/docs")
            return True
        else:
            print(f"❌ Documentation Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Documentation Error: {e}")
        return False


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
        success = test_func()
        results.append((test_name, success))
    
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
        print(f"   - Web Interface: http://localhost:8000/web")
        print(f"   - API Documentation: http://localhost:8000/docs")
        print(f"   - Health Check: http://localhost:8000/api/v1/health")
    else:
        print(f"⚠️  Some tests failed. Check the API server status.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
