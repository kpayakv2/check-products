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


def test_root_endpoint():
    """Test API root endpoint."""
    print("\n🔍 Testing Root Endpoint...")
    response = requests.get("http://127.0.0.1:8000/")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "operational"
    assert "web_interface" not in data.get("endpoints", {})
    assert "/api/v1/health" in data.get("endpoints", {}).values()
    print("✅ Root Endpoint Success (No /web dead link)!")


def test_clean_texts():
    """Test text cleaning endpoint."""
    print("\n🔍 Testing Text Cleaning...")
    payload = {"texts": ["ปากกาลูกลื่น 0.5มม. (แพ็ค 3 ด้าม)"]}
    response = requests.post(
        "http://127.0.0.1:8000/api/v1/clean",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert len(data["results"]) == 1
    assert "cleaned" in data["results"][0]
    print(f"✅ Clean Success: {data['results'][0]['cleaned']}")


def test_scan_internal():
    """Test internal duplicates scan initiation."""
    print("\n🔍 Testing Internal Duplicates Scan...")
    payload = {"threshold": 0.95, "limit": 5}
    response = requests.post(
        "http://127.0.0.1:8000/api/v1/match/scan-internal",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert "task_id" in data
    print(f"✅ Scan Task Started! Task ID: {data['task_id']}")


def test_api_documentation():
    """Test API documentation endpoint."""
    print("\n🔍 Testing API Documentation...")
    response = requests.get("http://127.0.0.1:8000/docs")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    print(f"✅ API Documentation Available!")
    print(f"   URL: http://127.0.0.1:8000/docs")


def main():
    """Run all API tests."""
    print("🚀 API Client Testing Suite")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("Root Endpoint", test_root_endpoint),
        ("Text Cleaning", test_clean_texts),
        ("Internal Scan", test_scan_internal),
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
        print(f"🎉 All tests passed! API is working perfectly!")
        print(f"\n🌐 Access Points:")
        print(f"   - API Documentation: http://127.0.0.1:8000/docs")
        print(f"   - Health Check: http://127.0.0.1:8000/api/v1/health")
    else:
        print(f"⚠️  Some tests failed. Check the API server status.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
