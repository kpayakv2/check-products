#!/usr/bin/env python3
"""
Test API endpoints directly
"""

import json
import requests
import time
import threading
from web_server import app

def start_test_server():
    """Start server for testing"""
    app.run(debug=False, host='localhost', port=5001, use_reloader=False)

def test_api_endpoints():
    """Test all API endpoints"""
    print("🧪 Testing API Endpoints")
    print("=" * 50)
    
    base_url = "http://localhost:5001"
    
    try:
        # Test 1: API Status
        print("1. Testing /api/status...")
        response = requests.get(f"{base_url}/api/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ API Status OK")
            print(f"   System: {data.get('system', 'Unknown')}")
            print(f"   Version: {data.get('version', 'Unknown')}")
            print(f"   Model cache: {data.get('cache_statistics', {}).get('cached_models', 0)} models")
        else:
            print(f"   ❌ API Status failed: {response.status_code}")
        
        # Test 2: Save Feedback
        print("\\n2. Testing /save-feedback...")
        feedback_data = {
            'old_product': 'ไม่แขวนเสื้อ - ลวด 111 หนุมาน STCP',
            'new_product': 'ไม่แขวนเสื้อ - ลวด 111 หนุมาน STCP',
            'human_feedback': 'similar',
            'similarity': 1.0,
            'confidence': 0.95,
            'reviewer': 'test_user',
            'comments': 'ทดสอบระบบ API'
        }
        
        response = requests.post(
            f"{base_url}/save-feedback", 
            json=feedback_data,
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"   ✅ Feedback saved successfully")
                print(f"   Message: {result.get('message', 'No message')}")
                print(f"   Total feedback: {result.get('total_feedback', 0)}")
            else:
                print(f"   ❌ Feedback save failed: {result.get('message', 'Unknown error')}")
        else:
            print(f"   ❌ Feedback API failed: {response.status_code}")
        
        # Test 3: Cache Stats
        print("\\n3. Testing /api/cache/stats...")
        response = requests.get(f"{base_url}/api/cache/stats", timeout=5)
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                stats = result.get('cache_stats', {})
                print(f"   ✅ Cache stats retrieved")
                print(f"   Cached models: {stats.get('cached_models', 0)}")
                print(f"   Memory usage: {stats.get('memory_usage_mb', 0):.1f} MB")
                
                recommendations = result.get('recommendations', [])
                if recommendations:
                    print("   Recommendations:")
                    for rec in recommendations:
                        print(f"     - {rec}")
            else:
                print(f"   ❌ Cache stats failed: {result.get('message', 'Unknown error')}")
        else:
            print(f"   ❌ Cache stats API failed: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    print("🚀 Starting test server on port 5001...")
    
    # Start server in background thread
    server_thread = threading.Thread(target=start_test_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    time.sleep(3)
    
    # Test endpoints
    test_api_endpoints()
    
    print("\\n🎯 API Testing Complete!")
    print("Press Ctrl+C to stop the test server")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\\n👋 Test server stopped")