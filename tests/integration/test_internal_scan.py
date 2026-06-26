#!/usr/bin/env python3
"""
Test Internal Catalog Scan API
==============================

Verifies that the /api/v1/match/scan-internal endpoint is functional.
"""

import requests
import json
import time

API_BASE_URL = "http://127.0.0.1:8000"

def test_internal_scan():
    print("\n" + "=" * 60)
    print("Testing Internal Catalog Deduplication Scan")
    print("=" * 60)
    
    payload = {
        "threshold": 0.95,
        "limit": 10
    }
    
    start_time = time.time()
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/match/scan-internal",
            json=payload,
            timeout=90
        )
        elapsed = time.time() - start_time
        
        if response.ok:
            data = response.json()
            print("Connection successful!")
            print(f"API Response time: {data['processing_time']}s (Total: {elapsed:.3f}s)")
            print(f"Total Scanned Products: {data['total_scanned']}")
            print(f"Pairs Found Exceeding 0.95: {data['pairs_found']}")
            
            # Print top matches
            print(f"\nTop {len(data['results'])} Duplicate Matches Returned:")
            for i, result in enumerate(data['results'], 1):
                print(f"\n{i}. Cosine Match Similarity: {result['similarity']:.4f}")
                print(f"   A: '{result['name_a']}'")
                print(f"      SKU: {result['sku_a']} | Category: {result['category_a']}")
                print(f"   B: '{result['name_b']}'")
                print(f"      SKU: {result['sku_b']} | Category: {result['category_b']}")
            
            print("\n" + "=" * 60)
            print("Internal duplicate scan API test completed successfully!")
            print("=" * 60)
            return True
        else:
            print(f"Scan failed: {response.status_code}")
            print(f"   {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("\nConnection Error!")
        print("   Make sure the FastAPI server is running: python -m uvicorn src.api.api_server:app --reload")
        return False
    except Exception as e:
        print(f"\n[ERR] Error occurred: {e}")
        return False

if __name__ == "__main__":
    test_internal_scan()
