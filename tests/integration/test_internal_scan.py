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
    
    response = requests.post(
        f"{API_BASE_URL}/api/v1/match/scan-internal",
        json=payload,
        timeout=15
    )
    
    assert response.status_code == 200, f"Failed to start scan: {response.text}"
    start_res = response.json()
    task_id = start_res.get("task_id")
    assert task_id, f"No task_id in start response: {start_res}"
    
    print(f"Scan started, task_id: {task_id}")
    
    # Poll status
    status = "pending"
    # สแกนของจริงใช้เวลา ~58 วิ ตอนรันเดี่ยว (วัด 30 ส.ค. 2569) เพดานเดิม 30x2 วิ = 60 วิ
    # จึงตกแบบสุ่มเมื่อรันทั้งชุดพร้อมกัน เพราะ FastAPI ตัวเดียวกันรับงานจากเทสต์อื่นด้วย
    max_polls = 90
    poll_interval = 2.0
    body = {}
    for _ in range(max_polls):
        status_res = requests.get(f"{API_BASE_URL}/api/v1/match/scan-internal/status/{task_id}", timeout=10)
        assert status_res.status_code == 200, f"Failed to get task status: {status_res.text}"
        body = status_res.json()
        status = body.get("status")
        if status in ("completed", "failed"):
            break
        time.sleep(poll_interval)
        
    assert status == "completed", f"Scan task did not complete (status={status}, error={body.get('error')})"
    data = body.get("result")
    assert data is not None, f"No result in body: {body}"
    
    print("Connection successful!")
    print(f"API Response time: {data['processing_time']}s")
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


if __name__ == "__main__":
    test_internal_scan()
