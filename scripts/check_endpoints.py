# =============================================================================
# check_endpoints.py — API Health & Status Dashboard
# =============================================================================
# สคริปต์สำหรับตรวจสอบสุขภาพของทุก Endpoint บน API Server (พอร์ต 8000)
# ช่วยเช็คว่า Backend ทำงานปกติ ครบถ้วน และตอบสนองรวดเร็วหรือไม่
#
# วิธีใช้:
#   .\.venv\Scripts\python scripts\check_endpoints.py
# =============================================================================

import os
import sys
import time
import requests
import json

# ตั้งค่าการแสดงผลภาษาไทยให้ถูกต้องบน Windows Console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

API_BASE = "http://127.0.0.1:8000"

# ANSI Colors สำหรับตกแต่ง Terminal ให้เป็นระเบียบและสวยงาม
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def check_endpoint(name, method, path, payload=None):
    url = f"{API_BASE}{path}"
    print(f"📡 Testing {Colors.BOLD}{name:<30}{Colors.END} [{method}] {path} ... ", end="")
    sys.stdout.flush()
    
    start_time = time.time()
    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=payload, timeout=5)
        else:
            print(f"{Colors.RED}Unsupported Method{Colors.END}")
            return False, 0, "N/A"
            
        elapsed = (time.time() - start_time) * 1000  # ms
        
        if response.status_code in [200, 201]:
            print(f"{Colors.GREEN}🟢 OK{Colors.END} ({elapsed:.1f} ms)")
            return True, elapsed, response.status_code
        else:
            print(f"{Colors.RED}🔴 FAILED{Colors.END} (HTTP {response.status_code})")
            return False, elapsed, response.status_code
            
    except requests.exceptions.ConnectionError:
        print(f"{Colors.RED}🔴 CONNECTION REFUSED{Colors.END}")
        return False, 0, "No Conn"
    except Exception as e:
        print(f"{Colors.RED}🔴 ERROR: {e}{Colors.END}")
        return False, 0, "Error"

def main():
    print(f"\n{Colors.CYAN}============================================================{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}   🚀 PHAYAK Engine — API Endpoints Checker Dashboard{Colors.END}")
    print(f"{Colors.CYAN}============================================================{Colors.END}\n")
    
    print(f"🔗 Target Server: {Colors.BOLD}{API_BASE}{Colors.END}")
    print("------------------------------------------------------------\n")
    
    # กำหนด Endpoint ต่างๆ และข้อมูลทดสอบ
    endpoints = [
        {
            "name": "1. Root Info",
            "method": "GET",
            "path": "/",
            "payload": None
        },
        {
            "name": "2. System Health",
            "method": "GET",
            "path": "/api/v1/health",
            "payload": None
        },
        {
            "name": "3. Single Embedding (384-dim)",
            "method": "POST",
            "path": "/api/embed",
            "payload": {"text": "ทดสอบเวกเตอร์สินค้า"}
        },
        {
            "name": "4. Batch Embeddings (384-dim)",
            "method": "POST",
            "path": "/api/embed/batch",
            "payload": {"texts": ["สินค้า ก", "สินค้า ข"]}
        },
        {
            "name": "5. Hybrid Classification",
            "method": "POST",
            "path": "/api/classify/category",
            "payload": {"product_name": "สว่านไฟฟ้าไร้สาย Makita", "method": "hybrid", "top_k": 3}
        },
        {
            "name": "6. Keyword Classification",
            "method": "POST",
            "path": "/api/classify/category",
            "payload": {"product_name": "จานเซรามิก ลายไทย", "method": "keyword", "top_k": 3}
        },
        {
            "name": "7. Embedding Classification",
            "method": "POST",
            "path": "/api/classify/category",
            "payload": {"product_name": "ถังน้ำพลาสติก 20 ลิตร", "method": "embedding", "top_k": 3}
        },
        {
            "name": "8. Product Similarity Match",
            "method": "POST",
            "path": "/api/match",
            "payload": {"source_product": "กะละมัง 549 ดำ SMK", "target_product": "กะละมัง 549 สี SMK"}
        }
    ]
    
    success_count = 0
    total_time = 0
    operational_endpoints = []
    
    for ep in endpoints:
        ok, elapsed, code = check_endpoint(ep["name"], ep["method"], ep["path"], ep["payload"])
        if ok:
            success_count += 1
            total_time += elapsed
            operational_endpoints.append(ep["name"])
            
    print("\n------------------------------------------------------------")
    print(f"{Colors.BOLD}📊 สรุปผลการตรวจสอบสุขภาพ API:{Colors.END}")
    print(f"  → ทั้งหมด: {success_count}/{len(endpoints)} Endpoints ทำงานได้ตามปกติ")
    if success_count > 0:
        print(f"  → ความเร็วเฉลี่ย (Response Time): {total_time / success_count:.1f} ms")
        
    if success_count < len(endpoints):
        print(f"\n{Colors.YELLOW}⚠️  คำแนะนำ:{Colors.END}")
        print(f"  หากมีหลายระบบขึ้น 🔴 (FAILED/CONNECTION REFUSED):")
        print(f"  1. กรุณาตรวจสอบว่าได้เปิดฐานข้อมูลแล้ว (supabase start)")
        print(f"  2. กรุณาตรวจสอบว่าได้เริ่ม API server แล้ว (START_PHAYAK.bat หรือ python -m src.api.api_server)")
    else:
        print(f"\n{Colors.GREEN}✅ ระบบ API พร้อมใช้งานเต็มรูปแบบ 100%!{Colors.END}")
    print("============================================================\n")

if __name__ == "__main__":
    main()
