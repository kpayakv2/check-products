#!/usr/bin/env python3
"""
🔍 ทดสอบผลกระทบจริงของปุ่มใน Web UI ต่อ Backend และ API
===============================================================

ทดสอบว่าการกดปุ่มแต่ละปุ่มจะส่งผลกระทบจริงหรือไม่
"""

import requests
import json
import sqlite3
import os
from pathlib import Path
import time

def test_api_endpoint():
    """ทดสอบ API endpoint /save-feedback"""
    
    print("🔍 ทดสอบ API Endpoint /save-feedback")
    print("=" * 50)
    
    # กำหนด URL
    base_url = "http://localhost:5000"
    endpoint = f"{base_url}/save-feedback"
    
    # ข้อมูลทดสอบ (จำลองการกดปุ่มต่างๆ)
    test_cases = [
        {
            'name': '🔴 ปุ่มสินค้าซ้ำ',
            'data': {
                'id': 'test-001',
                'old_product': 'กะละมัง 549 ดำ SMK',
                'new_product': 'กะละมัง 549 สี SMK', 
                'similarity': 0.772,
                'ml_prediction': 'similar',
                'human_feedback': 'duplicate',  # กดปุ่มแดง
                'comments': 'ทดสอบปุ่มสีแดง',
                'reviewer': 'test_user',
                'timestamp': '2025-09-13T10:00:00Z'
            }
        },
        {
            'name': '🟡 ปุ่มคล้าย แต่ไม่ซ้ำ', 
            'data': {
                'id': 'test-002',
                'old_product': 'แชมพู 200ml A',
                'new_product': 'แชมพู 250ml A',
                'similarity': 0.850,
                'ml_prediction': 'similar',
                'human_feedback': 'similar',  # กดปุ่มเหลือง
                'comments': 'ทดสอบปุ่มสีเหลือง',
                'reviewer': 'test_user', 
                'timestamp': '2025-09-13T10:01:00Z'
            }
        },
        {
            'name': '🟢 ปุ่มต่างกัน',
            'data': {
                'id': 'test-003', 
                'old_product': 'สบู่ขาว 100g',
                'new_product': 'ยาสีฟัน 150g',
                'similarity': 0.150,
                'ml_prediction': 'different',
                'human_feedback': 'different',  # กดปุ่มเขียว
                'comments': 'ทดสอบปุ่มสีเขียว',
                'reviewer': 'test_user',
                'timestamp': '2025-09-13T10:02:00Z'
            }
        },
        {
            'name': '⚪ ปุ่มไม่แน่ใจ',
            'data': {
                'id': 'test-004',
                'old_product': 'ครีมกันแดด SPF30',
                'new_product': 'ครีมกันแดด SPF50', 
                'similarity': 0.920,
                'ml_prediction': 'similar',
                'human_feedback': 'uncertain',  # กดปุ่มเทา
                'comments': 'ทดสอบปุ่มสีเทา',
                'reviewer': 'test_user',
                'timestamp': '2025-09-13T10:03:00Z'
            }
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n📤 ทดสอบ: {test_case['name']}")
        print("-" * 30)
        
        try:
            # ส่ง POST request
            response = requests.post(
                endpoint,
                json=test_case['data'],
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ สำเร็จ: {result.get('message', 'OK')}")
                print(f"📊 Response: {json.dumps(result, ensure_ascii=False, indent=2)}")
                
                results.append({
                    'test': test_case['name'],
                    'status': 'SUCCESS',
                    'response': result
                })
            else:
                print(f"❌ ล้มเหลว: HTTP {response.status_code}")
                print(f"📄 Response: {response.text}")
                
                results.append({
                    'test': test_case['name'], 
                    'status': 'FAILED',
                    'error': f"HTTP {response.status_code}: {response.text}"
                })
                
        except requests.exceptions.ConnectionError:
            print(f"❌ ไม่สามารถเชื่อมต่อกับ {base_url}")
            print(f"💡 กรุณาเริ่มเซิร์ฟเวอร์ด้วย: python web_server.py")
            
            results.append({
                'test': test_case['name'],
                'status': 'CONNECTION_ERROR',
                'error': 'Server not running'
            })
            
        except Exception as e:
            print(f"❌ ข้อผิดพลาด: {str(e)}")
            
            results.append({
                'test': test_case['name'],
                'status': 'ERROR', 
                'error': str(e)
            })
            
        time.sleep(0.5)  # รอระหว่างการทดสอบ
        
    return results

def check_database_impact():
    """ตรวจสอบผลกระทบในฐานข้อมูล human_feedback.db"""
    
    print(f"\n💾 ตรวจสอบฐานข้อมูล human_feedback.db")
    print("=" * 50)
    
    db_path = "human_feedback.db"
    
    if not os.path.exists(db_path):
        print(f"❌ ไม่พบไฟล์ฐานข้อมูล: {db_path}")
        print(f"💡 ฐานข้อมูลจะถูกสร้างเมื่อมีการบันทึก feedback ครั้งแรก")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # ตรวจสอบโครงสร้างตาราง
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='feedback'")
        table_structure = cursor.fetchone()
        
        if table_structure:
            print(f"✅ พบตาราง feedback:")
            print(f"📋 โครงสร้าง: {table_structure[0]}")
        else:
            print(f"❌ ไม่พบตาราง feedback")
            return False
        
        # นับจำนวนข้อมูล
        cursor.execute("SELECT COUNT(*) FROM feedback")
        total_count = cursor.fetchone()[0]
        print(f"📊 จำนวนข้อมูลทั้งหมด: {total_count} รายการ")
        
        # วิเคราะห์ข้อมูลตาม feedback type
        cursor.execute("""
            SELECT human_feedback, COUNT(*) as count
            FROM feedback 
            WHERE human_feedback IS NOT NULL
            GROUP BY human_feedback
            ORDER BY count DESC
        """)
        
        feedback_stats = cursor.fetchall()
        
        print(f"\n📈 สถิติ Human Feedback:")
        print("-" * 30)
        for feedback_type, count in feedback_stats:
            emoji = {'duplicate': '🔴', 'similar': '🟡', 'different': '🟢', 'uncertain': '⚪'}.get(feedback_type, '❓')
            print(f"{emoji} {feedback_type}: {count} รายการ")
        
        # ข้อมูลล่าสุด 5 รายการ
        cursor.execute("""
            SELECT id, product1, product2, similarity_score, ml_prediction, human_feedback, created_at
            FROM feedback
            ORDER BY created_at DESC
            LIMIT 5
        """)
        
        recent_data = cursor.fetchall()
        
        if recent_data:
            print(f"\n📋 ข้อมูลล่าสุด 5 รายการ:")
            print("-" * 70)
            for row in recent_data:
                id_val, prod1, prod2, sim, ml_pred, human_fb, created = row
                emoji = {'duplicate': '🔴', 'similar': '🟡', 'different': '🟢', 'uncertain': '⚪'}.get(human_fb, '❓')
                print(f"{emoji} {id_val}: {prod1[:20]}... vs {prod2[:20]}... ({sim:.1%}) -> {human_fb}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ ข้อผิดพลาดในการอ่านฐานข้อมูล: {str(e)}")
        return False

def check_app_state_impact():
    """ตรวจสอบผลกระทบใน app_state (memory)"""
    
    print(f"\n🧠 ตรวจสอบผลกระทบใน App State (Memory)")
    print("=" * 50)
    
    try:
        # ทดสอบการเรียก API status
        response = requests.get("http://localhost:5000/api/status", timeout=5)
        
        if response.status_code == 200:
            status_data = response.json()
            
            feedback_count = status_data.get('app_state', {}).get('feedback_count', 0)
            print(f"📊 จำนวน feedback ใน memory: {feedback_count} รายการ")
            
            if feedback_count > 0:
                print(f"✅ มีข้อมูล feedback ใน app_state")
                print(f"💾 ข้อมูลจะถูกส่งออกใน CSV export")
            else:
                print(f"⚪ ยังไม่มีข้อมูล feedback ใน app_state")
            
            return True
            
        else:
            print(f"❌ ไม่สามารถเรียก API status ได้: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ ไม่สามารถเชื่อมต่อเซิร์ฟเวอร์")
        print(f"💡 กรุณาเริ่มเซิร์ฟเวอร์ด้วย: python web_server.py")
        return False
        
    except Exception as e:
        print(f"❌ ข้อผิดพลาด: {str(e)}")
        return False

def test_export_impact():
    """ทดสอบผลกระทบต่อการส่งออกข้อมูล"""
    
    print(f"\n📤 ทดสอบผลกระทบต่อการส่งออกข้อมูล")
    print("=" * 50)
    
    try:
        # ทดสอบการส่งออก ML Data
        response = requests.post("http://localhost:5000/export-ml-data", timeout=10)
        
        if response.status_code == 200:
            export_data = response.json()
            
            if export_data.get('success'):
                print(f"✅ การส่งออกข้อมูลสำเร็จ")
                
                # ตรวจสอบไฟล์ที่สร้าง
                files_created = export_data.get('files', {})
                
                for file_type, filename in files_created.items():
                    file_path = f"output/{filename}"
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                        print(f"📄 {file_type}: {filename} ({file_size} bytes)")
                    else:
                        print(f"❌ ไม่พบไฟล์: {filename}")
                        
                # ตรวจสอบว่ามี feedback data ในไฟล์หรือไม่
                feedback_file = files_created.get('human_feedback_file')
                if feedback_file:
                    feedback_path = f"output/{feedback_file}"
                    if os.path.exists(feedback_path):
                        with open(feedback_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            lines = content.strip().split('\n')
                            print(f"📋 Human feedback file มี {len(lines)} บรรทัด (รวม header)")
                            
                            if len(lines) > 1:
                                print(f"✅ มีข้อมูล feedback ในไฟล์ส่งออก")
                            else:
                                print(f"⚪ ไม่มีข้อมูล feedback ในไฟล์ส่งออก")
                        
                return True
            else:
                print(f"❌ การส่งออกล้มเหลว: {export_data.get('message')}")
                return False
                
        else:
            print(f"❌ API ส่งออกข้อมูลล้มเหลว: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ ข้อผิดพลาดในการทดสอบส่งออก: {str(e)}")
        return False

def run_comprehensive_test():
    """รันการทดสอบครบถ้วน"""
    
    print("🎯 การทดสอบผลกระทบของปุ่ม Web UI ต่อ Backend")
    print("=" * 60)
    print(f"⏰ เวลา: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1. ทดสอบ API Endpoint
    api_results = test_api_endpoint()
    
    # 2. ตรวจสอบฐานข้อมูล
    db_impact = check_database_impact()
    
    # 3. ตรวจสอบ App State
    app_state_impact = check_app_state_impact()
    
    # 4. ทดสอบการส่งออก
    export_impact = test_export_impact()
    
    # สรุปผล
    print(f"\n🎯 สรุปผลการทดสอบ")
    print("=" * 60)
    
    total_tests = len(api_results)
    successful_tests = len([r for r in api_results if r['status'] == 'SUCCESS'])
    
    print(f"📊 การทดสอบ API:")
    print(f"  • ทดสอบทั้งหมด: {total_tests} test cases")
    print(f"  • สำเร็จ: {successful_tests} รายการ")
    print(f"  • อัตราสำเร็จ: {(successful_tests/total_tests*100):.1f}%")
    
    print(f"\n📊 การทดสอบส่วนประกอบ:")
    print(f"  • 💾 Database Impact: {'✅ ใช้งานได้' if db_impact else '❌ มีปัญหา'}")
    print(f"  • 🧠 App State Impact: {'✅ ใช้งานได้' if app_state_impact else '❌ มีปัญหา'}")
    print(f"  • 📤 Export Impact: {'✅ ใช้งานได้' if export_impact else '❌ มีปัญหา'}")
    
    # คำแนะนำ
    print(f"\n💡 คำแนะนำ:")
    if successful_tests == total_tests and db_impact and app_state_impact and export_impact:
        print(f"  ✅ ปุ่มใน Web UI ทำงานได้อย่างสมบูรณ์")
        print(f"  📈 มีผลกระทบจริงต่อ Backend และ Database")
        print(f"  💾 ข้อมูล feedback ถูกบันทึกและส่งออกได้")
    else:
        print(f"  ⚠️ มีส่วนประกอบบางอย่างที่ไม่ทำงาน")
        if not db_impact:
            print(f"    - ตรวจสอบการติดตั้ง human_feedback_system.py")
        if not app_state_impact:
            print(f"    - ตรวจสอบการทำงานของ Flask server")
        if not export_impact:
            print(f"    - ตรวจสอบ permission ในการเขียนไฟล์")

if __name__ == "__main__":
    run_comprehensive_test()