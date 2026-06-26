import sys
import os
import io

# เพิ่มพาธเพื่อให้เรียกใช้โมดูลในโปรเจกต์ได้
sys.path.append(os.getcwd())

from src.core.fresh_implementations import ThaiTextProcessor

def test_dedup_cleaning():
    print("🧪 เริ่มต้นการทดสอบ ThaiTextProcessor (Deduplication Mode)...")
    print("-" * 50)
    
    # 1. ทดสอบโหมดลบช่องว่างทั้งหมด (remove_all_spaces=True)
    processor_dedup = ThaiTextProcessor(remove_all_spaces=True)
    
    test_cases_dedup = [
        ("น้ำดื่ม ตราสิงห์ 500 มล.", "น้ำดื่มตราสิงห์500มล"),
        ("เลย์  รสส้ม   50  กรัม", "เลย์รสส้ม50กรัม"),
        ("กล่องล็อค\u00a0ฝาล็อค\u3000560\u2000มล", "กล่องล็อคฝาล็อค560มล"), # ช่องว่างพิเศษ Unicode
    ]
    
    all_passed = True
    for input_text, expected in test_cases_dedup:
        result = processor_dedup.process(input_text)
        passed = result == expected
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"Input:    {input_text}")
        print(f"Result:   {result}")
        print(f"Expected: {expected}")
        print(f"Status:   {status}")
        print("-" * 50)
        if not passed:
            all_passed = False

    # 2. ทดสอบการคุ้มครองหน่วยวัดสำคัญ (นิ้ว และ เปอร์เซ็นต์)
    test_cases_units = [
        ("พัดลม 16\" สีแดง", "พัดลม16\"สีแดง"),
        ("เครื่องดูดฝุ่น 20\" ประหยัดไฟ 100% ของแท้", "เครื่องดูดฝุ่น20\"ประหยัดไฟ100%ของแท้"),
        ("แอลกอฮอล์ 75% ปริมาณ 450มล", "แอลกอฮอล์75%ปริมาณ450มล"),
    ]

    print("\n🧪 เริ่มต้นการทดสอบการปกป้องหน่วยวัด นิ้ว (\") และเปอร์เซ็นต์ (%)...")
    print("-" * 50)
    for input_text, expected in test_cases_units:
        result = processor_dedup.process(input_text)
        passed = result == expected
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"Input:    {input_text}")
        print(f"Result:   {result}")
        print(f"Expected: {expected}")
        print(f"Status:   {status}")
        print("-" * 50)
        if not passed:
            all_passed = False
            
    if all_passed:
        print("🎉 การทดสอบความถูกต้องขั้นสูงสำเร็จ! ระบบขจัดช่องว่างโดยสิ้นเชิง และปกป้องนิ้ว (\") รวมถึงเปอร์เซ็นต์ (%) ได้สมบูรณ์แบบครับ")
    else:
        print("⚠️ การทดสอบล้มเหลว กรุณาตรวจสอบโค้ดการทำความสะอาดข้อความอีกครั้ง")
        sys.exit(1)

if __name__ == "__main__":
    # แก้ไขปัญหาภาษาไทยบน Windows Console เฉพาะเมื่อรันสคริปต์ตรงๆ เท่านั้น (ไม่ใช่ผ่าน pytest)
    if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        except Exception:
            pass
    test_dedup_cleaning()
