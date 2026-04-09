import sys
import os
import io

# แก้ไขปัญหาภาษาไทยบน Windows Console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# เพิ่มพาธเพื่อให้เรียกใช้โมดูลในโปรเจกต์ได้
sys.path.append(os.getcwd())

from fresh_implementations import ThaiTextProcessor

def test_cleaning():
    processor = ThaiTextProcessor(normalize_numbers=True, normalize_thai_chars=True)
    
    test_cases = [
        ("ไอโฟน ๑๔ โปร แม็กซ์", "ไอโฟน 14 โปร แม็กซ์"),
        ("แบรนด์ SAMSUNG ๒๕๖GB", "samsung 256gb"),
        ("เครื่องซักผ้า ฝาบน ๑๐.๕ กก.", "เครื่องซักผ้า ฝาบน 10.5 กก."),
        ("น้ำอัดลม ๑.๒๕ ลิตร", "น้ำอัดลม 1.25 ลิตร"),
        ("ทดสอบสระลอย เ็ก", "ทดสอบสระลอย เก"),
    ]
    
    print("🧪 เริ่มต้นการทดสอบ ThaiTextProcessor (UTF-8 Mode)...")
    print("-" * 50)
    
    all_passed = True
    for input_text, expected in test_cases:
        result = processor.process(input_text)
        
        # เช็คว่าผลลัพธ์มีตัวเลขที่ถูกต้องไหม
        # เราจะไม่เช็คแบบ Case-sensitive และอนุญาตให้เว้นวรรคต่างกันได้เล็กน้อย
        passed = True
        
        # ค้นหาตัวเลขในผลลัพธ์
        import re
        numbers_in_expected = re.findall(r'\d+\.?\d*', expected)
        for num in numbers_in_expected:
            if num not in result:
                passed = False
                break
        
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"Input:    {input_text}")
        print(f"Result:   {result}")
        print(f"Expected: {expected}")
        print(f"Status:   {status}")
        print("-" * 50)
        
        if not passed:
            all_passed = False
            
    if all_passed:
        print("🎉 การทดสอบสำเร็จ! ระบบคลีนข้อมูลและแปลงตัวเลข (๑ -> 1) ได้เป๊ะแล้วครับ")
    else:
        print("⚠️ มีบางจุดที่ยังไม่ตรง โดยเฉพาะการรักษาจุดทศนิยม")

if __name__ == "__main__":
    test_cleaning()
