import os

def inspect_file():
    path = r"input/new_product/POS_เพิ่มสินค้า_20250727_063658_จากไฟล์สินค้าใหม่.csv"
    print(f"🔍 กำลังตรวจสอบไฟล์: {path}")
    
    if not os.path.exists(path):
        print("❌ ไม่พบไฟล์!")
        return

    # 1. อ่านแบบ Raw Bytes เพื่อดู BOM
    with open(path, 'rb') as f:
        raw = f.read(100)
        print(f"📦 Raw Bytes (First 100): {raw}")

    # 2. ลองอ่านแบบหลายรหัสภาษา
    encodings = ['utf-8', 'windows-874', 'tis-620', 'utf-16']
    for enc in encodings:
        try:
            with open(path, 'r', encoding=enc) as f:
                content = f.read(500)
                lines = content.splitlines()
                print(f"\n✅ อ่านด้วย {enc} สำเร็จ!")
                print(f"📏 จำนวนบรรทัดที่พบใน 500 ตัวอักษรแรก: {len(lines)}")
                if lines:
                    print(f"📋 หัวตาราง: {lines[0]}")
                    if len(lines) > 1:
                        print(f"📋 บรรทัดแรก: {lines[1]}")
        except Exception as e:
            print(f"❌ อ่านด้วย {enc} พัง: {str(e)}")

if __name__ == "__main__":
    inspect_file()
