# 📦 การย้ายไฟล์เข้า check-products

## 🎯 **ไฟล์ที่ควรย้าย (High Priority)**

### 1. 📁 **output/ → check-products/output/**
**เหตุผล**: โปรเจกต์อ้างอิงโฟลเดอร์ output ใน code หลายจุด
**ไฟล์ที่ได้รับผลกระทบ**:
- `web_server.py` line 1359: `os.makedirs('output', exist_ok=True)`
- `filter_matched_products.py` line 10: `Path(os.getenv("OUTPUT_DIR", "output"))`
- `README.md` line 345: `└── output/` ในโครงสร้างโปรเจกต์

**ไฟล์ใน output/ (37 ไฟล์)**:
```
analysis_summary_*.csv          # รายงานการวิเคราะห์
human_feedback_results_*.csv    # ผลลัพธ์ feedback จากผู้ใช้
products_need_review_*.csv      # สินค้าที่ต้องตรวจสอบ
unique_new_products_*.csv       # สินค้าใหม่ที่ไม่ซ้ำ
approved_products_for_import_*.csv  # สินค้าที่อนุมัติให้นำเข้า
rejected_duplicates_*.csv       # สินค้าที่ปฏิเสธเพราะซ้ำ
```

### 2. 📁 **uploads/ → check-products/uploads/**
**เหตุผล**: Web server อ้างอิงโฟลเดอร์ uploads
**ไฟล์ที่ได้รับผลกระทบ**:
- `web_server.py` line 99: `app.config['UPLOAD_FOLDER'] = 'uploads'`
- `web_server.py` line 108: `os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)`

**ไฟล์ใน uploads/ (3 ไฟล์)**:
```
new_POS__20250727_063658_.csv   # ข้อมูล POS
new_products.xlsx               # สินค้าใหม่
old_cleaned_products.csv        # สินค้าเก่าที่ทำความสะอาดแล้ว
```

### 3. 🗄️ **human_feedback.db → check-products/human_feedback.db**
**เหตุผล**: Database file สำหรับ human feedback system
**ไฟล์ที่ได้รับผลกระทบ**:
- `human_feedback_system.py` อาจอ้างอิง database file
- `web_server.py` ใช้ในการเก็บ feedback ข้อมูล

## 🔄 **ไฟล์ที่อาจย้าย (Medium Priority)**

### 4. 📁 **tests/migration_execution_plan.md → check-products/docs/migration_execution_plan.md**
**เหตุผล**: เป็นเอกสารโปรเจกต์ ควรอยู่ใน docs
**ขนาด**: 1 ไฟล์

## 🚫 **ไฟล์ที่ไม่ควรย้าย**

### 5. 📁 **.venv/** - Python Virtual Environment
**เหตุผล**: Environment-specific, ไม่ควรอยู่ใน version control

### 6. 📁 **.vscode/** - VS Code Settings
**เหตุผล**: IDE-specific settings, อาจเป็น personal preference

### 7. 📁 **.github/** - GitHub Configuration
**เหตุผล**: Repository-level configuration, ควรอยู่ที่ root

## 🛠️ **คำสั่งสำหรับการย้าย (PowerShell)**

```powershell
# เข้าไปยัง directory หลัก
cd "d:\product_checker"

# 1. ย้าย output folder
Move-Item -Path "output" -Destination "check-products\output" -Force
Write-Host "✅ ย้าย output/ เสร็จแล้ว"

# 2. ย้าย uploads folder  
Move-Item -Path "uploads" -Destination "check-products\uploads" -Force
Write-Host "✅ ย้าย uploads/ เสร็จแล้ว"

# 3. ย้าย human_feedback.db
Move-Item -Path "human_feedback.db" -Destination "check-products\human_feedback.db" -Force
Write-Host "✅ ย้าย human_feedback.db เสร็จแล้ว"

# 4. ย้าย migration plan
Move-Item -Path "tests\migration_execution_plan.md" -Destination "check-products\docs\migration_execution_plan.md" -Force
Write-Host "✅ ย้าย migration plan เสร็จแล้ว"

# 5. ลบโฟลเดอร์ tests ที่เหลือ (ถ้าว่างเปล่า)
if ((Get-ChildItem "tests" -Force | Measure-Object).Count -eq 0) {
    Remove-Item "tests" -Force
    Write-Host "✅ ลบโฟลเดอร์ tests ว่างเปล่าแล้ว"
}

Write-Host "🎉 การย้ายไฟล์เสร็จสมบูรณ์!"
```

## 📊 **สรุปผลประโยชน์**

### ✅ **ข้อดี**:
1. **โครงสร้างเดียว**: ไฟล์โปรเจกต์ทั้งหมดอยู่ในโฟลเดอร์เดียว
2. **Relative Paths**: ไม่ต้องใช้ absolute paths ในโค้ด
3. **Deployment ง่าย**: Deploy แค่โฟลเดอร์ check-products
4. **Version Control**: ติดตาม changes ได้ครบถ้วน
5. **Backup ง่าย**: สำรองแค่โฟลเดอร์เดียว

### ⚠️ **ข้อระวัง**:
1. **ต้องอัพเดทโค้ด**: เปลี่ยน paths ที่ hard-coded
2. **ทดสอบหลังย้าย**: ให้แน่ใจว่าทุกฟีเจอร์ยังทำงานได้
3. **Backup ก่อน**: สำรอง folder ปัจจุบันก่อนย้าย

## 🔍 **การตรวจสอบหลังย้าย**

```powershell
# รัน tests เพื่อตรวจสอบ
cd "check-products"
python -m pytest tests/ -v

# รัน web server
python web_server.py

# รัน API server  
python api_server.py

# ทดสอบ main functionality
python main.py --help
```

## 🎯 **สรุป Priority**

| Priority | ไฟล์/โฟลเดอร์ | เหตุผล | Impact |
|----------|-------------|-------|---------|
| 🔴 High | output/ | Code dependencies | Critical |
| 🔴 High | uploads/ | Web server config | Critical |
| 🟡 Medium | human_feedback.db | Database file | Moderate |
| 🟢 Low | migration_execution_plan.md | Documentation | Low |

**แนะนำ**: เริ่มย้าย High Priority ก่อน แล้วทดสอบการทำงาน
