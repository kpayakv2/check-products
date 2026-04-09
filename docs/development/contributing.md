# Contributing to Product Similarity Checker

เราขอขอบคุณที่สนใจมีส่วนร่วมในการพัฒนาโปรเจกต์นี้! คู่มือนี้จะช่วยให้คุณเริ่มต้นการมีส่วนร่วมได้อย่างราบรื่น

## 🚀 วิธีการเริ่มต้น

### 1. Fork และ Clone Repository

```bash
# Fork repository บน GitHub
# จากนั้น clone มาที่เครื่องของคุณ
git clone https://github.com/YOUR_USERNAME/check-products.git
cd check-products
```

### 2. ตั้งค่า Development Environment

```bash
# สร้าง virtual environment
python -m venv venv

# เปิดใช้งาน virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux  
source venv/bin/activate

# ติดตั้ง dependencies
pip install -r requirements-dev.txt
```

### 3. ตั้งค่า Pre-commit Hooks (แนะนำ)

```bash
pip install pre-commit
pre-commit install
```

## 📝 Code Style และ Standards

### Formatting
เราใช้ Black และ isort สำหรับ code formatting:

```bash
# จัดรูปแบบโค้ด
black .
isort .

# ตรวจสอบก่อน commit
black --check .
isort --check-only .
```

### Naming Conventions
- **ตัวแปร**: `snake_case`
- **ฟังก์ชัน**: `snake_case`
- **คลาส**: `PascalCase`  
- **ค่าคงที่**: `UPPER_SNAKE_CASE`

### Documentation
- เขียน docstring สำหรับฟังก์ชันและคลาสทุกตัว
- ใช้ภาษาอังกฤษสำหรับ code และ comment
- ใช้ภาษาไทยสำหรับเอกสาร user-facing

## 🧪 Testing

### รันการทดสอบ

```bash
# รันทุกการทดสอบ
pytest

# รันพร้อม coverage
pytest --cov=main --cov-report=html

# รันเฉพาะไฟล์เดียว
pytest tests/test_functions.py -v
```

### เขียนการทดสอบใหม่

1. ใส่ไฟล์ทดสอบใน `tests/` directory
2. ตั้งชื่อไฟล์ขึ้นต้นด้วย `test_`
3. ตั้งชื่อฟังก์ชันขึ้นต้นด้วย `test_`
4. ใช้ pytest fixtures สำหรับ setup/teardown

ตัวอย่าง:

```python
import pytest
import pandas as pd
from main import remove_duplicates

def test_remove_duplicates_basic():
    df = pd.DataFrame({"รายการ": ["A", "B", "A"]})
    result, dupes = remove_duplicates(df)
    
    assert len(result) == 2
    assert len(dupes) == 2  # original duplicates
    assert result["รายการ"].tolist() == ["A", "B"]
```

## 🐛 Bug Reports

เมื่อพบ bug กรุณา:

1. ตรวจสอบว่า bug นั้นยังไม่มีใครรายงานใน Issues
2. สร้าง Issue ใหม่พร้อมข้อมูล:
   - คำอธิบายปัญหาอย่างชัดเจน
   - ขั้นตอนการทำซ้ำ (reproduction steps)
   - Expected vs Actual behavior
   - Environment details (Python version, OS, etc.)
   - Error messages หรือ screenshots (ถ้ามี)

### Bug Report Template

```markdown
## Bug Description
[คำอธิบายปัญหาสั้นๆ]

## Steps to Reproduce
1. [ขั้นตอนที่ 1]
2. [ขั้นตอนที่ 2]  
3. [ขั้นตอนที่ 3]

## Expected Behavior
[สิ่งที่คาดหวังว่าจะเกิดขึ้น]

## Actual Behavior  
[สิ่งที่เกิดขึ้นจริง]

## Environment
- Python version: 
- OS: 
- Package versions: (pip freeze output)

## Additional Context
[ข้อมูลเพิ่มเติม screenshots logs ฯลฯ]
```

## ✨ Feature Requests

สำหรับ feature ใหม่:

1. เปิด Issue ประเภท "Feature Request"
2. อธิบายปัญหาที่ feature นี้จะแก้ไข
3. อธิบาย solution ที่เสนอ
4. ระบุ alternatives ที่พิจารณาแล้ว

## 📤 Pull Request Process

### 1. เตรียม Branch

```bash
# สร้าง branch ใหม่สำหรับ feature
git checkout -b feature/your-feature-name

# หรือสำหรับ bugfix
git checkout -b fix/bug-description
```

### 2. ทำการเปลี่ยนแปลง

- เขียนโค้ดตาม style guide
- เพิ่มการทดสอบสำหรับโค้ดใหม่
- อัปเดตเอกสารหากจำเป็น
- ตรวจสอบว่าการทดสอบผ่านทั้งหมด

### 3. Commit

```bash
# ใช้ commit message ที่มีความหมาย
git add .
git commit -m "feat: add product similarity threshold option"

# หรือ
git commit -m "fix: handle empty product name list"
git commit -m "docs: update API documentation"
```

### Commit Message Convention

เราใช้ [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` สำหรับ feature ใหม่
- `fix:` สำหรับ bug fixes  
- `docs:` สำหรับการอัปเดตเอกสาร
- `test:` สำหรับการเพิ่ม/แก้ไขการทดสอบ
- `refactor:` สำหรับ code refactoring
- `style:` สำหรับ formatting changes
- `chore:` สำหรับ maintenance tasks

### 4. Push และสร้าง PR

```bash
# Push branch ไปยัง fork ของคุณ
git push origin feature/your-feature-name

# สร้าง Pull Request บน GitHub
```

### PR Checklist

- [ ] ผ่านการทดสอบทั้งหมด (`pytest`)
- [ ] Code formatting ถูกต้อง (`black --check .`, `isort --check-only .`)
- [ ] มีการทดสอบสำหรับโค้ดใหม่
- [ ] อัปเดตเอกสารหากเปลี่ยนแปลง API
- [ ] Commit messages ตาม convention
- [ ] PR description อธิบายการเปลี่ยนแปลงชัดเจน

## 🏗️ โครงสร้างโปรเจกต์

### ไฟล์หลัก
- `main.py` - ไฟล์หลักสำหรับการประมวลผล
- `run_analysis.py` - สคริปต์วิเคราะห์ผลลัพธ์  
- `clean_csv_products.py` - ทำความสะอาดข้อมูล CSV
- `filter_matched_products.py` - กรองและแยกผลลัพธ์

### Testing Infrastructure
- `test_mocks/` - Mock implementation สำหรับการทดสอบ
  - ⚠️ **ไม่ใช้ใน production** - ใช้เฉพาะใน testing environment
  - มี mock `SentenceTransformer` class และ `cos_sim` function
- `tests/` - Unit tests และ integration tests

### การแยก Production vs Testing
```python
# Production (main.py)
from sentence_transformers import SentenceTransformer  # จาก pip package

# Testing (tests/)
import main  # ใช้ monkeypatch กับ mock implementation
```

## 🏗️ Development Workflow

### Daily Development

```bash
# ดึงการอัปเดตล่าสุดจาก main branch
git checkout main
git pull upstream main

# สร้าง feature branch ใหม่
git checkout -b feature/new-feature

# ทำงาน... แล้ว commit
git add .
git commit -m "feat: implement new feature"

# รันการทดสอบ
pytest

# จัดรูปแบบโค้ด
black .
isort .

# Push และสร้าง PR
git push origin feature/new-feature
```

### การอัปเดต Dependencies

```bash
# อัปเดต package
pip install --upgrade package-name

# อัปเดต requirements file
pip freeze > requirements.txt

# ทดสอบว่ายังทำงานได้
pytest
```

## 🎯 Areas for Contribution

เรายินดีรับความช่วยเหลือในด้านต่างๆ:

### 🔧 Code
- Performance optimization
- Memory usage improvement  
- Additional similarity algorithms
- Better error handling
- CLI enhancements

### 📚 Documentation
- API documentation
- Tutorial และ examples
- Translation ไปภาษาอื่น
- Video tutorials

### 🧪 Testing
- Edge case testing
- Performance testing
- Integration testing
- Test data generation

### 🌟 Features
- Web interface
- Batch processing
- Custom model support
- Configuration files
- Logging improvements

## ❓ ขอความช่วยเหลือ

หากต้องการความช่วยเหลือ:

1. ตรวจสอบ existing Issues และ Discussions
2. สร้าง Issue ใหม่พร้อมแท็ก "question" หรือ "help wanted"
3. อธิบายปัญหาและสิ่งที่คุณลองทำแล้ว

## 📞 Contact

- GitHub Issues: สำหรับ bugs และ feature requests
- GitHub Discussions: สำหรับคำถามทั่วไปและการสอนถาม

ขอบคุณสำหรับการมีส่วนร่วม! 🙏
