# 🧹 Git Hygiene & Large File Protection

## 🚫 Blocked Content
- **NEVER** commit files larger than 100MB to GitHub.
- **Model Weights:** โฟลเดอร์ `model_cache/` ต้องอยู่ใน `.gitignore` เสมอ
- **Build Artifacts:** `node_modules/` และ `.next/` ห้ามหลุดขึ้น Git
- **Binary Files:** ไฟล์นามสกุล `.safetensors`, `.node`, `.exe` ต้องถูกตรวจสอบอย่างเข้มงวด

## 🛠️ Recovery Action
- หากเผลอ Commit ไฟล์ใหญ่ ให้ใช้ `git rm -r --cached <path>` ทันที
- ใช้ `git reset --soft HEAD~1` เพื่อถอยออกมาแก้ไขก่อน Push
- ตรวจสอบ `git status` และ `git diff --stat` ก่อนก้าวออกจากเครื่องเสมอ
