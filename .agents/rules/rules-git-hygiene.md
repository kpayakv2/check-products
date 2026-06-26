---
name: rules-git-hygiene
description: |
  กฎการดูแลความสะอาดของ Git repository
  ป้องกันไฟล์ขนาดใหญ่และ Build Artifacts หลุดขึ้น GitHub

triggers:
  - ก่อน git commit หรือ git push ทุกครั้ง
  - เมื่อเพิ่มโมเดล AI หรือ Model Weights
  - เมื่อสร้าง build artifacts (node_modules, .next, __pycache__)
  - เมื่อพบ error ที่เกี่ยวกับ file size ใน GitHub
---

# 🧹 Git Hygiene & Large File Protection

## 🚫 Blocked Content
- **NEVER** commit files larger than 100MB to GitHub
- **Model Weights:** โฟลเดอร์ `model_cache/` ต้องอยู่ใน `.gitignore` เสมอ
- **Build Artifacts:** `node_modules/` และ `.next/` ห้ามหลุดขึ้น Git
- **Binary Files:** ไฟล์นามสกุล `.safetensors`, `.node`, `.exe` ต้องถูกตรวจสอบอย่างเข้มงวด
- **Sensitive Data:** `.env`, `.env.local`, `oauth_creds.json` ห้ามขึ้น Git เด็ดขาด

## ✅ Pre-commit Checklist
```bash
git status          # ดูว่ามีไฟล์ที่ไม่ควร stage ไหม
git diff --stat     # เช็คขนาดไฟล์ที่จะ commit
git log --oneline -5  # ดู commit ล่าสุด
```

**ตรวจ Circular Dependencies (Socraticode) ก่อน push:**
```
codebase_graph_circular { projectPath: "d:\\product_checker\\check-products" }
```
→ หากพบ circular deps ให้แก้ไขก่อน อย่า commit ทับ


## 🛠️ Recovery Action
- หากเผลอ Commit ไฟล์ใหญ่ ให้ใช้ `git rm -r --cached <path>` ทันที
- ใช้ `git reset --soft HEAD~1` เพื่อถอยออกมาแก้ไขก่อน Push
- ตรวจสอบ `git status` และ `git diff --stat` ก่อนก้าวออกจากเครื่องเสมอ
