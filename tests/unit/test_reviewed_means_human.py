"""`reviewed=true` ใน similarity_matches หมายถึงคนตัดสินแล้วเท่านั้น (CONTEXT.md)

บั๊กนี้โผล่มาแล้วสามที่ — ตัวสร้างข้อมูลฝึก, การสแกนสินค้าซ้ำ และ route นำเข้า — เพราะแต่ละที่
เขียน `reviewed: true` ให้การตัดสินด้วย threshold ของตัวเอง แล้วส่งผลสองทาง: ML เอาไปสอนตัวเอง
และคู่ที่ผิดถูกซ่อนจากคนถาวร เทสต์นี้ตรวจแหล่งที่เขียนค่านั้นเป็นค่าคงที่ ถ้ามีที่ใหม่โผล่มา
ต้องมาแก้รายการข้างล่างโดยจงใจ พร้อมเหตุผลว่าทำไมที่นั่นคือคนตัดสิน

ตรวจเฉพาะการเขียนค่า `true` ตรงๆ ไม่ใช่นิพจน์ เช่น `reviewed: item.reviewed_by_user === true`
ซึ่งเปิดทางให้ที่ที่รู้จริงว่าคนตัดสิน
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# ไฟล์ที่เขียน reviewed เป็น true ตรงๆ ได้ เพราะเป็นจุดที่คนกดตัดสินเอง
HUMAN_DECISION_WRITERS = {
    # ปุ่ม merge / keep_both ในหน้า Verify
    "taxonomy-app/app/api/similarity/resolve/route.ts",
    # บันทึกผลที่ผู้ใช้ตัดสินเองจากชุดวัดผลแช่แข็ง
    "scripts/apply_human_dedup_labels.py",
    # ไม่ได้ตัดสินใหม่ — โหมด --revert คืนค่าให้ตรงเดิมพร้อมเวลาเดิมที่เก็บไว้ ไว้ใช้ถอยหลังเท่านั้น
    "scripts/unmark_machine_import_matches.py",
}

SCAN_DIRS = ["scripts", "src", "taxonomy-app/app", "taxonomy-app/utils", "taxonomy-app/components"]

# `reviewed: true,` / `reviewed=True,` / `"reviewed": True,` — ต้องตามด้วย , ) } เพื่อไม่ให้
# จับประโยคอธิบายในคอมเมนต์ที่เขียนว่า reviewed=true แล้วต่อด้วยคำอื่น
WRITE = re.compile(r"""reviewed["']?\s*[:=]\s*(?:True|true)\s*[,)}]""")


def test_only_human_decision_paths_write_reviewed_true():
    offenders = []
    for scan_dir in SCAN_DIRS:
        for path in (REPO / scan_dir).rglob("*"):
            if path.suffix not in {".py", ".ts", ".tsx"} or "node_modules" in path.parts or ".next" in path.parts:
                continue
            relative = path.relative_to(REPO).as_posix()
            if relative in HUMAN_DECISION_WRITERS:
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                # คอมเมนต์ที่เล่าถึง reviewed=true ไม่ใช่การเขียนค่า
                if line.lstrip().startswith(("#", "//", "*", "/*")):
                    continue
                if WRITE.search(line):
                    offenders.append(f"{relative}:{number}: {line.strip()[:90]}")

    assert offenders == [], (
        "เขียน reviewed=true ที่ไม่ใช่เส้นทางที่คนตัดสิน — ถ้าเป็นการตัดสินของเครื่อง "
        "ให้เขียน reviewed=false แล้วเก็บคำตอบของเครื่องไว้ที่ match_type:\n  " + "\n  ".join(offenders)
    )
