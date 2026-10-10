"""สคริปต์บันทึกผลที่คนตัดสินต้องรันโหมดดูก่อนได้กับชุดวัดผลจริง

เคยล้มด้วย KeyError ทั้งที่ยังไม่ได้เขียนฐานข้อมูล เพราะสคริปต์อ่านฟิลด์ที่ถูกเอาออกจากไฟล์
ชุดวัดผลไปแล้ว (คำตอบของกฎไม่ถูกเก็บในไฟล์ เพราะมันเปลี่ยนได้และจะกลายเป็นข้อมูลค้าง)
ถ้าไม่รันจริงทั้งสคริปต์กับไฟล์จริง ความไม่ตรงกันแบบนี้จะไม่มีใครเห็นจนกว่าจะถึงตอนกดเขียน
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "apply_human_dedup_labels.py"


def test_dry_run_reads_the_frozen_fixture_without_writing():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={"PYTHONIOENCODING": "utf-8", "SYSTEMROOT": "C:\\Windows"},
        timeout=60,
    )

    assert result.returncode == 0, result.stderr[-400:]
    assert "88 คู่" in result.stdout
    assert "ไม่ได้เขียนอะไร" in result.stdout
