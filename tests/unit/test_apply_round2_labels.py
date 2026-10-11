"""รอบสอง: คำตอบของคน → สิ่งที่ต้องเขียนลง DB

ตรรกะอยู่ใน plan_operations ซึ่งไม่แตะ DB จึงทดสอบได้เต็มที่ ส่วนที่เขียนจริงเป็นเปลือกบางๆ ที่รันสิ่งที่
plan_operations คืนมา สิ่งสำคัญคือ **ที่ไหนห้ามเขียน**: ไม่แน่ใจไม่เขียนอะไร และ "ตัวเดียวกัน" ของคู่ในสตอก
ไม่เขียนอะไรเช่นกัน เพราะการรวมคือการถอดสินค้าออกจากสตอก ต้องให้คนเลือกฝั่งที่จะเก็บก่อน
(เคยกดผิดมาแล้ว: ขันน้ำลายไทย 8 CM) ถ้าตีตราว่าตรวจแล้วโดยยังไม่รวม คู่นั้นจะหายจากคิวทั้งที่ยังซ้ำอยู่
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "apply_human_dedup_labels_r2.py"
FIXTURE = REPO / "tests" / "fixtures" / "dedup_human_labels_v2.json"


def load():
    spec = importlib.util.spec_from_file_location("apply_human_dedup_labels_r2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # @dataclass หาโมดูลของตัวเองจาก sys.modules ตอนสร้างคลาส — ไม่ลงทะเบียนจะได้ None แล้วพัง
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plan_operations = load().plan_operations

STOCK = {"imp-1": "stock-1", "imp-2": "stock-2", "imp-3": "stock-3"}


def label(kind, ident, verdict, at="2026-10-11T01:00:00.000Z"):
    return {"kind": kind, "id": ident, "name_a": "ก", "name_b": "ข", "human_verdict": verdict, "labelled_at": at}


class TestImportedProducts:
    def test_same_means_already_in_stock_so_the_product_is_set_aside(self):
        ops = plan_operations([label("import", "imp-1", "same")], STOCK)

        assert ops.product_status == {"imp-1": "rejected"}
        pair = ops.pair_inserts[0]
        assert (pair["product_a_id"], pair["product_b_id"]) == ("imp-1", "stock-1")
        assert pair["is_duplicate"] is True and pair["reviewed"] is True
        assert pair["reviewed_at"] == "2026-10-11T01:00:00.000Z"

    def test_diff_means_new_product_and_records_a_negative_label(self):
        ops = plan_operations([label("import", "imp-2", "diff")], STOCK)

        assert ops.product_status == {"imp-2": "pending_review_category"}
        assert ops.pair_inserts[0]["is_duplicate"] is False
        assert ops.pair_inserts[0]["reviewed"] is True

    def test_unsure_changes_nothing(self):
        ops = plan_operations([label("import", "imp-3", "unsure")], STOCK)

        assert ops.product_status == {} and ops.pair_inserts == [] and ops.pairs_marked_different == {}

    def test_a_product_without_a_recorded_stock_match_is_reported_not_guessed(self):
        ops = plan_operations([label("import", "imp-9", "same")], STOCK)

        assert ops.product_status == {} and ops.pair_inserts == []
        assert ops.skipped == ["imp-9"]


class TestStockPairs:
    def test_diff_is_recorded_as_a_person_checked_different(self):
        ops = plan_operations([label("pair", "m-1", "diff")], STOCK)

        assert ops.pairs_marked_different == {"m-1": "2026-10-11T01:00:00.000Z"}

    def test_same_is_not_written_because_merging_needs_a_person_to_pick_a_side(self):
        ops = plan_operations([label("pair", "m-2", "same")], STOCK)

        assert ops.pairs_marked_different == {}
        assert ops.pair_inserts == [] and ops.product_status == {}
        assert ops.to_merge == ["m-2"]

    def test_unsure_changes_nothing(self):
        ops = plan_operations([label("pair", "m-3", "unsure")], STOCK)

        assert ops.pairs_marked_different == {} and ops.to_merge == []


def test_the_frozen_fixture_is_complete_and_unambiguous():
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert len(rows) == 481
    assert len({row["id"] for row in rows}) == len(rows), "id ซ้ำ — คำตอบของคู่หนึ่งจะทับอีกคู่"
    assert {row["human_verdict"] for row in rows} <= {"same", "diff", "unsure"}
    assert {row["kind"] for row in rows} == {"import", "pair"}
